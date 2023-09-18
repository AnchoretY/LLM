'''
Author: AnchoretY
Date: 2023-07-10 22:25:59
LastEditors: AnchoretY
LastEditTime: 2023-09-18 04:06:16
'''
"""
deepspeed finetune_prefix_tuning.py  \
    --num_train_epochs 1  \
    --zero_stage 2  \
    --num_virtual_tokens 30 \
    --model_dir /home/yhk/github/DeepSpeed/facebook/opt-350m/
"""
from transformers import AutoModelForCausalLM,AutoTokenizer,get_scheduler,SchedulerType
import torch
import deepspeed
import argparse
import math
from torch.utils.data import RandomSampler, DataLoader,SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from dataset import Seq2SeqDataSet,coll_fn
import os
from shutil import copy

from peft import (
    LoraConfig, 
    PrefixTuningConfig,
    get_peft_model,
    TaskType
)
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from utils.ds_utils import get_train_ds_config
from utils.dl_helper import get_optimizer_grouped_parameters,to_device,print_rank_0,print_trainable_parameters
from utils.eval_helper import evaluation_ppl

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='/home/yhk/github/DeepSpeed/llm_dataset/CodeInstruction/', type=str, help='')
    parser.add_argument('--model_dir', default="/home/yhk/github/DeepSpeed/lmsys/vicuna-7b-v1.3/", type=str, help='')
    parser.add_argument('--num_train_epochs', default=5, type=int, help='')
    parser.add_argument('--train_batch_size', default=2, type=int, help='')
    parser.add_argument('--local_rank', default=-1, type=int, help='local_rank for distributed training on gpus')
    parser.add_argument('--offload',action='store_true',help='Enable ZeRO Offload techniques.')
    parser.add_argument('--zero_stage',type=int,default=0,help='ZeRO optimization stage for Actor model (and clones).')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='')
    parser.add_argument('--output_dir', default='output_dir/', type=str, help='')
    parser.add_argument('--log_steps', type=int, default=10, help='')
    parser.add_argument('--max_len', type=int, default=768, help='')
    parser.add_argument("--lr_scheduler_type",type=SchedulerType,default="cosine",help="The scheduler type to use.",choices=["linear", "cosine", "cosine_with_restarts", "polynomial","constant", "constant_with_warmup"],)
    parser.add_argument("--weight_decay",type=float,default=0.,help="Weight decay to use.")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate (after the potential warmup period) to use.')
    parser.add_argument("--num_warmup_steps",type=int,default=0,help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument('--max_src_len', type=int, default=450, help='')
    parser.add_argument('--num_virtual_tokens', type=int, default=30, help='Prefix Token nums in ')
    parser.add_argument('--prompt_text', type=str,
                        default="You are a bug checker, I need you to check out the bugs in this code:",
                        help='')
    return parser.parse_args()

def main():
    args = set_args()
    # -1表示不是用分布式，其他值则表示分布在指定GPU上
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()
    # 获得进程编号
    args.global_rank = torch.distributed.get_rank()

    # 配置deepspeed配置
    ds_conf = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage)
    
    # 固定随机和种子
    # set_random_seed(args.seed)
    
    # 等待所有进程执行完成
    torch.distributed.barrier()
    
    # ------------------- 正式开始训练 -------------------
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    # 将pad_token也设置为截止符
    tokenizer.pad_token = tokenizer.eos_token
    print("model and tokenizer load completed!")

    # 将模型改为prefix-tune结构 
    prefix_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        num_virtual_tokens=args.num_virtual_tokens)

    model = get_peft_model(model, prefix_config) # 这里使用lora配置后，会自动冻结其余部分参数，只有lora部分可训练
    # 打印可训练层和参数数量
    print_trainable_parameters(model)

    # 训练数据准备
    train_dataset = Seq2SeqDataSet(args.train_path, tokenizer, args.max_len, args.prompt_text,True)
    eval_dataset = Seq2SeqDataSet(args.train_path, tokenizer, args.max_len, args.prompt_text,False)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=ds_conf["train_micro_batch_size_per_gpu"],
                                  sampler=train_sampler,
                                  collate_fn=coll_fn)
    eval_dataloader = DataLoader(eval_dataset,
                                batch_size=ds_conf["train_micro_batch_size_per_gpu"],
                                sampler=eval_sampler,
                                collate_fn=coll_fn)
    
    print("Dataset load completed!")

     # 参数切分成两组，一组需要优化速率权重衰减，另一组不需要
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))
    
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    
    # 学习效率调度器
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )


    # deepspeed模型、优化器等、学习速率调度器初始化
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(config=ds_conf,
                                                         model=model,
                                                         optimizer=optimizer,
                                                         lr_scheduler=lr_scheduler,
                                                         dist_init_required=True # 使用分布式训练初始化
                                                         )


    print("-"*10+"Train"+"-"*10)
    print_rank_0("***** Running training *****", args.global_rank)
    print_rank_0(
        f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)
    perplexity = evaluation_ppl(model_engine, eval_dataloader,device)
    print_rank_0(f"ppl: {perplexity}", args.global_rank)

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        
        model_engine.train()
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch,device)
            outputs = model_engine(**batch, use_cache=False)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()

        # Evaluate perplexity on the validation set.
        print_rank_0(
            f"***** Evaluating perplexity, Epoch {epoch+1}/{args.num_train_epochs} *****",
            args.global_rank)
        perplexity = evaluation_ppl(model_engine, eval_dataloader,device)
        print_rank_0(f"ppl: {perplexity}", args.global_rank)
        model_engine.tput_timer.update_epoch_count()

    # 模型存储
    if args.output_dir is not None:
        print_rank_0('saving the final model ...', args.global_rank)

        save_path = os.path.join(args.output_dir,"prefix_tune")
        if args.global_rank == 0:
            model_engine.module.save_pretrained(save_path)

        if args.zero_stage == 3:
            # 对于zero stage 3，由于模型也被切分到了多个GPU上，因此需要专门的模型存储函数
            print_rank_0("Zero stage lora save unsported!!!")
    
if __name__ == "__main__":
    main()
    
    