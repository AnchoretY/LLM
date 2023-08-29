'''
Author: AnchoretY
Date: 2023-07-12 03:29:18
LastEditors: AnchoretY
LastEditTime: 2023-07-13 00:14:15
'''

# 打印模型中全部参数量与可训练参数量信息
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

# 将参数分为学习速率降低与学习速率不变两组
def get_optimizer_grouped_parameters(model,
                                     weight_decay,
                                     no_decay_name_list=[
                                         "bias", "LayerNorm.weight"
                                     ]):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n
                            for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]
    return optimizer_grouped_parameters

# 只在rank0的进程上打印
def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)

# 将字典型的数据全部转化到对应GPU设备上
def to_device(batch,device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.cuda(device)
        except:
            output[k] = v
    return output