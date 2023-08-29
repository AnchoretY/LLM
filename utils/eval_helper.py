'''
Author: AnchoretY
Date: 2023-07-13 00:16:42
LastEditors: AnchoretY
LastEditTime: 2023-07-19 06:21:37
'''
import torch
from tqdm import tqdm
from .dl_helper import to_device

def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor

def evaluation_ppl(model, eval_dataloader,device):
        model.eval()
        losses = 0
        for step, batch in tqdm(enumerate(eval_dataloader)):
            batch = to_device(batch,device)
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity