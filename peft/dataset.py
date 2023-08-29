'''
Author: Yhk
Date: 2023-07-16 04:05:10
LastEditors: AnchoretY
LastEditTime: 2023-08-29 00:14:50
Description: 
'''
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset,DataLoader
from datasets import load_dataset

def get_input(samples,tokenizer,prompt_text,max_seq_len):
    input_data = []
    # for prompt,completion in zip(samples['prompt'],samples['completion']):
    for prompt,completion in zip(samples['input'],samples['output']):
        input_data.append("Users:"+prompt_text+prompt+"\nAssistant:"+completion)

    tokens = tokenizer(input_data,
              max_length=max_seq_len,
              padding="max_length",
              truncation=True,
              return_tensors="pt"
            )
    samples['input_ids'] = tokens['input_ids']
    samples['attention_mask'] = tokens['attention_mask'].squeeze(0)
    return samples

class Seq2SeqDataSet(Dataset):
    """数据处理函数"""
    def __init__(self, data_path, tokenizer, max_len, prompt_text,train):
        # data = load_dataset(data_path)
        data = load_dataset(
            "json",
            data_dir=data_path,
            split='train'
        ).train_test_split(test_size=0.1,seed=1234)
        
        data = data.map(
            lambda samples:get_input(samples,tokenizer,prompt_text,max_len),
            batched=True,
        )

        if train:
            self.all_data = data['train']
        else:
            self.all_data = data['test']

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        instance = self.all_data[item]
        return instance
    
def coll_fn(batch):
    input_ids_list, attention_mask_list,label_list = [], [],[]
    for instance in batch:
        input_ids_list.append(instance["input_ids"])
        attention_mask_list.append(instance["attention_mask"])
        label_list.append(instance['input_ids'])

    return {"input_ids": torch.tensor(input_ids_list,dtype=torch.long),
            "attention_mask":  torch.tensor(attention_mask_list,dtype=torch.long),
            "labels":  torch.tensor(label_list,dtype=torch.long)}