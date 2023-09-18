'''
Author: AnchoretY
Date: 2023-07-12 03:16:19
LastEditors: AnchoretY
LastEditTime: 2023-09-18 03:58:15
'''
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel, PeftConfig
from utils.dl_helper import print_trainable_parameters


peft_model_id = "output_dir_lora/"
# 1.读 lora 训练配置文件
config = PeftConfig.from_pretrained(peft_model_id)

# 2. 加载基础模型
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
# 3. 读 lora 
model = PeftModel.from_pretrained(model, peft_model_id)
print_trainable_parameters(model)
# 4. 将LoRA参数合并
model = model.merge_and_unload() 
print_trainable_parameters(model)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

device = torch.device("cuda")
model = model.to(device)
model.eval()


inputs = tokenizer("Tweet text : @HondaCustSvc Your customer service has been horrible during the recall process. I will never purchase a Honda again. Label :", return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=10)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])