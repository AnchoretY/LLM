prefix-tuning  virtual-tokens30  trainable params: 1474560 || all params: 332670976 || trainable%: 0.4432487672143662

prefix-tuning  virtual-tokens10   trainable params: 491520 || all params: 331687936 || trainable%: 0.14818748186246966





使用peft库中包含的高效微调手段并配合 deepspeed 对 LLM 进行高效微调：

- finetune_lora.py
- finetune_prefix_tuning.py
- finetune_prompt_tuning.py
- finetune_ptuning.py


使用训练好的 lora 进行推理

- lora_inference.py
