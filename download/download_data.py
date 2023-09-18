from datasets import load_dataset

data_repo = "stanfordnlp/SHP"
save_path = "/Volumes/Extreme_SSD/LLM dataset/stanfordnlp/SHP"
dataset = load_dataset(data_repo)
dataset.save_to_disk(save_path)

