'''
Author: AnchoretY
Date: 2023-08-14 10:20:42
LastEditors: AnchoretY
LastEditTime: 2023-08-14 10:26:17
'''
from huggingface_hub import snapshot_download

repo_id = "bigscience/bloomz-7b1-mt"
local_dir = "/Volumes/Extreme_SSD/LLM model/bigscience/bloomz-7b1-mt"
cache_dir = local_dir

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    cache_dir=cache_dir,
    ignore_patterns=["*.h5", "*.ot", "*.msgpack"],
)




