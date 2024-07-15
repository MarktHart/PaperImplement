import os

import torch

from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from utils.map_state_dict import MappingRule


def hf_safetensor_state_dict(model_id, device, ignore_pattern="IGNORE"):
    ext = ".safetensors"
    download_folder = snapshot_download(repo_id=model_id, allow_patterns=f"*{ext}", ignore_patterns=f"*{ignore_pattern}*")

    safe_tensors = [os.path.join(download_folder, f) for f in os.listdir(download_folder) if f.endswith(ext) and ignore_pattern not in f]

    for file in safe_tensors:
        yield load_file(file, device=str(device))


def hf_fast_model_init(model_id, mapping_rules, model_class, config, device):
    with torch.device("meta"):
        model = model_class(config=config)
        expected_keys = list(model.state_dict().keys())

    hf_keys = []
    leftover_state_dict = {}
    for partial_state_dict in hf_safetensor_state_dict(model_id=model_id, device=device):
        hf_keys.extend(list(partial_state_dict.keys()))
        leftover_state_dict |= partial_state_dict
        model.load_state_dict(MappingRule.apply_rules(rules=mapping_rules, state_dict=leftover_state_dict), assign=True, strict=False)
    MappingRule.validate_rules(rules=mapping_rules, all_expected_inputs=hf_keys, all_expected_outputs=expected_keys)
    model.eval().to(device=device)
    return model
