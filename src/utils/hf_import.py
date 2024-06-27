import os
import re

import torch

from huggingface_hub import snapshot_download
from safetensors.torch import load_file


def hf_safetensor_statedict(model):
    ext = ".safetensors"
    download_folder = snapshot_download(repo_id=model, allow_patterns=f"*{ext}")

    safe_tensors = [os.path.join(download_folder, f) for f in os.listdir(download_folder) if f.endswith(ext)]

    params = {}

    for file in safe_tensors:
        params |= load_file(file)
    return params


def fstring_reverse(fstring, formatted):
    reg = r"\{(.+?)\}"
    parsed_rule = re.split(reg, fstring)
    keys = parsed_rule[1::2]
    values_regex = "^" + "(.+?)".join(parsed_rule[::2]) + "$"
    matches = re.match(values_regex, formatted)
    values = matches.groups() if matches is not None else ()
    if len(values) != len(keys):
        return None
    kwargs = {k: v for k, v in zip(keys, values)}
    if fstring.format(**kwargs) != formatted:
        return None
    return kwargs


def weight_transform(input_name, transformation):
    def inner(state_dict):
        kwargs = [kwarg for k in state_dict.keys() if (kwarg := fstring_reverse(input_name, formatted=k)) is not None]
        assert len(kwargs) > 0
        for kwarg in kwargs:
            state_dict[input_name.format(**kwarg)] = transformation(state_dict[input_name.format(**kwarg)])
    return inner


def weight_direct(input_name, output_name):
    def inner(state_dict):
        new_state_dict = {}
        kwargs = [kwarg for k in state_dict.keys() if (kwarg := fstring_reverse(input_name, formatted=k)) is not None]
        assert len(kwargs) > 0
        for kwarg in kwargs:
            new_state_dict[output_name.format(**kwarg)] = state_dict[input_name.format(**kwarg)]
        return new_state_dict
    return inner


def weight_stack(input_names, output_name):
    def inner(state_dict):
        new_state_dict = {}
        kwargs = [kwarg for k in state_dict.keys() if (kwarg := fstring_reverse(input_names[0], formatted=k)) is not None]
        assert len(kwargs) > 0
        for kwarg in kwargs:
            new_state_dict[output_name.format(**kwarg)] = torch.cat([state_dict[input_name.format(**kwarg)] for input_name in input_names], dim=0)
        return new_state_dict
    return inner

