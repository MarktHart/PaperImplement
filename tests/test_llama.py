import torch

from transformers import AutoTokenizer

from models.llama.model import Llama, LlamaConfig, GenerationConfig
from models.llama.hf_llama import state_dict_from_huggingface


import pytest


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda:0")


@pytest.fixture(scope="module")
def model_id():
    return "meta-llama/Meta-Llama-3-8B-Instruct"


@pytest.fixture(scope="module")
def mapping(model_id):
    return state_dict_from_huggingface(model_id=model_id)


@pytest.fixture(scope="module")
def model(device, mapping):
    with torch.device("meta"):
        model = Llama(LlamaConfig.llama3_8b())
        ours = set(model.state_dict().keys())
        missing = ours - set(mapping.keys())
        assert len(missing) == 0, f"{len(missing)=}\n{missing}"
        model.load_state_dict(mapping, assign=True)
    model.eval().to(device=device)
    return model


@pytest.fixture
def tokenizer(model_id):
    return AutoTokenizer.from_pretrained(model_id)


def test_nop_time_model_mapping(mapping):
    pass


def test_nop_time_model(model):
    pass


@torch.inference_mode
def test_answer_forward(model, tokenizer, device):
    messages = [
        {"role": "system", "content": "Answer the followling multiple choice question with a single capitalized letter."},
        {"role": "user", "content": "Which number is the largest?\nA : 10\nB : 100\nC : 5\nD: 42\n"},
    ]

    prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True
    )

    output = []
    for i in range(8):
        output.append(model.forward(torch.tensor(prompt + output, dtype=torch.int64, device=device)[None, :])[0, -1, :].max(dim=-1).indices.tolist())
        if tokenizer.decode([output[-1]]) == "<|eot_id|>":
            break
    print(tokenizer.decode(prompt) + "\n\n")
    print(tokenizer.decode(prompt + output))
    assert tokenizer.decode([output[0]]) == "B"
    assert tokenizer.decode([output[1]]) == "<|eot_id|>"


@torch.inference_mode
def test_trace(model, tokenizer, device):
    with torch.jit.optimized_execution(False):
        model = torch.jit.trace(model, (torch.arange(21, dtype=torch.int64, device=device).reshape(3, 7),))
        
        for f in [test_answer_forward, test_answer_generate]:
            f(model=model, tokenizer=tokenizer, device=device)
            f(model=model, tokenizer=tokenizer, device=device) # Second, not first, call is when the model is jit compiled


@torch.inference_mode
def test_script(model, tokenizer, device):
    with torch.jit.optimized_execution(False):
        model = torch.jit.script(model, (torch.arange(21, dtype=torch.int64, device=device).reshape(3, 7),))
        
        for f in [test_answer_forward, test_answer_generate]:
            f(model=model, tokenizer=tokenizer, device=device)
            f(model=model, tokenizer=tokenizer, device=device) # Second, not first, call is when the model is jit compiled


@torch.inference_mode
def test_answer_generate(model, tokenizer, device):
    messages = [
        {"role": "system", "content": "Answer the followling multiple choice question with a single capitalized letter."},
        {"role": "user", "content": "Which number is the largest?\nA : 10\nB : 100\nC : 5\nD: 42\n"},
    ]

    prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True
    )
    tokens = torch.tensor(prompt, dtype=torch.int64, device=device)[None, :]
    generation_config = GenerationConfig(max_new_tokens=8)
    output = model.generate(tokens=tokens, config=generation_config).tolist()
    print(output)
    print(tokenizer.decode(output[0]))
    assert tokenizer.decode(output[0][0]) == "B"
    assert output[0][1] == generation_config.stop_token
    assert tokenizer.decode(output[0][1]) == "<|eot_id|>"

