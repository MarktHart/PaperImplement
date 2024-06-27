import torch

from transformers import AutoTokenizer

from models.llama.model import Llama, LlamaConfig
from models.llama.hf_llama import state_dict_from_huggingface


def test_answer_8B_Instruct():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    mapping = state_dict_from_huggingface(model_id=model_id)
    with torch.device("meta"):
        model = Llama(LlamaConfig.llama3_8b())
        ours = set(model.state_dict().keys())
        missing = ours - set(mapping.keys())
        assert len(missing) == 0, f"{len(missing)=}\n{missing}"
        model.load_state_dict(mapping, assign=True)

    with torch.inference_mode():
        device = torch.device("cuda:0")
        model.to(device=device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)

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
        for i in range(2):
            output.append(model.forward(torch.tensor(prompt + output, dtype=torch.int64, device=device)[None, :]).float()[0, -1, :].max(dim=-1).indices.tolist())
        print(tokenizer.decode(prompt + output))
        assert tokenizer.decode([output[0]]) == "B"

