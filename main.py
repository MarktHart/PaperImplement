from transformers import AutoModel, AutoTokenizer
from implementations.bert.model import Bert
from mapper.map import map_model
import torch

def main():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    dummy_inputs = tokenizer(["test", "other test"], padding=True)
    dummy_inputs = {k: torch.tensor(v) for k, v in dummy_inputs.items()}
    lengths = dummy_inputs['attention_mask'].sum(axis=-1)

    mapping = map_model(original=model, clone=Bert.base(dict_size=model.config.vocab_size), original_input=dummy_inputs, clone_input={'x': dummy_inputs['input_ids'], 'lengths': lengths})
    print(mapping)
    

if __name__ == "__main__":
    main()
