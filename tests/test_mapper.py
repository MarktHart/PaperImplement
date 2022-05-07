from transformers import AutoConfig, AutoModel, AutoTokenizer
from implementations.bert.model import Bert
from mapper.map import map_model
import torch
import logging
import pytest


@pytest.fixture
def hf_model():
    config = AutoConfig.from_pretrained("bert-base-uncased")
    smaller_attrs = {
        'hidden_size': 64,
        'intermediate_size': 4 * 64,
        'num_attention_heads': 4,
        'num_hidden_layers': 2,
    }
    for attr, value in smaller_attrs.items():
        setattr(config, attr, value)
    model = AutoModel.from_config(config)
    model.pooler = None
    return model


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture
def dummy_inputs(tokenizer):
    r = tokenizer(["test", "other test"], padding=True)
    return {k: torch.tensor(v) for k, v in r.items()}


@pytest.fixture
def lengths(dummy_inputs):
    return dummy_inputs['attention_mask'].sum(axis=-1)


@pytest.fixture
def model(hf_model):
    l = hf_model.config.num_hidden_layers
    a = hf_model.config.num_attention_heads
    h = hf_model.config.hidden_size
    return Bert(L=l, H=h, A=a, dict_size=hf_model.config.vocab_size)


@pytest.fixture
def clone_inputs(dummy_inputs, lengths):
    return {'x': dummy_inputs['input_ids'], 'lengths': lengths}


def test_map_model_on_bert(caplog, hf_model, model, dummy_inputs, clone_inputs):
    caplog.set_level(logging.DEBUG, logger=map_model.__module__)

    mapping = map_model(original=hf_model, clone=model, original_input=dummy_inputs, clone_input=clone_inputs)
    assert mapping is not None, "No valid mapping found for huggingface Bert and the implementation provided"


def test_map_model_extra_lin_on_bert(caplog, hf_model, model, dummy_inputs, clone_inputs):
    caplog.set_level(logging.DEBUG, logger=map_model.__module__)

    features = model.layers[0].out[0].in_features
    model.layers[0].out = torch.nn.Sequential(torch.nn.Linear(in_features=features, out_features=features, bias=True), *model.layers[0].out)

    mapping = map_model(original=hf_model, clone=model, original_input=dummy_inputs, clone_input=clone_inputs)
    assert mapping is None, "No valid mapping should be found for huggingface Bert and the implementation provided"


def test_map_model_wrong_dropout_p_on_bert(caplog, hf_model, model, dummy_inputs, clone_inputs):
    caplog.set_level(logging.DEBUG, logger=map_model.__module__)

    model.layers[0].out_drop = torch.nn.Dropout(0.2)

    mapping = map_model(original=hf_model, clone=model, original_input=dummy_inputs, clone_input=clone_inputs)
    assert mapping is None, "No valid mapping should be found when a dropout probability is changed"


def test_map_model_wrong_layernorm_on_bert(caplog, hf_model, model, dummy_inputs, clone_inputs):
    caplog.set_level(logging.DEBUG, logger=map_model.__module__)

    model.layers[0].out_norm = torch.nn.LayerNorm((hf_model.config.hidden_size,), eps=1e-6)
    mapping = map_model(original=hf_model, clone=model, original_input=dummy_inputs, clone_input=clone_inputs)
    assert mapping is None, "No valid mapping should be found for huggingface Bert and the implementation provided"
