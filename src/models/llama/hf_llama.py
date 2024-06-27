from utils.hf_import import weight_transform, weight_stack, weight_direct, hf_safetensor_statedict


def state_dict_from_huggingface(model_id):
    return parse_huggingface(hf_safetensor_statedict(model=model_id))


def parse_huggingface(state_dict):
    # Some HF weights were not how I expected them to be. No clue why, but the "transformation" monstrosity fixes it.
    weight_transform(
        input_name="model.layers.{layer}.self_attn.q_proj.{t}",
        transformation=lambda x: x.transpose(0, 1).view(x.size(1), -1, 2, 64).transpose(2, 3).resize(x.size(1), x.size(0)).transpose(0, 1).contiguous(),
    )(state_dict=state_dict)
    weight_transform(
        input_name="model.layers.{layer}.self_attn.k_proj.{t}",
        transformation=lambda x: x.transpose(0, 1).view(x.size(1), -1, 2, 64).transpose(2, 3).resize(x.size(1), x.size(0)).transpose(0, 1).contiguous(),
    )(state_dict=state_dict)

    rules = [
        weight_stack(
            input_names=[
                "model.layers.{layer}.self_attn.q_proj.{t}",
                "model.layers.{layer}.self_attn.k_proj.{t}",
                "model.layers.{layer}.self_attn.v_proj.{t}",
            ],
            output_name="layers.{layer}.attention.block.in_proj.{t}",
        ),
        weight_stack(
            input_names=[
                "model.layers.{layer}.mlp.gate_proj.{t}",
                "model.layers.{layer}.mlp.up_proj.{t}",
            ],
            output_name="layers.{layer}.proj.block.in_proj.{t}",
        ),
        weight_direct(input_name="model.layers.{layer}.self_attn.o_proj.{t}", output_name="layers.{layer}.attention.block.out_proj.{t}"),
        weight_direct(input_name="model.embed_tokens.{t}", output_name="embedding.{t}"),
        weight_direct(input_name="model.norm.{t}", output_name="norm.{t}"),
        weight_direct(input_name="model.layers.{layer}.mlp.down_proj.{t}", output_name="layers.{layer}.proj.block.out_proj.{t}"),
        weight_direct(input_name="model.layers.{layer}.post_attention_layernorm.{t}", output_name="layers.{layer}.proj.norm.{t}"),
        weight_direct(input_name="model.layers.{layer}.input_layernorm.{t}", output_name="layers.{layer}.attention.norm.{t}"),
        weight_direct(input_name="lm_head.{t}", output_name="lm_head.{t}"),
    ]

    result = {}
    for rule in rules:
        result |= rule(state_dict=state_dict)
    return result
