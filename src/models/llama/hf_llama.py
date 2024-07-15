from utils.map_state_dict import StackMappingRule, DirectMappingRule, MappingRule


def hf_mapping_rules() -> list[MappingRule]:
    return [
        StackMappingRule(
            input_names=[
                "model.layers.{layer}.self_attn.q_proj.{t}",
                "model.layers.{layer}.self_attn.k_proj.{t}",
                "model.layers.{layer}.self_attn.v_proj.{t}",
            ],
            output_name="layers.{layer}.attention.block.in_proj.{t}",
        ),
        StackMappingRule(
            input_names=[
                "model.layers.{layer}.mlp.gate_proj.{t}",
                "model.layers.{layer}.mlp.up_proj.{t}",
            ],
            output_name="layers.{layer}.proj.block.in_proj.{t}",
        ),
        DirectMappingRule(
            input_name="model.layers.{layer}.self_attn.o_proj.{t}", output_name="layers.{layer}.attention.block.out_proj.{t}"
        ),
        DirectMappingRule(input_name="model.embed_tokens.{t}", output_name="embedding.{t}"),
        DirectMappingRule(input_name="model.norm.{t}", output_name="norm.{t}"),
        DirectMappingRule(input_name="model.layers.{layer}.mlp.down_proj.{t}", output_name="layers.{layer}.proj.block.out_proj.{t}"),
        DirectMappingRule(input_name="model.layers.{layer}.post_attention_layernorm.{t}", output_name="layers.{layer}.proj.norm.{t}"),
        DirectMappingRule(input_name="model.layers.{layer}.input_layernorm.{t}", output_name="layers.{layer}.attention.norm.{t}"),
        DirectMappingRule(input_name="lm_head.{t}", output_name="lm_head.{t}"),
    ]
