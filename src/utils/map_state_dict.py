import re
import torch


def fstring_reverse(fstring: str, formatted: str) -> dict[str, str] | None:
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


class MappingRule:
    def __init__(self, input_name: str, output_name: str) -> None:
        self.input_name: str = input_name
        self.output_name: str = output_name
        self.applied_inputs: list[str] = []
        self.applied_outputs: list[str] = []

    def apply(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError()

    def parse_kwargs(
        self, state_dict: dict[str, torch.Tensor], fstrings: list[str], fstrings_callbacks: list[list[str]]
    ) -> list[list[str]]:
        kwargs = self.applicable_kwargs(state_dict=state_dict)
        results = [[fstring.format(**kwarg) for fstring in fstrings] for kwarg in kwargs]
        for formatted in results:
            for result, fstring_callback in zip(formatted, fstrings_callbacks):
                fstring_callback.append(result)
        return results

    def applicable_kwargs(self, state_dict: dict[str, torch.Tensor]) -> list[dict[str, str]]:
        return [kwarg for k in state_dict.keys() if (kwarg := fstring_reverse(self.input_name, formatted=k)) is not None]

    def applied(self) -> bool:
        return len(self.applied_inputs) > 0

    @staticmethod
    def validate_rules(rules: list["MappingRule"], all_expected_inputs: list[str], all_expected_outputs: list[str]):
        for rule in rules:
            assert rule.applied(), "Rule {rule} was never applied"
        MappingRule._validate_non_duplicate(all_expected_inputs, name="the expected inputs")
        MappingRule._validate_non_duplicate(all_expected_outputs, name="the expected outputs")

        rule_inputs = [s for rule in rules for s in rule.applied_inputs]
        rule_outputs = [s for rule in rules for s in rule.applied_outputs]
        MappingRule._validate_non_duplicate(rule_inputs, name="the rule inputs")
        MappingRule._validate_non_duplicate(rule_outputs, name="the rule outputs")

        MappingRule._validate_one_to_one(rule_inputs, all_expected_inputs, name="inputs")
        MappingRule._validate_one_to_one(rule_outputs, all_expected_outputs, name="outputs")

    @staticmethod
    def _validate_non_duplicate(strs: list[str], name: str):
        assert len(strs) == len(set(strs)), f"There are duplicates in {name}, namely:\n{[s for s in set(strs) if strs.count(s) > 1]}"

    @staticmethod
    def _validate_one_to_one(strs_a: list[str], strs_b: list[str], name: str):
        missing = set(strs_a) - set(strs_b)
        extra = set(strs_b) - set(strs_a)
        assert len(missing) == 0 and len(extra) == 0, f"The {name} do not map one-to-one.\nMissing:\n{missing}\nExtra:{extra}"

    @staticmethod
    def apply_rules(rules: list["MappingRule"], state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for rule in rules:
            result |= rule.apply(state_dict=state_dict)
        return result


class TransformMappingRule(MappingRule):
    def __init__(self, input_name, transformation) -> None:
        super().__init__(input_name=input_name, output_name=input_name)
        self.transformation = transformation
        self.transformation_used_on: list[str] = []

    def apply(self, state_dict) -> dict[str, torch.Tensor]:
        keys = self.parse_kwargs(state_dict=state_dict, fstrings=[self.input_name], fstrings_callbacks=[self.transformation_used_on])[0]
        for key in keys:
            state_dict[key] = self.transformation(state_dict[key])
        return {}

    def applied(self) -> bool:
        return len(self.transformation_used_on) > 0


class DirectMappingRule(MappingRule):
    def apply(self, state_dict) -> dict[str, torch.Tensor]:
        parsed = self.parse_kwargs(
            state_dict=state_dict,
            fstrings=[self.input_name, self.output_name],
            fstrings_callbacks=[self.applied_inputs, self.applied_outputs],
        )
        return {key_out: state_dict.pop(key_in) for key_in, key_out in parsed}


class StackMappingRule(MappingRule):
    def __init__(self, input_names, output_name) -> None:
        input_name, *extra_input_names = input_names
        super().__init__(input_name=input_name, output_name=output_name)
        self.extra_input_names = extra_input_names

    def apply(self, state_dict):
        parsed = self.parse_kwargs(
            state_dict=state_dict,
            fstrings=[self.input_name] + self.extra_input_names + [self.output_name],
            fstrings_callbacks=[self.applied_inputs for _ in range(len(self.extra_input_names) + 1)] + [self.applied_outputs],
        )

        partly_missing = lambda key_in: any(required not in state_dict for required in key_in)
        for *key_in, key_out in parsed:
            if partly_missing(key_in=key_in):
                for required in key_in:
                    if required in self.applied_inputs:
                        self.applied_inputs.remove(required)
                self.applied_outputs.remove(key_out)
        return {
            key_out: torch.cat([state_dict.pop(key) for key in key_in], dim=0)
            for *key_in, key_out in parsed
            if not partly_missing(key_in=key_in)
        }
