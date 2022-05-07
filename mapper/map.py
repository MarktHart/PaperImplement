import itertools
import torch
import logging

logger = logging.getLogger(__name__)


def map_model(original, clone, original_input, clone_input):
    original.train()
    clone.train()
    original_modules = named_lower_modules(original)
    clone_modules = named_lower_modules(clone)

    original_parameters = [k for k in original_modules if next(k.parameters(), None) is not None]
    clone_parameters = [k for k in clone_modules if next(k.parameters(), None) is not None]

    if not verify_learned_parameter_types(original_parameters, clone_parameters):
        return

    original_non_parameters = [k for k in original_modules if next(k.parameters(), None) is None]
    clone_non_parameters = [k for k in clone_modules if next(k.parameters(), None) is None]
    matching_non_learned_types = extra_non_learned_parameters(original_non_parameters, clone_non_parameters)

    original_parameters = [k for k in original_modules if k in original_parameters or type(k) in matching_non_learned_types]
    clone_parameters = [k for k in clone_modules if k in clone_parameters or type(k) in matching_non_learned_types]

    cached_original_param_inputs = hook_result_caching(original_parameters)
    cached_clone_param_inputs = hook_result_caching(clone_parameters)
    expected_original_order = hook_expected_order(original_parameters)

    torch.manual_seed(11)
    original(**original_input)

    def clone_forward():
        torch.manual_seed(11)
        clone(**clone_input)
        return cached_original_param_inputs, cached_clone_param_inputs
    mapping = {}
    previous_len_mapping = -1

    while len(clone_parameters) > len(mapping) > previous_len_mapping:
        previous_len_mapping = len(mapping)
        match = find_matching_parameter(clone_forward)

        if match is None:
            logger.warning("No matching inputs found between clone and original")
            break

        match_module, match_input = match

        all_original_matches = [k for k, v in cached_original_param_inputs.items() if type(k) == type(match_module) and equal(match_input, v)]
        all_clone_matches = [k for k, v in cached_clone_param_inputs.items() if type(k) == type(match_module) and equal(match_input, v)]

        if len(all_clone_matches) != len(all_original_matches):
            logger.warning("Number of suitable modules for clone and original do not match")
            break

        if len(all_clone_matches) == 1:
            if not map_single_match(all_clone_matches[0], all_original_matches[0]):
                return

            mapping[all_clone_matches[0]] = all_original_matches[0]
            del cached_original_param_inputs[all_original_matches[0]]

        elif len(all_clone_matches) > 1:
            assert len(all_clone_matches) < 6

            match_order = map_multiple_matches(all_clone_matches, all_original_matches, clone_forward, )
            if match_order is None:
                logger.warning(f"No valid permutations found in: {[m._name for m in all_original_matches]} -x> {[m._name for m in all_clone_matches]}")
                break

            for original_match, clone_match in zip(all_original_matches, match_order):
                mapping[clone_match] = original_match
                del cached_original_param_inputs[original_match]

    if len(clone_parameters) > len(mapping):
        logger.warning(f"Mapping exited with {len(cached_original_param_inputs)} original modules and {len(clone_parameters) - len(mapping)} clone modules left")
        probable_module = [m for m in expected_original_order if m in cached_original_param_inputs][0]
        logger.warning(f"Expected mismatch around {original_modules[probable_module]} \n\t\t\tinput shape: {cached_original_param_inputs[probable_module].shape})")
    else:
        logger.info("Mapping done!")
        return mapping


def equal(a, b):
    if type(a) == tuple and len(a) == 1:
        return equal(a[0], b)

    if type(b) == tuple and len(b) == 1:
        return equal(a, b[0])

    if type(a) == type(b) == tuple:
        return len(a) == len(b) and all(equal(m, n) for m, n in zip(a, b))
    else:
        return type(a) == type(b) and a.shape == b.shape and torch.allclose(a, b, atol=1e-10)


def named_lower_modules(model):
    r = {module: name for name, module in model.named_modules(remove_duplicate=False)
            if all(m == module for _, m in module.named_modules())}
    for k, v in r.items():
        k._name = v
    return list(r.keys())


def hook_result_caching(modules):
    cache = {}

    def hook(module, ins, _):
        assert len(ins) == 1
        cache[module] = ins[0].detach().clone()
    for module in modules:
        module.register_forward_hook(hook)
    return cache


def hook_expected_order(modules):
    order = []

    def hook(module, _ins, _):
        order.append(module)
    for module in modules:
        module.register_forward_hook(hook)
    return order


def count_types(modules):
    types = list(map(type, modules))
    return {t: types.count(t) for t in set(types)}


def verify_learned_parameter_types(original, clone):
    original_counts = count_types(original)
    clone_counts = count_types(clone)
    if original_counts != clone_counts:
        logger.warning(f"Types of modules to be cloned do not match")
        for t in set(list(original_counts.keys()) + list(clone_counts.keys())):
            original_count = original_counts.get(t, 0)
            clone_count = clone_counts.get(t, 0)
            if original_count != clone_count:
                logger.warning(f"{t}: (original) {original_count} != {clone_count} (clone)")
                logger.debug(f"Original: {[module._name for module in original if type(module) == t]}")
                logger.debug(f"Clone: {[module._name for module in clone if type(module) == t]}")

    return original_counts == clone_counts


def extra_non_learned_parameters(original, clone):
    original_counts = count_types(original)
    clone_counts = count_types(clone)
    r = []
    for t in set(list(original_counts.keys()) + list(clone_counts.keys())):
        original_count = original_counts.get(t, 0)
        clone_count = clone_counts.get(t, 0)
        if original_count == clone_count:
            r.append(t)

    logger.debug(f"Appending matching types with: {r}")
    return r


def find_matching_parameter(clone_forward):
    original, clone = clone_forward()
    for module, ins in original.items():
        if any(type(module) == type(clone_module) and equal(ins, clone_ins) for clone_module, clone_ins in clone.items()):
            return (module, ins)


def map_single_match(clone, original):
    logger.info(f"{original._name} -> {clone._name}")
    mismatched_parameters = [(param, value, vars(clone)[param]) for param, value in vars(original).items()
                                if not param.startswith("_") and value != vars(clone)[param]]

    if len(mismatched_parameters) > 0:
        class_str = str(type(clone)).split('.')[-1][:-2]
        for param, original_value, clone_value in mismatched_parameters:
            logger.warning(f"Found mismatch in {class_str} (clone.){clone._name}.{param}={clone_value}, while {original._name}.{param}={original_value}")
        return False

    clone.load_state_dict(original.state_dict())
    return True


def map_multiple_matches(all_clone_matches, all_original_matches, clone_forward):
    previous_match_order = all_clone_matches

    for match_order in itertools.permutations(all_clone_matches):
        changed_matches = [(a, b) for a, b, c in zip(all_original_matches, match_order, previous_match_order) if b != c]
        previous_match_order = match_order

        logger.debug(f"{[m._name for m in all_original_matches]} ->? {[m._name for m in match_order]}")

        for a, b in changed_matches:
            b.load_state_dict(a.state_dict())

        original_cached_inputs, clone_cached_inputs = clone_forward()
        if any(
                type(module) == type(clone_module) and equal(ins, clone_ins)
                for module, ins in original_cached_inputs.items()
                for clone_module, clone_ins in clone_cached_inputs.items()
                if clone_module not in all_clone_matches and module not in all_original_matches):

            logger.debug(f"{[m._name for m in all_original_matches]} ->! {[m._name for m in match_order]}")
            return match_order
