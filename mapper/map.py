from itertools import permutations
import torch
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def map_model(original, clone, original_input, clone_input):
    original.eval()
    clone.eval()
    original_modules = named_lower_modules(original)
    clone_modules = named_lower_modules(clone)
    cached_original_inputs = create_caching(original_modules)
    cached_clone_inputs = create_caching(clone_modules)
    expected_original_order = expected_order(original_modules)
    
    original_output = original(**original_input)
    stop_condition = 1e12
    mapping = {}

    while len(cached_original_inputs) < stop_condition:
        stop_condition = len(cached_original_inputs)
        clone_output = clone(**clone_input)
        for module, ins in cached_original_inputs.items():
            if any(equal(ins, clone_ins) and type(module) == type(clone_module) for clone_module, clone_ins in cached_clone_inputs.items()):
                break
        
        original_matches = [k for k,v in cached_original_inputs.items() if equal(ins, v) and type(k) == type(module)]
        clone_matches = [k for k,v in cached_clone_inputs.items() if equal(ins, v) and type(k) == type(module)]

        if len(clone_matches) == 0:
            continue

        if len(clone_matches) != len(original_matches):
            logger.warning("Number of suitable modules for clone and original do not match")
            continue

        if len(clone_matches) == 1:
            clone_match = clone_matches[0]
            original_match = original_matches[0]
            clone_str = clone_modules[clone_match]
            original_str = original_modules[original_match]

            logger.info(f"{original_str} -> {clone_str}")
            mismatched_parameters = [(param, value, vars(clone_match)[param]) for param, value in vars(original_match).items() if not param.startswith("_") and value != vars(clone_match)[param]]
            if len(mismatched_parameters) > 0:
                class_str = str(type(clone_match)).split('.')[-1][:-2]
                for param, original_value, clone_value in mismatched_parameters:
                    logger.warning(f"Found mismatch in {class_str} (clone.){clone_str}.{param} = {clone_value}, while {original_str}.{param} = {original_value}")
            clone_match.load_state_dict(original_match.state_dict())
            mapping[clone_str] = original_str
            del cached_original_inputs[original_match]
        elif len(clone_matches) > 1:
            assert len(clone_matches) < 6
            
            previous_match_order = [None] * len(clone_matches)

            for match in original_matches:
                del cached_original_inputs[match]
            
            for match_order in permutations(clone_matches):
                changed_matches = [(a, b) for a, b, c in zip(original_matches, match_order, previous_match_order) if b != c]
                previous_match_order = match_order

                logger.debug(f"{[original_modules[m] for m in original_matches]} ->? {[clone_modules[m] for m in match_order]}")

                for a, b in changed_matches:
                    b.load_state_dict(a.state_dict())

                clone_output = clone(**clone_input)
                if any(
                        type(module) == type(clone_module) and equal(ins, clone_ins)
                        for module, ins in cached_original_inputs.items()
                        for clone_module, clone_ins in cached_clone_inputs.items()
                        if clone_module not in clone_matches and module not in original_matches
                    ):
                    for original_module, clone_module in zip(original_matches, match_order):
                        clone_str = clone_modules[clone_module]
                        original_str = original_modules[original_module]
                        mapping[clone_str] = original_str
                    break
    
    if len(clone_modules) > len(mapping):
        logger.warning(f"Mapping exited with {len(cached_original_inputs)} original modules and {len(clone_modules) - len(mapping)} clone modules left")
        probable_module = [m for m in expected_original_order if m in cached_original_inputs][0]
        logger.warning(f"Expected mismatch around {original_modules[probable_module]} \n\t\t\tinput shape: {cached_original_inputs[probable_module].shape})")
    else:
        logger.info("Mapping done!")
    return mapping

def equal(a, b):
    if type(a) == tuple and len(a) == 1:
        return equal(a[0], b)

    if type(b) == tuple and len(b) == 1:
        return equal(a, b[0])
    
    if type(a) == type(b) == tuple:
        return len(a) == len(b) and all(equal(m,n) for m,n in zip(a,b))
    else:
        return type(a) == type(b) and a.shape == b.shape and torch.allclose(a, b, atol=1e-10)

def named_lower_modules(model):
    module_names = {name for name, _ in model.named_modules()}
    return {module: name for name, module in model.named_modules() if name != "" and not any(key != name and key.startswith(name + ".") for key in module_names)}

def create_caching(modules):
    cache = {}
    def hook(module, ins, _):
        assert len(ins) == 1
        cache[module] = ins[0].detach().clone()
    for module in modules:
        module.register_forward_hook(hook)
    return cache

def expected_order(modules):
    order = []
    def hook(module, _ins, _):
        order.append(module)
    for module in modules:
        module.register_forward_hook(hook)
    return order