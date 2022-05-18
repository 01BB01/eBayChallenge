import torch.nn as nn


def get_configured_parameters(module, base_lr=None, weight_decay=0.01, lr_multiplier=1):
    # module param can either be a nn.Module or in some cases can also be
    # a list of named parameters for a nn.Module
    if isinstance(module, nn.Module):
        param_optimizer = list(module.named_parameters())
    elif isinstance(module, list):
        param_optimizer = module

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []

    for n, p in param_optimizer:
        param_dict = {}
        param_dict["params"] = [p]
        param_dict["lr"] = base_lr * lr_multiplier
        if not any(nd in n for nd in no_decay):
            param_dict["weight_decay"] = weight_decay
        else:
            param_dict["weight_decay"] = 0.0
        optimizer_grouped_parameters.append(param_dict)
    return optimizer_grouped_parameters
