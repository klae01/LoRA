#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn

from typing import Dict

from .layers import Conv2d, Embedding, Linear, LoRALayer


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = "none") -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == "none":
        return {k: my_state_dict[k] for k in my_state_dict if "lora_" in k}
    elif bias == "all":
        return {
            k: my_state_dict[k] for k in my_state_dict if "lora_" in k or "bias" in k
        }
    elif bias == "lora_only":
        to_return = {}
        for k in my_state_dict:
            if "lora_" in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def convert_to_lora(
    module: nn.Module, r: int = 0, lora_alpha: int = 1, lora_dropout: float = 0
) -> nn.Module:
    factory_kwargs = lambda: {
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "merge_weights": False,
        "device": module.weight.data.device,
        "dtype": module.weight.data.dtype,
    }
    training = module.training
    module_output = module
    if type(module) is nn.Linear:
        module_output = Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            **factory_kwargs(),
        )
        module_output.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            module_output.bias.data.copy_(module.bias.data)
        module_output.train(training)
    elif type(module) is nn.Conv2d:
        module_output = Conv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            **factory_kwargs(),
        )
        module_output.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            module_output.bias.data.copy_(module.bias.data)
        module_output.train(training)
    elif type(module) is nn.Embedding:
        module_output = Embedding(
            module.num_embeddings,
            module.embedding_dim,
            padding_idx=module.padding_idx,
            max_norm=module.max_norm,
            norm_type=module.norm_type,
            scale_grad_by_freq=module.scale_grad_by_freq,
            sparse=module.sparse,
            **factory_kwargs(),
        )
        module_output.weight.data.copy_(module.weight.data)
        module_output.train(training)

    for name, child in module.named_children():
        module_output.add_module(
            name, convert_to_lora(child, r, lora_alpha, lora_dropout)
        )

    return module_output


def convert_from_lora(module: nn.Module) -> nn.Module:
    training = module.training
    factory_kwargs = lambda: {
        "device": module.weight.data.device,
        "dtype": module.weight.data.dtype,
    }
    module_output = module
    if type(module) is Linear:
        module_output = nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            **factory_kwargs(),
        )
        module.eval()
        module_output.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            module_output.bias.data.copy_(module.bias.data)
        module_output.train(training)
    elif type(module) is Conv2d:
        module_output = nn.Conv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            **factory_kwargs(),
        )
        module.eval()
        module_output.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            module_output.bias.data.copy_(module.bias.data)
        module_output.train(training)
    elif type(module) is Embedding:
        module_output = nn.Embedding(
            module.num_embeddings,
            module.embedding_dim,
            padding_idx=module.padding_idx,
            max_norm=module.max_norm,
            norm_type=module.norm_type,
            scale_grad_by_freq=module.scale_grad_by_freq,
            sparse=module.sparse,
            **factory_kwargs(),
        )
        module.eval()
        module_output.weight.data.copy_(module.weight.data)
        module_output.train(training)

    for name, child in module.named_children():
        module_output.add_module(name, convert_from_lora(child))

    return module_output
