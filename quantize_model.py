"""
Module B — Weight Quantization for LingBot-VA
Shreyas Joshi
"""
import torch
import os
from torchao.quantization import quantize_, Int8WeightOnlyConfig

def apply_quantization(model, config="full"):
    """
    Apply INT8 weight quantization to different parts of the model.
    config options:
    - "full": quantize entire transformer
    - "feedforward": quantize only feedforward layers
    - "attention": quantize only attention projection layers
    """
    if config == "full":
        print(f"Applying INT8 to full model...")
        quantize_(model, Int8WeightOnlyConfig())

    elif config == "feedforward":
        print(f"Applying INT8 to feedforward layers only...")
        for name, module in model.named_modules():
            if any(k in name.lower() for k in ["ffn", "feed_forward", "mlp", "ff."]):
                if isinstance(module, torch.nn.Linear):
                    quantize_(module, Int8WeightOnlyConfig())

    elif config == "attention":
        print(f"Applying INT8 to attention layers only...")
        for name, module in model.named_modules():
            if any(k in name.lower() for k in ["attn", "attention", "qkv", "to_q", "to_k", "to_v"]):
                if isinstance(module, torch.nn.Linear):
                    quantize_(module, Int8WeightOnlyConfig())

    print(f"Quantization config '{config}' applied successfully")
    return model

if __name__ == "__main__":
    print("Quantization module ready")
    print("Configs: full, feedforward, attention")
