# LingBot-VA Quantization

INT8 quantization experiments on [LingBot-VA](https://github.com/Robbyant/lingbot-va), a vision-action model for robot manipulation tasks. Evaluated across 16 episodes per task on a A100 GPU (HPC Clusters).

## My Contributions

- `quantize_model.py` — INT8 quantization script for the WAN-VA model
- `wan_va/wan_va_server_quant.py` — Quantized inference server
- `run_quant_int8_full.sh` — Run full INT8 quantization
- `run_quant_int8_ff.sh` — Run feed-forward only INT8 quantization
- `run_quant_int8_attn.sh` — Run attention only INT8 quantization

## Results

### Task Success Rate (16 episodes each)

| Configuration     | adjust_bottle | open_microwave | click_bell |
|------------------|---------------|----------------|------------|
| Baseline (BF16)  | 15/16 = 93.8% | 7/16 = 43.8%   | 16/16 = 100% |
| INT8 Full        | 16/16 = 100%  | 5/16 = 31.2%   | 16/16 = 100% |
| INT8 FF-Only ⭐  | 16/16 = 100%  | 14/16 = 87.5%  | 16/16 = 100% |
| INT8 Attn-Only   | 15/16 = 93.8% | 9/16 = 56.2%   | 16/16 = 100% |

### Efficiency Metrics

| Configuration    | Peak VRAM | Avg Latency/Chunk | VRAM Reduction | Speedup |
|-----------------|-----------|-------------------|----------------|---------|
| Baseline (BF16) | 30.41 GB  | ~10.3s            | —              | —       |
| INT8 Full       | 25.19 GB  | ~7.5s             | -17.2%         | +27%    |
| INT8 FF-Only ⭐ | 27.32 GB  | ~5.3s             | -10.2%         | +49%    |
| INT8 Attn-Only  | 27.62 GB  | ~5.7s             | -9.2%          | +45%    |

⭐ **INT8 FF-Only is the recommended config** — best balance of speed (+49%), memory efficiency, and task success rate.

## How to Run

```bash
# Baseline
bash run_baseline.sh

# Full INT8 quantization
bash run_quant_int8_full.sh

# Feed-forward only (recommended)
bash run_quant_int8_ff.sh

# Attention only
bash run_quant_int8_attn.sh
```

## Base Repository

Built on top of [LingBot-VA](https://github.com/Robbyant/lingbot-va).# LingBot-VA-Quantization
