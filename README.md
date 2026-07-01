# LingBot-VA INT8 Quantization

Selective INT8 weight quantization of the **LingBot-VA** video-action transformer policy, evaluated on the **RoboTwin 2.0** manipulation benchmark. This project investigates *which transformer components are precision-sensitive* by comparing feedforward-only, attention-only, and full-model quantization against a BF16 baseline.

## Key Finding

Feedforward layers tolerate INT8 quantization well, while attention layers are more precision-sensitive. Feedforward-only quantization offers the most favorable efficiency–accuracy tradeoff: **~42–47% lower inference latency** and **~10% less VRAM** with minimal impact on task success.

## Results

Evaluated over 50 episodes per configuration. Discriminative tasks (`open_microwave`, `turn_switch`) are reported as mean ± std over 3 random seeds; ceiling tasks (`adjust_bottle`, `click_bell`) over a single seed.

### Task Success Rate

| Config | adjust_bottle | click_bell | open_microwave | turn_switch |
|---|---|---|---|---|
| Baseline (BF16) | 98% | 100% | 60% ± 2% | 61% ± 2% |
| INT8 FF-Only | 96% | 100% | 56% ± 2% | 53% ± 5% |
| INT8 Full | 100% | 100% | 59% ± 3% | 56% ± 9% |
| INT8 Attn-Only | 96% | 100% | 48% ± 4% | 53% ± 6% |

### Efficiency (open_microwave)

| Config | Latency | VRAM | Speedup | VRAM Reduction |
|---|---|---|---|---|
| Baseline (BF16) | 11.30 s | 30.41 GB | — | — |
| INT8 FF-Only | 6.55 s | 27.36 GB | 42% | 10% |
| INT8 Full | 7.80 s | 25.10 GB | 31% | 17% |
| INT8 Attn-Only | 7.19 s | 27.72 GB | 36% | 9% |

Efficiency metrics are highly consistent across seeds (latency std < 0.1 s).

## Repository Structure

```
quantize_model.py                              # INT8 quantization of the WAN-VA transformer
wan_va/wan_va_server_quant.py                  # Quantized inference server (selective configs + metric logging)
evaluation/robotwin/eval_polict_client_openpi.py  # Evaluation client with per-episode checkpoint/resume
script/run_quant_int8_full.sh                  # Full-model INT8 run
script/run_quant_int8_ff.sh                    # Feedforward-only INT8 run
script/run_quant_int8_attn.sh                  # Attention-only INT8 run
```

## Quantization Configurations

The quantized server selects which components to quantize via the `QUANT_CONFIG` environment variable:

| Value | Quantizes |
|---|---|
| `feedforward` | Feedforward / MLP blocks only |
| `attention` | Attention blocks only |
| `full` | All linear layers |

## Requirements

- Python 3.10
- PyTorch 2.7.1 (CUDA 12.6)
- torchao (INT8 quantization)
- CUDA 12.8, FFmpeg 7.1.1
- RoboTwin 2.0 environment + assets
- NVIDIA GPU with ≥ 30 GB VRAM (developed on A100 80GB)

## Usage

Set the quantization mode and launch the server, then run the evaluation client. Example (feedforward-only):

```bash
QUANT_CONFIG=feedforward python -m torch.distributed.run \
    --nproc_per_node 1 --master_port 29061 \
    wan_va/wan_va_server_quant.py --config-name robotwin --port 29056 &

python -m evaluation.robotwin.eval_polict_client_openpi \
    --config task_config/demo_clean.yml --overrides \
    --task_name open_microwave --task_config demo_clean \
    --ckpt_setting quant_ff --seed 0 --policy_name LingBotVA \
    --save_root ./results/quant_ff --port 29056 --test_num 50
```

The provided `script/run_quant_int8_*.sh` files wrap this for SLURM clusters.

### Resuming Interrupted Runs

Evaluation writes a `checkpoint.json` (`last_seed`, `succ_seed`) after each episode. To resume an interrupted run, pass the next seed via `--st_seed`:

```bash
python -m evaluation.robotwin.eval_polict_client_openpi ... --st_seed <last_seed+1> --test_num <remaining>
```

## Takeaways

- **Efficiency gains are large and reliable.** Latency drops 31–47% and VRAM drops 9–17% across all quantization configs, with negligible variance across seeds.
- **Simple tasks are robust to quantization.** On ceiling tasks (`adjust_bottle`, `click_bell`), all configs retain 96–100% success — quantization can be applied freely.
- **Layer sensitivity differs.** Feedforward-only quantization consistently preserves accuracy better than attention-only, making it the best efficiency–accuracy tradeoff for complex tasks.

## Limitations & Future Work

- **Statistical scope.** Accuracy differences on complex tasks are small relative to seed variance (±2–9%). The efficiency results and the feedforward-vs-attention trend are the most robust claims; finer accuracy distinctions would need more episodes per seed.
- **Single model, simulation only.** Results are on LingBot-VA in RoboTwin 2.0. Generalization to other video-action models and to real-robot deployment is open.
- **Next steps.** Extend to additional models and tasks for generalizability; add TensorRT-optimized deployment for stronger real-world latency gains; analyze *why* attention layers are more precision-sensitive (e.g. activation range, softmax sensitivity).

## Acknowledgments

Built on the LingBot-VA and RoboTwin 2.0 codebases. Compute provided by the Northeastern University Discovery cluster as well;
## License

Licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details. Base components (LingBot-VA, RoboTwin 2.0) remain under their respective upstream licenses.
