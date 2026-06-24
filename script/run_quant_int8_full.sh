#!/bin/bash
#SBATCH --job-name=lingbot_quant_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=d1029
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=/home/%u/DLES/results/logs/%j_quant_full.out
#SBATCH --error=/home/%u/DLES/results/logs/%j_quant_full.err

module purge
module load anaconda3/2024.06 cuda/12.8.0
eval "$(micromamba shell hook --shell bash)"
micromamba activate /scratch/joshi.shreyas/lingbot_env/lingbot

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "from curobo.types.math import Pose; print('curobo ok')"

cd ~/DLES/lingbot-va
QUANT_CONFIG=full python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port 29061 \
    wan_va/wan_va_server_quant.py \
    --config-name robotwin \
    --port 29056 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"
echo "Waiting for server to load..."
sleep 150

echo "========== adjust_bottle =========="
cd ~/DLES/lingbot-va
PYTHONWARNINGS=ignore::UserWarning \
python -m evaluation.robotwin.eval_polict_client_openpi \
    --config task_config/demo_clean.yml \
    --overrides \
    --task_name adjust_bottle \
    --task_config demo_clean \
    --ckpt_setting quant_full \
    --seed 0 \
    --policy_name LingBotVA \
    --save_root ~/DLES/results/quant_full \
    --port 29056 \
    --test_num 16 \
    --video_guidance_scale 5 \
    --action_guidance_scale 1

echo "========== open_microwave =========="
PYTHONWARNINGS=ignore::UserWarning \
python -m evaluation.robotwin.eval_polict_client_openpi \
    --config task_config/demo_clean.yml \
    --overrides \
    --task_name open_microwave \
    --task_config demo_clean \
    --ckpt_setting quant_full \
    --seed 0 \
    --policy_name LingBotVA \
    --save_root ~/DLES/results/quant_full \
    --port 29056 \
    --test_num 16 \
    --video_guidance_scale 5 \
    --action_guidance_scale 1

echo "========== click_bell =========="
PYTHONWARNINGS=ignore::UserWarning \
python -m evaluation.robotwin.eval_polict_client_openpi \
    --config task_config/demo_clean.yml \
    --overrides \
    --task_name click_bell \
    --task_config demo_clean \
    --ckpt_setting quant_full \
    --seed 0 \
    --policy_name LingBotVA \
    --save_root ~/DLES/results/quant_full \
    --port 29056 \
    --test_num 16 \
    --video_guidance_scale 5 \
    --action_guidance_scale 1

kill $SERVER_PID
echo "Quant full done!"
