#!/usr/bin/env bash
# Runs ON a fresh Hostinger GPU box (Ubuntu 24.04, NVIDIA driver + CUDA 12.6 preinstalled, user ubuntu).
# Same library pins as the laptop's cv-rocm venv, CUDA build of torch. ~5 min.
# Measured 2026-10-15 on an RTX 4090 box: torch 2.9.1+cu126, cuda True.
set -e
export DEBIAN_FRONTEND=noninteractive
sudo -E apt-get update -qq
sudo -E apt-get install -y -qq git python3.12-venv python3-pip ffmpeg >/dev/null
python3 -m venv ~/cv-venv
~/cv-venv/bin/pip install -q --upgrade pip
~/cv-venv/bin/pip install -q torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
~/cv-venv/bin/pip install -q ultralytics==8.4.174 opencv-python-headless numpy==2.5.3 scipy==1.18.1 pandas==3.0.6
~/cv-venv/bin/python -c "import torch, ultralytics, cv2; assert torch.cuda.is_available(), 'NO GPU'; print('OK torch', torch.__version__, torch.cuda.get_device_name(0), 'ultralytics', ultralytics.__version__)"
# The container reports 64 cores but its cgroup quota is 16 (cpu.max 1600000/100000): print it so
# run_retries.sh can size threads from the QUOTA, not nproc.
echo "cpu quota: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"
echo SETUP-DONE
