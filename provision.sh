#!/usr/bin/env bash

# gcloud alpha compute ssh a100-8 --zone=us-central1-c --project=tpu-pytorch -- -o ProxyCommand='corp-ssh-helper %h %p'
# gcloud alpha compute tpus tpu-vm ssh yifeit-v5e --zone=us-west1-c --project=tpu-pytorch -- -o ProxyCommand='corp-ssh-helper %h %p'

# Run this on each VM
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
cd $HOME
mkdir -p resnet18
cd resnet18
uv python install 3.10.14
uv venv
source .venv/bin/activate
uv pip install -U "ray[data,train,tune,serve]"
PYTHON_VERSION=$(python3 --version)
if [[ $PYTHON_VERSION != *"3.10.14"* ]]; then
    echo "Python version is not 3.10.14. Please check your Python installation."
    exit 1
fi

uv pip install --prerelease=allow torch~=2.6.0 'torch_xla[tpu]~=2.6.0' -f https://storage.googleapis.com/libtpu-releases/index.html -f https://storage.googleapis.com/libtpu-wheels/index.html
uv pip install -U --pre jax==0.4.38 jaxlib==0.4.38 libtpu==0.0.7.1 requests -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
uv pip install 'torchax @ git+https://git@github.com/pytorch/xla.git#subdirectory=torchax'

# Then:

# On GPU:
ray start --head --port=6379

# On TPU:
ray start --address=$GPU_IP:6379
# ray start --address=10.128.0.32:6379

# Setup port forwarding
gcloud alpha compute ssh a100-8 --zone=us-central1-c --project=tpu-pytorch -- -o ProxyCommand='corp-ssh-helper %h %p' -L 8265:localhost:8265
