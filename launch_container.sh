# launch container with access to GPUs and port for TensorBoard
docker run -i --gpus all -p 6006:6006 -e WANDB_API_KEY -t mnist0 bash
