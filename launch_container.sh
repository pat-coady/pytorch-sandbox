# launch container with access to GPUs and port for TensorBoard
docker run -i --gpus all -t mnist0 -p 6006:6006 bash
