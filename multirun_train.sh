# launch hydra multirun job to demonstrate W&B and TensorBoard
python3 train.py -m data.num_workers=4 trainer.max_epochs=20 \
model.l1_chan=4,8 model.l2_chan=8,16 model.l3_chan=16,32 \
model.optimizer_params.lr=0.02,0.1 model.optimizer_params.momentum=0.8,0.9 \
model.optimizer_params.nesterov=0,1