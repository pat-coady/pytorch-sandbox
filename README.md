# PyTorch + Lightning + W&B + TensorBoard Reference Implementation

It is finally time to jump in the PyTorch pool. My work closed up shop for 2 weeks over the holidays, so this was a perfect time to play around with some new (to me) tools.

The key objective was to build a reference training implementation for a simple CNN for MNIST. From here, the plan is to implement and a few more interesting things and get the rust knocked off:

- Diffusion models
- Very small GPT-like model (e.g., nanoGPT)
- Low-bit quantization in PyTorch
- Graph neural networks

## Components of Reference Implementation

I've included a few more modern approaches to things since when I 

- PyTorch Lightning : Get some feeling for what I like and what I don't.
- YAML configuration (Hydra): Move away from CLI arguments to YAML configuration.
- Weights & Biases : See what a slick experiment tracking tool can do for me.
- TensorBoard : Still seems the go-to for basic training montoring.
- Dev Containers : Move from PyCharm to VS Code and give dev containers a try.
- TODO: Add type hinting to my code.

## Acknowledgements

I'd like to thank Tao Yu and Zsolt Majzik on my team for giving me some really helpful pointers as I got started.