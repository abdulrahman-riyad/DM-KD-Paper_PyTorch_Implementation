# Real Datasets

This directory is intended to store real datasets used for training teacher models or for evaluating student models.

## CIFAR-10

The `fine_tune_teacher_cifar10.py` and `main_train_cifar10.py` scripts expect the real CIFAR-10 dataset.
If you run these scripts, PyTorch's `torchvision.datasets.CIFAR10` will automatically download the CIFAR-10 dataset into a subdirectory here (e.g., `./data/cifar-10-batches-py/`) if it's not found.

## Other Datasets (e.g., ImageNet Validation)

If experiments are extended to other real datasets (like ImageNet for validation), they would typically be downloaded or linked here according to the respective script's instructions.

**Note:** The actual image files in this directory are ignored by Git (via `.gitignore`) to keep the repository size manageable. You need to run the scripts to download them or place them here manually if you have them already.