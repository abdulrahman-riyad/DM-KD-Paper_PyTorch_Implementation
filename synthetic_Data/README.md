# Synthetic Datasets

This directory is intended to store AI-generated synthetic image datasets used for training student models in the Knowledge Distillation pipeline.

## CIFAKE (CIFAR-10 Based)

For the CIFAR-10 experiments (`main_train_cifar10.py`), you need the synthetic images from the CIFAKE dataset.
1.  Download the dataset from Kaggle: [CIFAKE - Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
2.  From the downloaded archive, extract the images from the `train/FAKE/` directory.
3.  Place these extracted synthetic image files into a subdirectory here, for example:
    `./synthetic_data/cifake_train_fake/`
4.  Update the `--synthetic_data_dir` argument in the `main_train_cifar10.py` script if you use a different path.

## ImageNet-1K Based (Future Work)

For future experiments with ImageNet-1K, synthetic datasets like "Stable Imagenet1k" ([Kaggle Link](https://www.kaggle.com/datasets/vitaliykinakh/stable-imagenet1k)) would be placed in a relevant subdirectory here.

**Note:** The actual image files in this directory (and its subdirectories) are ignored by Git (via `.gitignore`) to keep the repository size manageable. You need to download and place them here as per the instructions.