"""
Data loading utilities for CIFAR-10 and synthetic CIFAKE datasets.
"""
import os
import re
from glob import glob
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# --- CIFAR-10 Constants ---
NUM_CLASSES_CIFAR10 = 10
CIFAR10_CLASSES_TUPLE = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Mapping from CIFAKE filename suffix to 0-indexed label
# Example: 'xxxx.jpg' or 'xxxx (1).jpg' for airplane (index 0), 'xxxx (2).jpg' for automobile (index 1)
FILENAME_SUFFIX_TO_LABEL = {
    '': 0,  # Default for first class if no number in suffix
    '(1)': 0,
    '(2)': 1,
    '(3)': 2,
    '(4)': 3,
    '(5)': 4,
    '(6)': 5,
    '(7)': 6,
    '(8)': 7,
    '(9)': 8,
    '(10)': 9,
}

# Standard CIFAR-10 normalization statistics
MEAN_CIFAR10 = [0.4914, 0.4822, 0.4465]
STD_CIFAR10 = [0.2023, 0.1994, 0.2010]

# --- Transformations for CIFAR-10 (32x32 images) ---
transform_cifar_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_CIFAR10, std=STD_CIFAR10),
])

transform_cifar_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_CIFAR10, std=STD_CIFAR10),
])


class CIFAKESyntheticDataset(Dataset):
    """
    Custom Dataset for loading CIFAKE synthetic images.
    Parses filenames to determine class labels.
    """

    def __init__(self, root_dir, filename_to_label_map, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.filename_to_label_map = filename_to_label_map
        self.image_paths = []
        self.labels = []
        # Regex to capture base filename and optional class suffix like (2)
        self.filename_pattern = re.compile(r"^\d+(?: \((?P<suffix_num>\d+)\))?\.(png|jpg|jpeg)$", re.IGNORECASE)
        self._load_samples()

    def _load_samples(self):
        if not os.path.isdir(self.root_dir):
            raise ValueError(f"Root directory not found: {self.root_dir}")

        image_files = glob(os.path.join(self.root_dir, "*.[jp][pn]g"))
        if not image_files:
            print(f"Warning: No image files found in {self.root_dir}.")
            return

        for img_path in image_files:
            filename = os.path.basename(img_path)
            match = self.filename_pattern.match(filename)

            if match:
                suffix_num_str = match.group('suffix_num')
                label_key = f"({suffix_num_str})" if suffix_num_str else ''

                if label_key in self.filename_to_label_map:
                    label = self.filename_to_label_map[label_key]
                    self.image_paths.append(img_path)
                    self.labels.append(label)

        if not self.image_paths:
            print("CRITICAL WARNING: No images processed. Check paths, patterns, and map.")
        else:
            print(f"Loaded {len(self.image_paths)} synthetic images from {self.root_dir}.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise e