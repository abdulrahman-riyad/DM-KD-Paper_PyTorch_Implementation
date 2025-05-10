"""
Script to fine-tune a ResNet18 model (pre-trained on ImageNet) on the real CIFAR-10 dataset.
The fine-tuned model will serve as the teacher for Knowledge Distillation.
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from models import get_resnet18_model
from utils import transform_cifar_train, transform_cifar_test, fine_tune_teacher, NUM_CLASSES_CIFAR10

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Real CIFAR-10 Data ---
    print("Loading REAL CIFAR-10 training set for fine-tuning...")
    real_cifar10_train_dataset = CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform_cifar_train
    )
    real_cifar10_train_loader = DataLoader(
        real_cifar10_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    print(f"Loaded REAL CIFAR-10 training set with {len(real_cifar10_train_dataset)} samples.")

    print("Loading REAL CIFAR-10 test set for validation during fine-tuning...")
    real_cifar10_test_dataset = CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform_cifar_test
    )
    real_cifar10_test_loader = DataLoader(
        real_cifar10_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    print(f"Loaded REAL CIFAR-10 test set with {len(real_cifar10_test_dataset)} samples.")

    # --- Initialize Teacher Model (ImageNet pre-trained) ---
    print("Initializing ImageNet pre-trained ResNet18 for fine-tuning...")
    teacher_to_finetune = get_resnet18_model(
        num_classes=NUM_CLASSES_CIFAR10,
        pretrained_on_imagenet=True,  # Start with ImageNet weights
        for_cifar=True
    )
    teacher_to_finetune.to(device)

    # --- Fine-Tune ---
    fine_tuned_teacher = fine_tune_teacher(
        teacher_model=teacher_to_finetune,
        train_loader=real_cifar10_train_loader,
        test_loader=real_cifar10_test_loader,
        ft_epochs=args.epochs,
        ft_lr=args.lr,
        device=device,
        model_save_path=args.save_path
    )
    print(f"Teacher fine-tuning complete. Fine-tuned model saved to {args.save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune Teacher Model for CIFAR-10')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for CIFAR-10 dataset')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for fine-tuning')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for fine-tuning')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--save_path', type=str, default='./teacher_cifar10_finetuned.pth',
                        help='Path to save the fine-tuned teacher model')

    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.save_path)):
        if os.path.dirname(args.save_path):
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    main(args)