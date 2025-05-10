"""
Main script to train a student model on synthetic CIFAKE data using Knowledge Distillation
with a fine-tuned teacher model for CIFAR-10.
"""
import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Project imports
from models import get_resnet18_model
from utils import (
    CIFAKESyntheticDataset,
    transform_cifar_train,
    transform_cifar_test,
    FILENAME_SUFFIX_TO_LABEL,
    CIFAR10_CLASSES_TUPLE,
    NUM_CLASSES_CIFAR10,
    train_kd_epoch,
    validate_epoch,
    classification_criterion
)
from losses import KnowledgeDistillationLoss
from torchvision.datasets import CIFAR10  # The real test set


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # For reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # --- DataLoaders ---
    print("Loading Synthetic CIFAKE training set for student...")
    synthetic_train_dataset = CIFAKESyntheticDataset(
        root_dir=args.synthetic_data_dir,
        filename_to_label_map=FILENAME_SUFFIX_TO_LABEL,
        transform=transform_cifar_train
    )
    synthetic_train_loader = DataLoader(
        synthetic_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    print(f"Loaded {len(synthetic_train_dataset)} synthetic training images.")

    print("Loading REAL CIFAR-10 test set for student validation...")
    real_cifar10_test_dataset = CIFAR10(
        root=args.real_data_dir,
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
    print(f"Loaded {len(real_cifar10_test_dataset)} real test images.")

    # --- Models ---
    print(f"Loading fine-tuned Teacher Model from: {args.teacher_weights_path}")
    teacher_model = get_resnet18_model(
        num_classes=NUM_CLASSES_CIFAR10,
        weights_path=args.teacher_weights_path,  # Path to fine-tuned teacher
        for_cifar=True
    )
    teacher_model.to(device)
    teacher_model.eval()

    print("Initializing Student Model (ResNet18)...")
    student_model = get_resnet18_model(
        num_classes=NUM_CLASSES_CIFAR10,
        pretrained_on_imagenet=False,  # Student starts from scratch
        for_cifar=True
    )
    student_model.to(device)

    # --- Optimizer, Scheduler, Loss ---
    optimizer_student = optim.SGD(
        student_model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4
    )
    # Adjust milestones based on total epochs for student
    milestone_fracs = [0.5, 0.8]
    milestones = [int(args.epochs * frac) for frac in milestone_fracs]
    scheduler_student = optim.lr_scheduler.MultiStepLR(
        optimizer_student,
        milestones=milestones,
        gamma=0.1
    )
    kd_criterion = KnowledgeDistillationLoss(temperature=args.kd_temp).to(device)

    # --- Training Loop ---
    print(f"\nStarting Knowledge Distillation training for {args.epochs} epochs...")
    train_kd_losses = []
    val_ce_losses = []
    val_accuracies = []
    best_val_accuracy = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")

        epoch_train_kd_loss = train_kd_epoch(
            student_model, teacher_model, synthetic_train_loader,
            optimizer_student, kd_criterion, device, epoch, args.epochs
        )
        train_kd_losses.append(epoch_train_kd_loss)
        print(f"Epoch {epoch} - Training KD Loss: {epoch_train_kd_loss:.4f}")

        epoch_val_ce_loss, epoch_val_accuracy = validate_epoch(
            student_model, real_cifar10_test_loader,
            classification_criterion, device, epoch, args.epochs
        )
        val_ce_losses.append(epoch_val_ce_loss)
        val_accuracies.append(epoch_val_accuracy)
        print(f"Epoch {epoch} - Val CE Loss: {epoch_val_ce_loss:.4f}, Val Acc: {epoch_val_accuracy * 100:.2f}%")

        scheduler_student.step()
        current_lr = optimizer_student.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6f}")

        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            best_epoch = epoch
            if args.save_student_path:
                if not os.path.exists(os.path.dirname(args.save_student_path)):
                    if os.path.dirname(args.save_student_path):
                        os.makedirs(os.path.dirname(args.save_student_path), exist_ok=True)
                torch.save(student_model.state_dict(), args.save_student_path)
                print(
                    f"*** New best val acc: {best_val_accuracy * 100:.2f}%. Student model saved to {args.save_student_path} ***")

    print("\n--- Training Complete ---")
    print(f"Best Validation Accuracy: {best_val_accuracy * 100:.2f}% at epoch {best_epoch}")
    if args.save_student_path:
        print(f"Best student model saved to {args.save_student_path}")

    # --- Plotting ---
    if args.plot_results:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, args.epochs + 1), train_kd_losses, label='Training KD Loss (Synthetic)')
        plt.plot(range(1, args.epochs + 1), val_ce_losses, label='Validation CE Loss (Real)')
        plt.xlabel('Epochs');
        plt.ylabel('Loss');
        plt.title('Loss Curves')
        plt.legend();
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, args.epochs + 1), [acc * 100 for acc in val_accuracies], label='Validation Accuracy (Real)')
        plt.xlabel('Epochs');
        plt.ylabel('Accuracy (%)');
        plt.title('Validation Accuracy Curve')
        plt.legend();
        plt.grid(True);
        plt.ylim(0, max(100, best_val_accuracy * 100 + 5))

        plot_save_path = "training_plots_cifar10.png"
        if args.save_student_path:  # Save plot
            plot_save_path = os.path.join(os.path.dirname(args.save_student_path), "training_plots_cifar10.png")
        plt.savefig(plot_save_path)
        print(f"Training plots saved to {plot_save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Student Model with KD on CIFAKE for CIFAR-10')
    # Data args
    parser.add_argument('--synthetic_data_dir', type=str, required=True,
                        help='Root directory of CIFAKE synthetic training images (train/FAKE)')
    parser.add_argument('--real_data_dir', type=str, default='./data', help='Directory for real CIFAR-10 dataset')
    parser.add_argument('--teacher_weights_path', type=str, required=True,
                        help='Path to fine-tuned teacher model weights (.pth)')
    parser.add_argument('--save_student_path', type=str, default='./student_cifar10_kd.pth',
                        help='Path to save the best student model')

    # Training args
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs for student')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=0.05, help='Initial learning rate for student')
    parser.add_argument('--kd_temp', type=float, default=10.0, help='Temperature for Knowledge Distillation')

    # System args
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--plot_results', action='store_true', help='Whether to generate and save training plots')

    args = parser.parse_args()
    main(args)