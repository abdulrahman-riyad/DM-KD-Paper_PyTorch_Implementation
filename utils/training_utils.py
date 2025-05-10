"""
Training and validation loop utilities.
"""
import torch
import torch.nn as nn
from tqdm import tqdm

# Standard classification criterion for validation
classification_criterion = nn.CrossEntropyLoss()

def train_kd_epoch(student_model, teacher_model, train_loader, optimizer, kd_criterion, device, epoch_num,
                   total_epochs):
    """
    Trains the student model for one epoch using knowledge distillation.
    """
    student_model.train()
    teacher_model.eval()
    running_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(train_loader, desc=f"Train KD Epoch {epoch_num}/{total_epochs}", leave=False)
    for images, _ in progress_bar:
        images = images.to(device)
        optimizer.zero_grad()

        with torch.no_grad():
            teacher_logits = teacher_model(images)

        student_logits = student_model(images)
        loss = kd_criterion(student_logits, teacher_logits)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        progress_bar.set_postfix(kd_loss=running_loss / total_samples if total_samples > 0 else 0.0)

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return epoch_loss


def validate_epoch(model, val_loader, criterion, device, epoch_num, total_epochs):
    """
    Validates the model on a given validation/test set.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(val_loader, desc=f"Validate Epoch {epoch_num}/{total_epochs}", leave=False)
    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)
            val_acc = correct_predictions / total_samples if total_samples > 0 else 0.0
            progress_bar.set_postfix(ce_loss=running_loss / total_samples if total_samples > 0 else 0.0, acc=val_acc)

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    return epoch_loss, epoch_accuracy


def fine_tune_teacher(teacher_model, train_loader, test_loader, ft_epochs, ft_lr, device, model_save_path):
    """
    Fine-tunes a given teacher model on a real dataset.
    """
    print(f"\n--- Starting Teacher Fine-Tuning for {ft_epochs} epochs ---")
    teacher_model.to(device)

    # Unfreeze all parameters for fine-tuning
    for param in teacher_model.parameters():
        param.requires_grad = True

    optimizer_teacher_ft = torch.optim.Adam(teacher_model.parameters(), lr=ft_lr)
    criterion_teacher_ft = nn.CrossEntropyLoss()
    best_teacher_acc = 0.0

    for epoch in range(1, ft_epochs + 1):
        teacher_model.train()
        running_ft_loss = 0.0
        correct_ft_preds = 0
        total_ft_samples = 0

        progress_bar_ft = tqdm(train_loader, desc=f"Teacher FT Epoch {epoch}/{ft_epochs}", leave=False)
        for images, labels in progress_bar_ft:
            images, labels = images.to(device), labels.to(device)
            optimizer_teacher_ft.zero_grad()
            outputs = teacher_model(images)
            loss = criterion_teacher_ft(outputs, labels)
            loss.backward()
            optimizer_teacher_ft.step()

            running_ft_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_ft_preds += torch.sum(preds == labels.data).item()
            total_ft_samples += images.size(0)
            ft_acc = correct_ft_preds / total_ft_samples if total_ft_samples > 0 else 0.0
            progress_bar_ft.set_postfix(loss=(running_ft_loss / total_ft_samples if total_ft_samples > 0 else 0.0),
                                        acc=ft_acc)

        epoch_ft_loss = running_ft_loss / total_ft_samples if total_ft_samples > 0 else 0.0
        epoch_ft_acc = correct_ft_preds / total_ft_samples if total_ft_samples > 0 else 0.0
        print(f"Teacher FT Epoch {epoch}: Train Loss: {epoch_ft_loss:.4f}, Train Acc: {epoch_ft_acc * 100:.2f}%")

        # Validate the fine-tuning teacher
        val_loss, val_acc = validate_epoch(teacher_model, test_loader, criterion_teacher_ft, device, epoch, ft_epochs)
        print(f"Teacher FT Epoch {epoch}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%")

        if val_acc > best_teacher_acc:
            best_teacher_acc = val_acc
            print(f"*** New best teacher val acc: {best_teacher_acc * 100:.2f}%. Saving model to {model_save_path} ***")
            torch.save(teacher_model.state_dict(), model_save_path)

    # Freeze the fine-tuned teacher model and set to eval mode
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    print(
        f"\nTeacher model fine-tuning complete. Best val acc: {best_teacher_acc * 100:.2f}%. Model saved to {model_save_path}")
    print("Teacher model is now frozen and in eval mode.")
    return teacher_model