"""
Functions to provide ResNet models (teacher/student) based on torchvision.
"""
import torch
import torch.nn as nn
from torchvision import models

def get_resnet18_model(num_classes, pretrained_on_imagenet=False, weights_path=None, for_cifar=True):
    """
    Gets a ResNet18 model, optionally loads ImageNet weights, adapts for num_classes.
    If for_cifar=True, it expects 32x32 input and uses a simpler first layer if no weights are loaded.
    If weights_path is provided, it loads those weights.
    """
    if pretrained_on_imagenet and not weights_path:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif weights_path: # Loading custom pre-trained weights (fine-tuned teacher)
        model = models.resnet18(weights=None, num_classes=num_classes) # Ensure correct num_classes for saved weights
    else: # Randomly initialized for student or CIFAR-specific training
        model = models.resnet18(weights=None, num_classes=num_classes)

    if for_cifar:
        # Standard torchvision ResNets have a large first conv and a maxpool designed for ImageNet.
        # For CIFAR (32x32), we adjust:
        # 1. Smaller kernel/stride for conv1
        # 2. Potentially remove initial MaxPool
        if not pretrained_on_imagenet and not weights_path:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # CIFAR-style conv1
            model.maxpool = nn.Identity()

    # Adapt the final fully connected layer if its input features don't match,
    # or if the number of classes in loaded weights is different from target num_classes.
    if model.fc.out_features != num_classes:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    if weights_path:
        print(f"Loading weights from: {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

    return model