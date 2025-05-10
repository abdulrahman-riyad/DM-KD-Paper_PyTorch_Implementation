"""
Knowledge Distillation Loss Function.
"""
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    """
    Standard Knowledge Distillation loss using KL Divergence.
    Loss = T^2 * KLDiv(log_softmax(student_logits/T), softmax(teacher_logits/T))
    """

    def __init__(self, temperature, reduction='batchmean'):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.kl_div_loss = nn.KLDivLoss(reduction=reduction)

    def forward(self, student_logits, teacher_logits):
        soft_student_outputs = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher_outputs = F.softmax(teacher_logits / self.temperature, dim=1)

        distillation_loss = self.kl_div_loss(soft_student_outputs, soft_teacher_outputs) * (self.temperature ** 2)
        return distillation_loss