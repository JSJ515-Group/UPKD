import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    def __init__(self, temperature=4.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, y_s, y_t, temp=None, unreduce=False):
        temp = self.temperature if temp is None else temp

        if not torch.is_tensor(temp):
            temp = torch.tensor(temp, device=y_s.device)

        p_s = F.log_softmax(y_s / temp, dim=1)
        p_t = F.softmax(y_t / temp, dim=1)

        if unreduce:
            loss = F.kl_div(p_s, p_t, reduction="none").sum(dim=1)
        else:
            loss = F.kl_div(p_s, p_t, reduction="batchmean")

        return loss * temp**2


class UPKDLoss(nn.Module):
    def __init__(
        self,
        alpha=1.0,
        beta=1.0,
        kd_temperature=4.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.kd_temperature = kd_temperature
        self.kd_loss = DistillKL()

    def forward(
        self,
        logit_s,
        logit_t,
        targets,
        feat_s=None,
        feat_t=None,
        temp=None,
        alpha=None,
        beta=None,
        epoch=None,
        warmup_epochs=20,
        feature_scale=5.0,
    ):
        alpha = self.alpha if alpha is None else alpha
        beta = self.beta if beta is None else beta
        temp = self.kd_temperature if temp is None else temp

        loss_ce = F.cross_entropy(logit_s, targets)
        loss_kd = self.kd_loss(logit_s, logit_t, temp)
        loss_feat = torch.tensor(0.0, device=logit_s.device)

        if feat_s is not None and feat_t is not None and isinstance(feat_s, list):
            num_layers = min(len(feat_s), len(feat_t))
            feat_s = feat_s[-num_layers:]
            feat_t = feat_t[-num_layers:]

            losses = []
            weights = []

            for index, (student_feat, teacher_feat) in enumerate(zip(feat_s, feat_t)):
                if (
                    student_feat.dim() == 4
                    and teacher_feat.dim() == 4
                    and student_feat.shape[-2:] != teacher_feat.shape[-2:]
                ):
                    teacher_feat = F.adaptive_avg_pool2d(
                        teacher_feat,
                        student_feat.shape[-2:],
                    )

                weight = (index + 1) / num_layers
                losses.append(F.mse_loss(student_feat, teacher_feat) * weight)
                weights.append(weight)

            loss_feat = sum(losses) / sum(weights)

        if epoch is not None:
            if warmup_epochs <= 0:
                raise ValueError(
                    "warmup_epochs must be greater than zero."
                )

            warmup = min(
                1.0,
                float(epoch + 1) / float(warmup_epochs),
            )

            if not torch.is_tensor(temp):
                temp = torch.tensor(
                    temp,
                    device=logit_s.device,
                )

            temp_ratio = (
                    torch.clamp(
                        temp.detach(),
                        2.0,
                        8.0,
                    )
                    / self.kd_temperature
            )

            feature_weight = torch.clamp(
                1.0 / (temp_ratio + 1e-6),
                0.5,
                2.0,
            )

            loss_feat = (
                    loss_feat
                    * warmup
                    * feature_scale
                    * feature_weight.mean()
            )

        loss = (
                loss_ce
                + alpha * loss_kd
                + beta * loss_feat
        )

        return loss, {
            "ce": loss_ce.item(),
            "kd": loss_kd.item(),
            "feat": loss_feat,
        }