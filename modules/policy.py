import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyAgent(nn.Module):

    def __init__(self, state_dim=8, hidden_dim=128):
        super().__init__()

        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, state):

        logits = self.policy(state)

        concentration = F.softplus(logits) + 1e-4

        dist = torch.distributions.Dirichlet(concentration)

        weights = dist.sample()

        log_prob = dist.log_prob(weights)

        entropy = dist.entropy()

        ratio = weights[:, 0:1]  # A single component is sufficient.
        return ratio, log_prob, entropy


def build_state(feat_s, feat_t, logit_s, logit_t, targets,temp):

    with torch.no_grad():
        batch_size = feat_s.size(0)

        feat_gap = torch.abs(feat_s - feat_t).mean(dim=1, keepdim=True)

        logit_gap = torch.abs(logit_s - logit_t).mean(dim=1, keepdim=True)

        s_conf = F.softmax(logit_s, 1).max(1)[0].unsqueeze(1)

        t_conf = F.softmax(logit_t, 1).max(1)[0].unsqueeze(1)

        ce_loss = F.cross_entropy(logit_s, targets, reduction='none').unsqueeze(1)

        if temp.dim() == 0:  # Scalar temperature.
            temp_expanded = temp.expand(batch_size, 1)
        elif temp.dim() == 1:  # Shape: [batch_size].
            if temp.size(0) == batch_size:
                temp_expanded = temp.unsqueeze(1)
            else:
                temp_expanded = temp.expand(batch_size, 1)
        elif temp.dim() == 2:  # Shape: [batch_size, 1].
            if temp.size(0) == batch_size:
                temp_expanded = temp
            else:
                temp_expanded = temp.expand(batch_size, 1)

        kd_loss = F.kl_div(
            F.log_softmax(logit_s / temp, 1),
            F.softmax(logit_t / temp, 1),
            reduction='none'
        ).sum(1, keepdim=True) * (temp ** 2)

        temp_log = torch.log(temp_expanded + 1e-6)
        temp_mean = temp_expanded.mean(dim=1, keepdim=True).expand(batch_size, 1)

        state = torch.cat([
            feat_gap,
            logit_gap,
            s_conf,
            t_conf,
            ce_loss,
            kd_loss,
            temp_log,
            temp_mean
        ], dim=1)

        state = (state - state.mean(0)) / (state.std(0) + 1e-6)

    return state