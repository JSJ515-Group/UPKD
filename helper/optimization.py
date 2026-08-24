import torch

from modules import build_state


def adjust_learning_rate(trainer, epoch):
    args = trainer.args

    if epoch < args.warmup_epochs:
        lr = args.warmup_lr + (
            args.lr - args.warmup_lr
        ) * (epoch + 1) / args.warmup_epochs
    else:
        lr = args.lr
        for milestone in args.lr_decay_epochs:
            if epoch >= milestone:
                lr *= args.lr_decay_rate

    for group in trainer.optimizer.param_groups:
        group["lr"] = lr
    return lr


def compute_temperature(trainer, student_logits, teacher_logits, epoch):
    args = trainer.args
    base = torch.tensor(
        args.kd_T,
        device=trainer.device,
        dtype=student_logits.dtype,
    )
    if trainer.temperature_module is None:
        return base

    if args.decay_loops > 0:
        coefficient = min(1.0, epoch / args.decay_loops)
    else:
        coefficient = 1.0

    learned = trainer.temperature_module(
        student_logits,
        teacher_logits,
        coefficient,
    )
    learned = torch.clamp(learned,args.t_start,args.t_end)
    mix = min(
        1.0,
        float(epoch + 1) / float(args.temperature_warmup_epochs),
    )
    return torch.clamp((1.0 - mix) * base + mix * learned,args.t_start,args.t_end)

def policy_weights(
    trainer,
    epoch,
    student_feature,
    teacher_feature,
    student_logits,
    teacher_logits,
    targets,
    temperature,
):
    args = trainer.args
    alpha = torch.tensor(
        args.alpha,
        device=trainer.device,
        dtype=student_logits.dtype,
    )
    beta = torch.tensor(
        args.beta,
        device=trainer.device,
        dtype=student_logits.dtype,
    )
    state = None

    if (
            trainer.policy_agent is None
            or epoch < args.policy_warmup_epochs
    ):
        return alpha, beta, None, None

    state = build_state(
        student_feature.detach(),
        teacher_feature.detach(),
        student_logits.detach(),
        teacher_logits.detach(),
        targets,
        temperature,
    ).detach()

    ratio, log_prob, entropy = trainer.policy_agent(state)
    ratio = ratio.mean()
    temperature_ratio = temperature.mean().detach() / args.kd_T
    alpha_new = ratio * (1.0 + 0.35 * temperature_ratio)
    beta_new = (1.0 - ratio) * (
        1.0 + 0.25 / (temperature_ratio + 1e-6)
    )
    alpha_new = torch.clamp(alpha_new, 0.3, 2.5)
    beta_new = torch.clamp(beta_new,0.2,args.beta_max)

    if trainer.alpha_prev is None:
        trainer.alpha_prev = alpha_new.detach()
        trainer.beta_prev = beta_new.detach()

    alpha = 0.7 * trainer.alpha_prev + 0.3 * alpha_new
    beta = 0.7 * trainer.beta_prev + 0.3 * beta_new
    trainer.alpha_prev = alpha.detach()
    trainer.beta_prev = beta.detach()
    return alpha, beta, log_prob, entropy

def update_policy(
        trainer,
        batch_index,
        log_prob,
        entropy,
        classification_loss,
        distillation_loss,
        feature_loss,
):
    args = trainer.args
    if (
            trainer.policy_agent is None
            or log_prob is None
            or batch_index % args.agent_step != 0
    ):
        return


    reward = -(
        classification_loss.detach()
        + 0.7 * distillation_loss.detach()
        + 0.3 * feature_loss.detach()
    )
    trainer.reward_baseline = (
        trainer.reward_momentum * trainer.reward_baseline
        + (1.0 - trainer.reward_momentum) * reward.item()
    )

    advantage = reward - trainer.reward_baseline
    trainer.reward_std = (
        0.99 * trainer.reward_std
        + 0.01 * abs(advantage.item())
    )
    advantage = (advantage / (trainer.reward_std + 1e-6)).detach()

    policy_loss = (
        -log_prob.mean() * advantage
        - trainer.entropy_weight * entropy.mean()
    )

    trainer.agent_optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        trainer.policy_agent.parameters(),
        5.0,
    )
    trainer.agent_optimizer.step()
