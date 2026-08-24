import torch
import torch.nn.functional as F

from .features import (
    align_feature_lists,
    align_logits,
    feature_transfer_loss,
    forward_teacher,
    get_last_feature,
    match_vector_dimensions,
    vectorize_feature,
)
from .optimization import (
    adjust_learning_rate,
    compute_temperature,
    policy_weights,
    update_policy,
)


def unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        return batch[0], batch[1]
    if isinstance(batch, dict):
        inputs = batch.get(
            "image",
            batch.get("input", batch.get("inputs", batch.get("data"))),
        )
        targets = batch.get(
            "target",
            batch.get("targets", batch.get("label")),
        )
        return inputs, targets
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def train_epoch(trainer, epoch):
    args = trainer.args
    trainer.student.train()

    if trainer.temperature_module is not None:
        trainer.temperature_module.train()
    if trainer.policy_agent is not None:
        trainer.policy_agent.train()

    adjust_learning_rate(trainer, epoch)
    total_loss = 0.0
    total_samples = 0
    gamma = args.gamma

    for batch_index, batch in enumerate(trainer.train_loader):
        inputs, targets = unpack_batch(batch)
        inputs = inputs.to(trainer.device, non_blocking=True)
        targets = targets.to(
            trainer.device,
            non_blocking=True,
        ).long()

        trainer.optimizer.zero_grad()

        if trainer.use_feature_transfer:
            student_features, student_logits = trainer.student(inputs)
        else:
            student_features, student_logits = trainer.student(
                inputs,
                is_feat=True,
            )

        if not isinstance(student_features, list):
            student_features = [student_features]

        with torch.no_grad():
            teacher_features, teacher_logits = forward_teacher(
                trainer.teacher,
                inputs,
                preact=trainer.use_feature_transfer,
            )

        teacher_logits = align_logits(
            teacher_logits,
            student_logits.size(1),
        )

        student_last = vectorize_feature(
            get_last_feature(student_features)
        )
        teacher_last = vectorize_feature(
            get_last_feature(teacher_features)
        )
        student_last, teacher_last = match_vector_dimensions(
            student_last,
            teacher_last,
        )
        temperature = compute_temperature(
            trainer,
            student_logits,
            teacher_logits,
            epoch,
        )
        alpha, beta, policy_log_prob, policy_entropy = policy_weights(
            trainer,
            epoch,
            student_last,
            teacher_last,
            student_logits,
            teacher_logits,
            targets,
            temperature,
        )
        classification_loss = F.cross_entropy(
            student_logits,
            targets,
        )
        distillation_loss = F.kl_div(
            F.log_softmax(
                student_logits / temperature,
                dim=1,
            ),
            F.softmax(
                teacher_logits.detach() / temperature,
                dim=1,
            ),
            reduction="batchmean",
        ) * (
            temperature.mean() ** 2
            if temperature.dim() > 0
            else temperature ** 2
        )

        if trainer.use_feature_transfer:
            feature_loss, _ = feature_transfer_loss(
                student_features,
                teacher_features,
                epoch,
                args,
            )
        else:
            loss_student, loss_teacher = align_feature_lists(
                student_features,
                teacher_features,
            )
            _, values = trainer.criterion(
                student_logits,
                teacher_logits,
                targets,
                feat_s=loss_student,
                feat_t=loss_teacher,
                temp=temperature,
                alpha=alpha,
                beta=beta,
                epoch=epoch,
                warmup_epochs=args.feature_warmup_epochs,
                feature_scale=2.0,
            )
            feature_loss = torch.tensor(
                values["feat"],
                device=trainer.device,
                dtype=student_logits.dtype,
            )

        loss = (
            gamma * classification_loss
            + alpha * distillation_loss
            + beta * feature_loss
        )

        if args.feature_consistency_weight > 0:
            consistency = F.mse_loss(
                student_last,
                teacher_last.detach(),
            )
            loss = (
                loss
                + args.feature_consistency_weight * consistency
            )
        else:
            consistency = torch.tensor(
                0.0,
                device=trainer.device,
            )

        loss.backward()
        trainer.optimizer.step()
        update_policy(
            trainer,
            batch_index,
            policy_log_prob,
            policy_entropy,
            classification_loss,
            distillation_loss,
            feature_loss,
        )

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

        if batch_index % args.print_freq == 0:
            learning_rate = trainer.optimizer.param_groups[0]["lr"]
            fields = [
                f"Train [{epoch + 1:03d}/{args.epochs:03d}]",
                (
                    f"Step [{batch_index + 1:04d}/"
                    f"{len(trainer.train_loader):04d}]"
                ),
                f"LR={learning_rate:.6f}",
                f"Loss={loss.item():.4f}",
                f"CE={classification_loss.item():.4f}",
                f"KD={distillation_loss.item():.4f}",
            ]
            if trainer.use_feature_transfer:
                fields.append(f"HFA={feature_loss.item():.4f}")
            if args.feature_consistency_weight > 0:
                fields.append(f"Cons={consistency.item():.4f}")
            print(" ".join(fields))

    return total_loss / max(total_samples, 1)


def validate(trainer):
    trainer.student.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for batch in trainer.val_loader:
            inputs, targets = unpack_batch(batch)
            inputs = inputs.to(trainer.device, non_blocking=True)
            targets = targets.to(trainer.device, non_blocking=True)

            output = trainer.student(inputs)
            logits = output[1] if isinstance(output, tuple) else output

            prediction = logits.topk(
                1,
                dim=1,
                largest=True,
                sorted=True,
            )[1].squeeze(1)
            correct_top1 += prediction.eq(targets).sum().item()

            top_k = min(5, logits.size(1))
            top5_prediction = logits.topk(
                top_k,
                dim=1,
                largest=True,
                sorted=True,
            )[1]
            correct = top5_prediction.eq(
                targets.view(-1, 1).expand_as(top5_prediction)
            )
            correct_top5 += correct.any(dim=1).sum().item()
            total += targets.size(0)

    top1 = 100.0 * correct_top1 / max(total, 1)
    top5 = 100.0 * correct_top5 / max(total, 1)
    return top1, top5
