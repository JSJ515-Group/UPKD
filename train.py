"""General training framework for UPKD."""

import argparse
from datetime import datetime
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.optim as optim

from distiller_zoo import UPKDLoss
from helper.data import build_dataloaders
from helper.loops import train_epoch, validate
from helper.model_setup import build_models
from modules import LearnableTemperature, PolicyAgent


def parse_option():
    parser = argparse.ArgumentParser("UPKD training")

    # Dataset and runtime options.
    parser.add_argument("--dataset",default="cifar100",choices=["cifar100", "cifar10", "tiny_imagenet"],)
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    # Optimization options.
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lr_decay_epochs", default="150,180,210")
    parser.add_argument("--lr_decay_rate", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_lr", type=float, default=0.01)

    # Model options.
    parser.add_argument("--model_s", default="resnet20", help="Student model.")
    parser.add_argument("--teacher", default="resnet56", help="Teacher model.")

    # Method.
    parser.add_argument("--method",type=str,default="upkd",choices=["upkd"],help="Distillation method.")

    # Learnable temperature options.
    parser.add_argument("--t_start", type=float, default=2.0)
    parser.add_argument("--t_end", type=float, default=6.0)
    parser.add_argument("--decay_loops", type=int, default=80)
    parser.add_argument("--temperature_warmup_epochs",type=int,default=15)

    # Policy options.
    parser.add_argument("--agent_lr", type=float, default=1e-3)
    parser.add_argument("--agent_step", type=int, default=10)
    parser.add_argument("--policy_warmup_epochs", type=int, default=40)

    # Loss options.
    parser.add_argument("--alpha", type=float, default=1.1)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--kd_T", type=float, default=4.0)
    parser.add_argument("--feature_warmup_epochs", type=int, default=20)
    parser.add_argument("--beta_max", type=float, default=1.2)
    parser.add_argument("--feature_consistency_weight",type=float,default=0.0)

    # Saving and resume options.
    parser.add_argument("--save_dir", default="./save")
    parser.add_argument("--resume",default=None,help="Checkpoint path used to resume training.",)
    parser.add_argument("--start_epoch", type=int, default=0)

    args = parser.parse_args()
    if args.method == "upkd":
        args.use_temperature = True
        args.use_policy = True
        args.use_feature_transfer = True
    student_name = args.model_s.lower()
    if args.lr == parser.get_default("lr") and (
            "shuffle" in student_name or "mobile" in student_name
    ):
        args.lr = 0.02

    args.lr_decay_epochs = [
        int(epoch) for epoch in args.lr_decay_epochs.split(",")
    ]
    return args

def get_enabled_modules(args):
    return ["TAKD", "PADW", "HFA"]

def get_method_name(args):
    return args.method.upper()

def get_run_name(args, timestamp):
    method_slug = get_method_name(args).lower().replace("+", "-")
    return (
        f"{args.dataset}_{args.teacher}-to-{args.model_s}_"
        f"{method_slug}_seed{args.seed}_{timestamp}"
    )


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_training_state(args):
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )

    student, teacher = build_models(args, device)

    if args.use_temperature:
        temperature_module = LearnableTemperature(
            t_start=args.t_start,
            t_end=args.t_end,
        ).to(device)
    else:
        temperature_module = None

    if args.use_policy:
        policy_agent = PolicyAgent(state_dim=8, hidden_dim=128).to(device)
        agent_optimizer = optim.Adam(
            policy_agent.parameters(),
            lr=args.agent_lr,
        )
    else:
        policy_agent = None
        agent_optimizer = None

    use_feature_transfer = bool(args.use_feature_transfer)

    criterion = UPKDLoss(
        alpha=args.alpha,
        beta=args.beta,
        kd_temperature=args.kd_T,
    ).to(device)

    parameters = list(student.parameters())
    if temperature_module is not None:
        parameters += list(temperature_module.parameters())

    optimizer = optim.SGD(
        parameters,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    train_loader, val_loader = build_dataloaders(args)

    return SimpleNamespace(
        args=args,
        device=device,
        student=student,
        teacher=teacher,
        criterion=criterion,
        optimizer=optimizer,
        temperature_module=temperature_module,
        policy_agent=policy_agent,
        agent_optimizer=agent_optimizer,
        use_feature_transfer=use_feature_transfer,
        train_loader=train_loader,
        val_loader=val_loader,
        reward_baseline=0.0,
        reward_momentum=0.9,
        reward_std=1.0,
        entropy_weight=0.01,
        alpha_prev=None,
        beta_prev=None,
    )


def load_checkpoint(state):
    args = state.args
    if args.resume is None:
        return 0, 0.0, 0.0

    if not os.path.isfile(args.resume):
        print(f"Checkpoint not found: {args.resume}")
        return 0, 0.0, 0.0

    checkpoint = torch.load(args.resume, map_location="cpu")
    state.student.load_state_dict(checkpoint["model"])
    state.optimizer.load_state_dict(checkpoint["optimizer"])

    if (
        state.temperature_module is not None
        and checkpoint.get("temp_module") is not None
    ):
        state.temperature_module.load_state_dict(
            checkpoint["temp_module"]
        )

    if (
        state.policy_agent is not None
        and checkpoint.get("policy_agent") is not None
    ):
        state.policy_agent.load_state_dict(checkpoint["policy_agent"])

    if (
        state.policy_agent is not None
        and checkpoint.get("agent_optimizer") is not None
    ):
        state.agent_optimizer.load_state_dict(
            checkpoint["agent_optimizer"]
        )

    start_epoch = (
        args.start_epoch
        if args.start_epoch > 0
        else checkpoint["epoch"] + 1
    )
    top1 = checkpoint.get(
        "accuracy_top1",
        checkpoint.get("accuracy", 0.0),
    )
    top5 = checkpoint.get("accuracy_top5", 0.0)

    print(
        f"resume={args.resume} start_epoch={start_epoch + 1} "
        f"best_top1={top1:.2f}% best_top5={top5:.2f}%"
    )
    return start_epoch, top1, top5


def build_checkpoint(state, epoch, top1, top5, run_name):
    return {
        "model": state.student.state_dict(),
        "optimizer": state.optimizer.state_dict(),
        "epoch": epoch,
        "accuracy": top1,
        "accuracy_top1": top1,
        "accuracy_top5": top5,
        "run_name": run_name,
        "temp_module": (
            state.temperature_module.state_dict()
            if state.temperature_module is not None
            else None
        ),
        "policy_agent": (
            state.policy_agent.state_dict()
            if state.policy_agent is not None
            else None
        ),
        "agent_optimizer": (
            state.agent_optimizer.state_dict()
            if state.policy_agent is not None
            else None
        ),
    }


def save_best_model(state, epoch, top1, top5, run_name):
    checkpoint = build_checkpoint(
        state,
        epoch,
        top1,
        top5,
        run_name,
    )
    save_path = os.path.join(
        state.args.save_dir,
        f"best_{run_name}.pth",
    )
    torch.save(checkpoint, save_path)
    return save_path


def print_startup(args, state, run_name):
    modules = ", ".join(get_enabled_modules(args)) or "None"
    print(f"run={run_name}")
    print(f"method={get_method_name(args)} modules=[{modules}]")
    print(
        f"dataset={args.dataset} teacher={args.teacher} "
        f"student={args.model_s} epochs={args.epochs} "
        f"batch_size={args.batch_size} device={state.device}"
    )


def main():
    args = parse_option()
    set_random_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = get_run_name(args, timestamp)
    state = build_training_state(args)
    os.makedirs(args.save_dir, exist_ok=True)
    print_startup(args, state, run_name)

    start_epoch, best_top1, best_top5 = load_checkpoint(state)
    best_model_path = None

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        training_loss = train_epoch(state, epoch)
        top1, top5 = validate(state)
        elapsed = time.time() - start_time

        is_best = top1 > best_top1
        if is_best:
            best_top1, best_top5 = top1, top5
            best_model_path = save_best_model(
                state,
                epoch,
                best_top1,
                best_top5,
                run_name,
            )

        best_marker = " best" if is_best else ""
        print(
            f"Epoch [{epoch + 1:03d}/{args.epochs:03d}] "
            f"Loss={training_loss:.4f} "
            f"Top-1={top1:.2f}% Top-5={top5:.2f}% "
            f"Time={elapsed:.1f}s{best_marker}"
        )

    print(
        f"Training completed: best_top1={best_top1:.2f}% "
        f"best_top5={best_top5:.2f}%"
    )
    if best_model_path is not None:
        print(f"best_model={best_model_path}")


if __name__ == "__main__":
    main()
