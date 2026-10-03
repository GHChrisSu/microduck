#!/usr/bin/env python3
"""Behaviour-clone a small visual-follow policy from simulator demonstrations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from follow_policy import (
    ACTION_COLUMNS,
    FEATURE_COLUMNS,
    FollowActor,
    MAX_FORWARD_MPS,
    MAX_YAW_RAD_S,
)


def parse_args() -> argparse.Namespace:
    default_output = Path.home() / ".cache/duck-sim/follow-training/follow-policy.ts"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="+", type=Path, help="one or more --record-csv demonstration files")
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch size, and patience must be positive")
    return args


def read_demonstrations(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    features: list[list[float]] = []
    actions: list[list[float]] = []
    required = (*FEATURE_COLUMNS, "teacher_vx", "teacher_vyaw")
    for path in paths:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            missing = set(required) - set(reader.fieldnames or ())
            if missing:
                raise SystemExit(f"{path} is missing columns: {', '.join(sorted(missing))}")
            for line_number, row in enumerate(reader, start=2):
                try:
                    x = [float(row[name]) for name in FEATURE_COLUMNS]
                    y = [float(row["teacher_vx"]), float(row["teacher_vyaw"])]
                except (TypeError, ValueError) as error:
                    raise SystemExit(f"invalid number in {path}:{line_number}: {error}") from error
                if np.isfinite(x).all() and np.isfinite(y).all():
                    features.append(x)
                    actions.append(y)
    if len(features) < 100:
        raise SystemExit(
            f"need at least 100 visible-target samples; found {len(features)}. "
            "Collect a longer run or add demonstrations from another target-motion seed."
        )
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(actions, dtype=np.float32)
    if np.any(y[:, 0] < -1e-5) or np.any(y[:, 0] > MAX_FORWARD_MPS + 1e-5):
        raise SystemExit(f"forward labels must be within 0..{MAX_FORWARD_MPS} m/s")
    if np.any(np.abs(y[:, 1]) > MAX_YAW_RAD_S + 1e-5):
        raise SystemExit(f"yaw labels must be within ±{MAX_YAW_RAD_S} rad/s")
    return x, y


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    x, y = read_demonstrations(args.data)

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested, but this machine does not expose an Apple GPU")
    print(f"Loaded {len(x)} samples from {len(args.data)} demonstration file(s); training on {device}.")

    permutation = torch.randperm(len(x))
    validation_count = max(1, int(round(len(x) * 0.2)))
    validation_indices = permutation[:validation_count]
    train_indices = permutation[validation_count:]
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    x_train = x_tensor[train_indices]
    y_train = y_tensor[train_indices]
    x_validation = x_tensor[validation_indices]
    y_validation = y_tensor[validation_indices]

    mean = x_train.mean(dim=0)
    scale = x_train.std(dim=0, unbiased=False).clamp_min(1e-3)
    model = FollowActor(mean, scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=0.1)
    action_scale = torch.tensor([MAX_FORWARD_MPS, MAX_YAW_RAD_S], device=device)
    dataset = TensorDataset(x_train.to(device), y_train.to(device))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    validation_x = x_validation.to(device)
    validation_y = y_validation.to(device)

    best_loss = float("inf")
    best_weights = None
    stale_epochs = 0
    for epoch in range(args.epochs):
        model.train()
        for batch_x, batch_y in loader:
            prediction = model(batch_x)
            loss = loss_fn(prediction / action_scale, batch_y / action_scale)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            validation_prediction = model(validation_x)
            validation_loss = loss_fn(
                validation_prediction / action_scale,
                validation_y / action_scale,
            ).item()
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"epoch {epoch + 1:4d} · validation scaled loss {validation_loss:.5f}", flush=True)
        if stale_epochs >= args.patience:
            break

    assert best_weights is not None
    model.load_state_dict(best_weights)
    model.eval()
    with torch.inference_mode():
        prediction = model(x_validation.to(device)).cpu().numpy()
    mae = np.abs(prediction - y_validation.numpy()).mean(axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_model = model.to("cpu").eval()
    example = torch.zeros((1, len(FEATURE_COLUMNS)), dtype=torch.float32)
    torch.jit.trace(export_model, example).save(str(args.output))
    metadata = {
        "algorithm": "behaviour cloning from the existing rule-based simulator controller",
        "features": list(FEATURE_COLUMNS),
        "actions": list(ACTION_COLUMNS),
        "samples": int(len(x)),
        "training_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "validation_mae": {"vx_mps": float(mae[0]), "vyaw_rad_s": float(mae[1])},
        "validation_split": "random sample split; adjacent simulator frames are correlated",
        "source_files": [str(path) for path in args.data],
        "seed": args.seed,
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"Saved TorchScript policy to {args.output}\n"
        f"Saved metadata to {metadata_path}\n"
        f"Validation MAE · vx {mae[0]:.3f} m/s · vyaw {mae[1]:.3f} rad/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
