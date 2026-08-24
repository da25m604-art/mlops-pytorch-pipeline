import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from dataset import get_dataloaders
from model import get_model


def log(payload):
    payload.setdefault("ts", round(time.time(), 3))
    print(json.dumps(payload), flush=True)


def resolve_config_path(cli_path):
    # precedence: --config flag, env var, k8s mount path, local path
    candidates = [
        cli_path,
        os.environ.get("TRAINING_CONFIG"),
        "/app/configs/training_config.yaml",
        "configs/training_config.yaml",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    raise FileNotFoundError("no training config found")


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    log({"event": "config_loaded", "path": str(config_path), "config": config})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log({"event": "device_selected", "device": str(device)})

    torch.manual_seed(int(os.environ.get("SEED", "42")))

    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        dataset=config["data"].get("dataset", "cifar10"),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]

    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / config["output"]["model_name"]

    for epoch in range(config["training"]["epochs"]):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        log({
            "event": "epoch_end",
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "epoch_seconds": round(time.time() - start, 1),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "config": config,
            }, save_path)
            log({"event": "checkpoint_saved", "path": str(save_path), "val_loss": round(val_loss, 4)})
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log({"event": "early_stopping", "epoch": epoch + 1})
                break

    log({"event": "training_complete", "best_val_loss": round(best_val_loss, 4), "checkpoint": str(save_path)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
