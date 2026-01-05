import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms

from medmnist import BloodMNIST, INFO

import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import json
device = "cuda" if torch.cuda.is_available() else "cpu"
data_flag = "bloodmnist"
info = INFO[data_flag]
n_classes = len(info["label"])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

def plot(x, y, ylabel="", name=""):
    plt.clf()
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.plot(list(x), y)
    plt.savefig(f"{name}.pdf", bbox_inches="tight")


def plot_compare(x, series_dict, ylabel="", title="", filename="compare"):
    plt.clf()
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    for label, y in series_dict.items():
        plt.plot(list(x), y, label=label)
    plt.legend()
    plt.savefig(f"{filename}.pdf", bbox_inches="tight")

class CNN(nn.Module):
    def __init__(self, num_classes=8, use_maxpool=False, output_softmax=False):
        super().__init__()
        self.use_maxpool = use_maxpool
        self.output_softmax = output_softmax

        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # if not max pool H=W=28
        # with 3 maxpool, 28->14->7->3
        if use_maxpool:
            in_features = 128 * 3 * 3
        else:
            in_features = 128 * 28 * 28

        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        if self.use_maxpool:
            x = self.pool(x)

        x = F.relu(self.conv2(x))
        if self.use_maxpool:
            x = self.pool(x)

        x = F.relu(self.conv3(x))
        if self.use_maxpool:
            x = self.pool(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        logits = self.fc2(x)

        if self.output_softmax:
            return F.softmax(logits, dim=1)
        return logits


def train_epoch(loader, model, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.squeeze().long().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_acc(loader, model):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.squeeze().long().to(device, non_blocking=True)

            outputs = model(imgs)
            preds += outputs.argmax(dim=1).cpu().tolist()
            targets += labels.cpu().tolist()
    return accuracy_score(targets, preds)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    batch_size = 64
    train_dataset = BloodMNIST(split="train", transform=transform, download=True, size=28)
    val_dataset = BloodMNIST(split="val", transform=transform, download=True, size=28)
    test_dataset = BloodMNIST(split="test", transform=transform, download=True, size=28)

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    epochs = 200
    lr = 1e-3

    # 4 modelos = pooling x softmax
    use_pooling_flags = [True, False]
    use_softmax_flags = [True, False]

    results = {
        "train_loss": {},
        "val_acc": {},
        "test_acc": {},
        "time_sec": {},
        "params": {},
        "best_val": {},
        "best_epoch": {},
        "total_time": {}
    }

    global_start = time.time()

    for pool_flag in use_pooling_flags:
        for soft_flag in use_softmax_flags:

            tag_pool = "with_pooling" if pool_flag else "no_pooling"
            tag_soft = "with_softmax" if soft_flag else "no_softmax"
            tag = f"{tag_pool}__{tag_soft}"  # tag único para 4 modelos

            print("\n" + "=" * 70)
            print("EXPERIMENT:", tag)
            print("=" * 70)

            # cria modelo com as 2 flags
            model = CNN(num_classes=n_classes, use_maxpool=pool_flag, output_softmax=soft_flag).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            results["best_val"][tag] = 0.0
            results["best_epoch"][tag] = 0

            print("Trainable params:", count_params(model))

            train_losses = []
            val_acc_list = []

            exp_start = time.time()

            for epoch in range(epochs):
                t0 = time.time()

                tr_loss = train_epoch(train_loader, model, criterion, optimizer)
                val_acc = evaluate_acc(val_loader, model)

                train_losses.append(tr_loss)
                val_acc_list.append(val_acc)

                dt = time.time() - t0

                print(
                    f"Epoch {epoch + 1:03d}/{epochs} | "
                    f"Loss: {tr_loss:.4f} | "
                    f"ValAcc: {val_acc:.4f} | "
                    f"Time: {dt:.2f}s"
                )

                if val_acc > results["best_val"][tag]:
                    results["best_val"][tag] = val_acc
                    results["best_epoch"][tag] = epoch + 1
                    torch.save(model.state_dict(), f"bloodmnist_cnn_{tag}.pth")
                    print(f"Best model saved: bloodmnist_cnn_{tag}.pth | Val: {results['best_val'][tag]:.4f}")

            test_acc = evaluate_acc(test_loader, model)

            exp_time = time.time() - exp_start
            results["total_time"][tag] = exp_time

            print(f"TestAcc: {test_acc:.4f}")
            print(f"Total time ({tag}): {exp_time / 60:.2f} min ({exp_time:.2f} s)")

            # Store results
            results["train_loss"][tag] = train_losses
            results["val_acc"][tag] = val_acc_list
            results["test_acc"][tag] = test_acc
            results["time_sec"][tag] = exp_time
            results["params"][tag] = count_params(model)

            # Per-model plots
            ep = range(len(train_losses))

            plot(ep, train_losses, ylabel="Loss",
                 name=f"CNN-training-loss_{tag}_lr{lr}")

            plot(ep, val_acc_list, ylabel="Accuracy",
                 name=f"CNN-validation-accuracy_{tag}_lr{lr}")


            with open("results_plots/results_all_4models.json", "w") as f:
                json.dump(results, f, indent=4)


    ep = range(epochs)

    plot_compare(
        ep,
        {
            "with_pooling | no_softmax": results["train_loss"]["with_pooling__no_softmax"],
            "with_pooling | with_softmax": results["train_loss"]["with_pooling__with_softmax"],
            "no_pooling | no_softmax": results["train_loss"]["no_pooling__no_softmax"],
            "no_pooling | with_softmax": results["train_loss"]["no_pooling__with_softmax"],
        },
        ylabel="Loss",
        title="Training Loss (4 models)",
        filename=f"COMPARE_training_loss_4models_lr{lr}"
    )

    plot_compare(
        ep,
        {
            "with_pooling | no_softmax": results["val_acc"]["with_pooling__no_softmax"],
            "with_pooling | with_softmax": results["val_acc"]["with_pooling__with_softmax"],
            "no_pooling | no_softmax": results["val_acc"]["no_pooling__no_softmax"],
            "no_pooling | with_softmax": results["val_acc"]["no_pooling__with_softmax"],
        },
        ylabel="Accuracy",
        title="Validation Accuracy (4 models)",
        filename=f"COMPARE_val_acc_4models_lr{lr}"
    )


    global_time = time.time() - global_start
    print("\n" + "=" * 70)
    print("DONE")
    print("Total run time:", f"{global_time / 60:.2f} min ({global_time:.2f} s)")
    print("Params:", results["params"])
    print("Best val:", results["best_val"])
    print("Best epoch:", results["best_epoch"])
    print("Test acc:", results["test_acc"])

if __name__ == '__main__':
    main()