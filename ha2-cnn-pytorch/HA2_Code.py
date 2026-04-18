# -*- coding: utf-8 -*-
"""
Deep Learning 1RT720 -- Hand-in Assignment 2
MNIST Classification with PyTorch

Author: Rafael
Running on: Google Colab GPU (CUDA)

Generative AI disclosure: GitHub Copilot was used as a coding assistant
for boilerplate, plotting, and markdown drafting. All architectural
decisions, hyperparameter choices, and written analysis are my own.
"""

# ============================================================
# Setup
# ============================================================

import sys

# Workaround: flush stale torchvision submodules to avoid C++ op
# registration conflicts that can occur when restarting Colab sessions.
for mod in [k for k in sys.modules if "torchvision" in k]:
    del sys.modules[mod]

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np
    import time
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
except RuntimeError as e:
    if "already a kernel registered" in str(e):
        raise RuntimeError(
            "torchvision C++ ops conflict detected.\n"
            "-> Go to Runtime -> Restart session, then run cells from here."
        ) from None
    raise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# ============================================================
# Data Loading
# ============================================================

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


def get_loaders(batch_size):
    """Create train and test DataLoaders.

    Args:
        batch_size: int, batch size for the training loader.

    Returns:
        (train_loader, test_loader) tuple of DataLoaders.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=1000,       shuffle=False)
    return train_loader, test_loader

print(f"Train: {len(train_dataset)} images")
print(f"Test:  {len(test_dataset)} images")

# ============================================================
# Plotting Function
# ============================================================

def training_curve_plot(title, train_costs, test_costs, train_accs, test_accs,
                        batch_size, learning_rate, num_epochs, elapsed):
    """Plot cost and accuracy curves for training and test sets.

    Same signature as utils.training_curve_plot from Hand-in 1.

    Args:
        title: str, plot title.
        train_costs, test_costs: lists of per-epoch costs.
        train_accs, test_accs: lists of per-epoch accuracies (fractions in [0,1]).
        batch_size: int, batch size used.
        learning_rate: float, learning rate used.
        num_epochs: int, total epochs.
        elapsed: float, training time in seconds.

    Returns:
        matplotlib Figure object.
    """
    epochs = np.arange(1, len(train_costs) + 1)
    time_str = f"{int(elapsed // 60)}min {elapsed % 60:.0f}sec"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"{title}\n| Batch size: {batch_size} | Learning rate: {learning_rate} "
        f"| Number of Epochs: {num_epochs} | Training time: {time_str} |",
        fontsize=11,
    )

    # Costs
    ax1.plot(epochs, train_costs, label=f"Final train cost: {train_costs[-1]:.4f}")
    ax1.plot(epochs, test_costs,  label=f"Final test cost: {test_costs[-1]:.4f}")
    ax1.set_title("Costs"); ax1.set_xlabel("Epochs"); ax1.set_ylabel("Cost")
    ax1.legend()

    # Accuracy
    tr_pct = np.array(train_accs) * 100
    te_pct = np.array(test_accs)  * 100
    ax2.plot(epochs, tr_pct, label=f"Final train accuracy: {tr_pct[-1]:.2f}%")
    ax2.plot(epochs, te_pct, label=f"Final test accuracy: {te_pct[-1]:.2f}%")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epochs"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    plt.tight_layout()
    return fig

# ============================================================
# Training & Evaluation Functions
# ============================================================

@torch.no_grad()
def evaluate(model, loader, criterion):
    """Evaluate model on a DataLoader.

    Args:
        model: nn.Module, the model to evaluate.
        loader: DataLoader, data to evaluate on.
        criterion: loss function.

    Returns:
        (avg_cost, accuracy_fraction) tuple.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        total_loss += criterion(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def train(model, train_loader, test_loader, optimizer, num_epochs,
          batch_size, learning_rate, title=""):
    """Full training loop with per-epoch evaluation and plotting.

    Args:
        model: nn.Module, the model to train.
        train_loader, test_loader: DataLoaders for train/test data.
        optimizer: torch.optim optimizer.
        num_epochs: int, number of training epochs.
        batch_size: int, batch size (for plot annotation).
        learning_rate: float, learning rate (for plot annotation).
        title: str, title for the learning curve plot.

    Returns:
        (model, history, elapsed) where history is a dict with keys
        'train_cost', 'test_cost', 'train_acc', 'test_acc'.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    train_costs, test_costs = [], []
    train_accs,  test_accs  = [], []

    start = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()
        batch_losses = []
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            out  = model(X_batch)
            loss = criterion(out, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            correct += (out.argmax(1) == y_batch).sum().item()
            total   += y_batch.size(0)

        # Epoch stats
        train_costs.append(np.mean(batch_losses))
        train_accs.append(correct / total)

        te_cost, te_acc = evaluate(model, test_loader, criterion)
        test_costs.append(te_cost)
        test_accs.append(te_acc)

        if epoch == 1 or epoch % 5 == 0 or epoch == num_epochs:
            print(f"Epoch {epoch:3d}/{num_epochs} -- "
                  f"Train: cost={train_costs[-1]:.4f} acc={100*train_accs[-1]:.2f}% | "
                  f"Test: cost={te_cost:.4f} acc={100*te_acc:.2f}%")

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s")

    fig = training_curve_plot(
        title, train_costs, test_costs, train_accs, test_accs,
        batch_size, learning_rate, num_epochs, elapsed,
    )
    plt.show()

    history = {
        'train_cost': np.array(train_costs), 'test_cost': np.array(test_costs),
        'train_acc':  np.array(train_accs),  'test_acc':  np.array(test_accs),
    }
    return model, history, elapsed


def count_parameters(model):
    """Print layer-wise and total learnable parameter counts.

    Args:
        model: nn.Module.

    Returns:
        int, total number of parameters.
    """
    print("\nParameter count:")
    print("-" * 55)
    total = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        print(f"  {name:35s} {n:>8,}  {list(p.shape)}")
    print("-" * 55)
    print(f"  {'TOTAL':35s} {total:>8,}\n")
    return total

# ============================================================
# Exercise 1: Multi-layer Fully-Connected Neural Network
# Architecture: [784, 128, 64, 10] + ReLU, SGD, same as HW1 Ex3.
# ============================================================

class FullyConnectedNet(nn.Module):
    """Fully-connected network: [784, 128, 64, 10] with ReLU.

    Same architecture as Hand-in 1 Exercise 3.

    Input:  (B, 1, 28, 28) or (B, 784) MNIST images.
    Output: (B, 10) raw logits for each digit class.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)       # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)              # raw logits

# --- Exercise 1: Training ---
bs1, lr1, ep1 = 64, 0.1, 50
loader_tr, loader_te = get_loaders(bs1)

model_fc = FullyConnectedNet()
count_parameters(model_fc)
opt1 = optim.SGD(model_fc.parameters(), lr=lr1)

model_fc, hist_fc, elapsed_fc = train(
    model_fc, loader_tr, loader_te, opt1,
    ep1, bs1, lr1,
    title="Exercise 1 -- FC Network (PyTorch) [784,128,64,10]",
)

# ============================================================
# Exercise 2: Multi-layer Convolutional Neural Network
# Three conv layers + ReLU + max pooling, then FC classifier.
# Trained with SGD.
# ============================================================

class ConvNet(nn.Module):
    """CNN for MNIST classification.

    Architecture:
        Conv(1->8, 3x3, pad1) -> ReLU -> MaxPool(2x2)
        Conv(8->16, 3x3, pad1) -> ReLU -> MaxPool(2x2)
        Conv(16->32, 3x3, pad1) -> ReLU -> Flatten -> FC(1568, 10)

    Input:  (B, 1, 28, 28) MNIST images.
    Output: (B, 10) raw logits.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,  8,  3, padding=1)
        self.conv2 = nn.Conv2d(8,  16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc    = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # -> (B, 8, 14,14)
        x = self.pool(F.relu(self.conv2(x)))   # -> (B,16,  7, 7)
        x = F.relu(self.conv3(x))               # -> (B,32,  7, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --- Exercise 2(a): Parameter count comparison ---
model_cnn = ConvNet()
count_parameters(model_cnn)

fc_total  = sum(p.numel() for p in FullyConnectedNet().parameters())
cnn_total = sum(p.numel() for p in model_cnn.parameters())
print(f"FC network:  {fc_total:,} params")
print(f"CNN:         {cnn_total:,} params")
print(f"-> CNN uses {fc_total / cnn_total:.1f}x fewer parameters.")

# --- Exercise 2(b): Training ---
bs2, lr2, ep2 = 64, 0.01, 20
loader_tr, loader_te = get_loaders(bs2)
opt2 = optim.SGD(model_cnn.parameters(), lr=lr2)

model_cnn, hist_cnn, elapsed_cnn = train(
    model_cnn, loader_tr, loader_te, opt2,
    ep2, bs2, lr2,
    title="Exercise 2 -- CNN with SGD",
)

# ============================================================
# Exercise 3: CNN with Adam Optimizer
# Same CNN architecture, Adam (default params, lr=0.001).
# ============================================================

bs3, lr3, ep3 = 64, 0.001, 20
loader_tr, loader_te = get_loaders(bs3)

model_adam = ConvNet()
opt3 = optim.Adam(model_adam.parameters(), lr=lr3)

model_adam, hist_adam, elapsed_adam = train(
    model_adam, loader_tr, loader_te, opt3,
    ep3, bs3, lr3,
    title="Exercise 3 -- CNN with Adam",
)

# ============================================================
# Exercise 4: Residual Connections
# ResConvNet: CNN with residual blocks after each conv stage.
# ============================================================

class ResidualBlock(nn.Module):
    """Residual block: two conv+ReLU pairs with a skip connection.

    Computes F(x) + x, where F is two 3x3 conv+ReLU layers.

    Args:
        channels: int, number of input/output channels (kept constant).

    Input:  (B, channels, H, W).
    Output: (B, channels, H, W).
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        return out + x                      # residual connection


class ResConvNet(nn.Module):
    """CNN with residual blocks after each convolutional stage.

    Architecture: 3 stages of (Conv -> ReLU -> ResidualBlock -> MaxPool),
    followed by a linear classifier. Deeper than ConvNet.

    Input:  (B, 1, 28, 28) MNIST images.
    Output: (B, 10) raw logits.
    """
    def __init__(self):
        super().__init__()
        # Stage 1
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.res1  = ResidualBlock(8)
        self.pool1 = nn.MaxPool2d(2, 2)
        # Stage 2
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.res2  = ResidualBlock(16)
        self.pool2 = nn.MaxPool2d(2, 2)
        # Stage 3
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.res3  = ResidualBlock(32)
        # Classifier
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool1(self.res1(F.relu(self.conv1(x))))   # -> (B, 8, 14,14)
        x = self.pool2(self.res2(F.relu(self.conv2(x))))   # -> (B,16,  7, 7)
        x = self.res3(F.relu(self.conv3(x)))                 # -> (B,32,  7, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --- Exercise 4: Training ---
bs4, lr4, ep4 = 64, 0.001, 20
loader_tr, loader_te = get_loaders(bs4)

model_res = ResConvNet()
count_parameters(model_res)
opt4 = optim.Adam(model_res.parameters(), lr=lr4)

model_res, hist_res, elapsed_res = train(
    model_res, loader_tr, loader_te, opt4,
    ep4, bs4, lr4,
    title="Exercise 4 -- CNN with Residual Connections",
)

# ============================================================
# Exercise 5: Improve the CNN with Regularization
# Variant A: Batch Normalization    Variant B: Dropout
# ============================================================

# ---- Variant A: Batch Normalization ----
class ResConvNetBN(nn.Module):
    """ResConvNet with BatchNorm2d after every convolutional layer.

    Input:  (B, 1, 28, 28) MNIST images.
    Output: (B, 10) raw logits.
    """
    def __init__(self):
        super().__init__()
        # Stage 1
        self.conv1     = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1       = nn.BatchNorm2d(8)
        self.res1_c1   = nn.Conv2d(8, 8, 3, padding=1)
        self.res1_bn1  = nn.BatchNorm2d(8)
        self.res1_c2   = nn.Conv2d(8, 8, 3, padding=1)
        self.res1_bn2  = nn.BatchNorm2d(8)
        self.pool1     = nn.MaxPool2d(2, 2)
        # Stage 2
        self.conv2     = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2       = nn.BatchNorm2d(16)
        self.res2_c1   = nn.Conv2d(16, 16, 3, padding=1)
        self.res2_bn1  = nn.BatchNorm2d(16)
        self.res2_c2   = nn.Conv2d(16, 16, 3, padding=1)
        self.res2_bn2  = nn.BatchNorm2d(16)
        self.pool2     = nn.MaxPool2d(2, 2)
        # Stage 3
        self.conv3     = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3       = nn.BatchNorm2d(32)
        self.res3_c1   = nn.Conv2d(32, 32, 3, padding=1)
        self.res3_bn1  = nn.BatchNorm2d(32)
        self.res3_c2   = nn.Conv2d(32, 32, 3, padding=1)
        self.res3_bn2  = nn.BatchNorm2d(32)
        # Classifier
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        # Stage 1
        x = F.relu(self.bn1(self.conv1(x)))
        r = x
        x = F.relu(self.res1_bn1(self.res1_c1(x)))
        x = F.relu(self.res1_bn2(self.res1_c2(x)))
        x = self.pool1(x + r)
        # Stage 2
        x = F.relu(self.bn2(self.conv2(x)))
        r = x
        x = F.relu(self.res2_bn1(self.res2_c1(x)))
        x = F.relu(self.res2_bn2(self.res2_c2(x)))
        x = self.pool2(x + r)
        # Stage 3
        x = F.relu(self.bn3(self.conv3(x)))
        r = x
        x = F.relu(self.res3_bn1(self.res3_c1(x)))
        x = F.relu(self.res3_bn2(self.res3_c2(x)))
        x = x + r
        # Classifier
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ---- Variant B: Dropout ----
class ResConvNetDropout(nn.Module):
    """ResConvNet with Dropout2d after pooling and Dropout before FC.

    Args:
        p_spatial: float, dropout probability for Dropout2d layers.
        p_fc: float, dropout probability before the FC classifier.

    Input:  (B, 1, 28, 28) MNIST images.
    Output: (B, 10) raw logits.
    """
    def __init__(self, p_spatial=0.25, p_fc=0.5):
        super().__init__()
        # Stage 1
        self.conv1   = nn.Conv2d(1, 8, 3, padding=1)
        self.res1_c1 = nn.Conv2d(8, 8, 3, padding=1)
        self.res1_c2 = nn.Conv2d(8, 8, 3, padding=1)
        self.pool1   = nn.MaxPool2d(2, 2)
        self.drop1   = nn.Dropout2d(p_spatial)
        # Stage 2
        self.conv2   = nn.Conv2d(8, 16, 3, padding=1)
        self.res2_c1 = nn.Conv2d(16, 16, 3, padding=1)
        self.res2_c2 = nn.Conv2d(16, 16, 3, padding=1)
        self.pool2   = nn.MaxPool2d(2, 2)
        self.drop2   = nn.Dropout2d(p_spatial)
        # Stage 3
        self.conv3   = nn.Conv2d(16, 32, 3, padding=1)
        self.res3_c1 = nn.Conv2d(32, 32, 3, padding=1)
        self.res3_c2 = nn.Conv2d(32, 32, 3, padding=1)
        # Classifier
        self.drop_fc = nn.Dropout(p_fc)
        self.fc      = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        # Stage 1
        x = F.relu(self.conv1(x))
        r = x
        x = F.relu(self.res1_c1(x))
        x = F.relu(self.res1_c2(x))
        x = self.drop1(self.pool1(x + r))
        # Stage 2
        x = F.relu(self.conv2(x))
        r = x
        x = F.relu(self.res2_c1(x))
        x = F.relu(self.res2_c2(x))
        x = self.drop2(self.pool2(x + r))
        # Stage 3
        x = F.relu(self.conv3(x))
        r = x
        x = F.relu(self.res3_c1(x))
        x = F.relu(self.res3_c2(x))
        x = x + r
        # Classifier
        x = x.view(x.size(0), -1)
        x = self.drop_fc(x)
        return self.fc(x)

# --- Exercise 5(b): Training with Batch Normalization ---
bs5, lr5, ep5 = 64, 0.001, 20
loader_tr, loader_te = get_loaders(bs5)

model_bn = ResConvNetBN()
opt_bn   = optim.Adam(model_bn.parameters(), lr=lr5)
model_bn, hist_bn, elapsed_bn = train(
    model_bn, loader_tr, loader_te, opt_bn,
    ep5, bs5, lr5,
    title="Exercise 5 -- CNN + Batch Normalization",
)

# --- Exercise 5(b): Training with Dropout ---
model_do = ResConvNetDropout()
opt_do   = optim.Adam(model_do.parameters(), lr=lr5)
model_do, hist_do, elapsed_do = train(
    model_do, loader_tr, loader_te, opt_do,
    ep5, bs5, lr5,
    title="Exercise 5 -- CNN + Dropout",
)

# ============================================================
# Exercise 5(c): Comparison & Confusion Matrix
# ============================================================

def plot_confusion_matrix(model, loader, title="Confusion Matrix"):
    """Plot confusion matrix for a model on a given DataLoader.

    Args:
        model: nn.Module, trained model.
        loader: DataLoader, test data.
        title: str, plot title.
    """
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            all_preds.extend(model(X).argmax(1).cpu().numpy())
            all_true.extend(y.cpu().numpy())

    cm = confusion_matrix(all_true, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(ax=ax, cmap='Blues')
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_misclassified(model, loader, title="Misclassified Examples"):
    """Show 10 misclassified images, prioritizing one per true-label class.

    Args:
        model: nn.Module, trained model.
        loader: DataLoader, test data.
        title: str, plot title.
    """
    model.eval()
    all_errors = []

    # 1. Collect all errors in one pass
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(1)

            incorrect_mask = preds != y
            for x_err, y_err, p_err in zip(X[incorrect_mask], y[incorrect_mask], preds[incorrect_mask]):
                all_errors.append((x_err.cpu(), y_err.item(), p_err.item()))

    # 2. Select up to 10, one per unique true class first
    examples = []
    seen_classes = set()

    for err in all_errors:
        true_label = err[1]
        if true_label not in seen_classes:
            examples.append(err)
            seen_classes.add(true_label)
        if len(examples) == 10:
            break

    # 3. Pad with remaining errors if fewer than 10 classes had errors
    if len(examples) < 10:
        for err in all_errors:
            if not any(err is ex for ex in examples):
                examples.append(err)
            if len(examples) == 10:
                break

    # Plotting
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(title, fontsize=14)
    for idx, ax in enumerate(axes.flatten()):
        if idx < len(examples):
            img, true, pred = examples[idx]
            ax.imshow(img.squeeze(), cmap='gray')
            ax.set_title(f"True: {true}, Pred: {pred}", color='red', fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# Compare and select best model
bn_acc = hist_bn['test_acc'][-1]
do_acc = hist_do['test_acc'][-1]
print(f"BatchNorm final test accuracy: {100*bn_acc:.2f}%")
print(f"Dropout   final test accuracy: {100*do_acc:.2f}%")

if bn_acc >= do_acc:
    best_model, best_name = model_bn, "Batch Normalization"
else:
    best_model, best_name = model_do, "Dropout"

print(f"\nBest model: {best_name}")

# Confusion matrix
plot_confusion_matrix(best_model, loader_te,
                      title=f"Exercise 5(c) -- Confusion Matrix ({best_name})")

# --- Exercise 5(d): Misclassified Examples ---
plot_misclassified(best_model, loader_te,
                   title=f"Exercise 5(d) -- Misclassified Examples ({best_name})")
