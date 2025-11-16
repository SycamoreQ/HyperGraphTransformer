import os
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from hyper_new import HyperVisionNet

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Make seed device-aware
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_transforms(img_size=224):
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform_train, transform_val

@torch.no_grad()
def accuracy(pred, target):
    return (pred.argmax(dim=1) == target).float().mean().item()

def f1_macro(pred, target, num_classes):
    # simple macro-F1 on CPU
    y_true = target.detach().cpu().numpy()
    y_pred = pred.argmax(dim=1).detach().cpu().numpy()
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s))

def maybe_init_wandb(args, model):
    if not args.wandb:
        return None
    try:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args), name=args.run_name or None)
        wandb.watch(model, log="all", log_freq=100)
        return wandb
    except Exception as e:
        print(f"W&B init failed: {e}")
        return None

# -----------------------------------------------------------------
# CORRECTED GRADIENT ACCUMULATION FUNCTION
# -----------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, num_classes, wandb_ref=None):
    ACCUMULATION_STEPS = 4
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss, total_acc, total_f1, count = 0.0, 0.0, 0.0, 0

    # BUG FIX 1: zero_grad() must be *before* the loop
    optimizer.zero_grad(set_to_none=True)

    for i, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Note: No zero_grad() here!
        logits = model(images, labels)
        
        # Scale loss to average the gradients
        loss = ce(logits, labels) / ACCUMULATION_STEPS
        loss.backward()

        # BUG FIX 2: Only step and zero_grad() on the accumulation step
        if (i + 1) % ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            acc = accuracy(logits, labels)
            f1 = f1_macro(logits, labels, num_classes)
        
        # BUG FIX 3: Log the *un-scaled* loss and update total_loss correctly
        unscaled_loss = float(loss.item()) * ACCUMULATION_STEPS
        total_loss += unscaled_loss * images.size(0)
        total_acc += acc * images.size(0)
        total_f1  += f1  * images.size(0)
        count += images.size(0)
        if wandb_ref:
            wandb_ref.log({"train/loss_step": unscaled_loss,
                           "train/acc_step": acc,
                           "train/f1_step": f1})
            
        # BUG FIX 2 (cont.): Removed the incorrect `if (i + 1) % ACCUMULATION_STEPS != 0:` block

    # BUG FIX 4: Add final step for leftover gradients
    if (i + 1) % ACCUMULATION_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / count, total_acc / count, total_f1 / count
# -----------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, num_classes, split="val", wandb_ref=None):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss, total_acc, total_f1, count = 0.0, 0.0, 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images, labels)
        loss = ce(logits, labels)
        acc = accuracy(logits, labels)
        f1 = f1_macro(logits, labels, num_classes)
        total_loss += float(loss.item()) * images.size(0)
        total_acc  += acc * images.size(0)
        total_f1   += f1  * images.size(0)
        count += images.size(0)
    avg_loss = total_loss / count
    avg_acc  = total_acc / count
    avg_f1   = total_f1 / count
    if wandb_ref:
        wandb_ref.log({f"{split}/loss": avg_loss, f"{split}/acc": avg_acc, f"{split}/f1": avg_f1})
    return avg_loss, avg_acc, avg_f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to ImageFolder root with class subfolders")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)

    # -----------------------------------------------------------------
    # BUG FIX 5: Added device auto-detection
    # -----------------------------------------------------------------
    if torch.backends.mps.is_available():
        default_device = "mps"
    elif torch.cuda.is_available():
        default_device = "cuda"
    else:
        default_device = "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    # -----------------------------------------------------------------

    # model/hypergraph
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--dilation", type=int, default=1)
    parser.add_argument("--use_hyperedges", action="store_true")
    parser.add_argument("--num_clusters", type=int, default=8)
    parser.add_argument("--hyper_threshold", type=float, default=0.5)
    parser.add_argument("--patch_dim", type=int, default=96)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    # wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="vision-hypergraph-mri")
    parser.add_argument("--run_name", type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)
    transform_train, transform_val = get_transforms(args.img_size)
    train_dataset = ImageFolder(args.data_dir, transform=transform_train)
    val_dataset   = ImageFolder(args.data_dir, transform=transform_val)

    # Stratified split (train/val/test = 70/15/15)
    indices = list(range(len(train_dataset)))
    labels  = [train_dataset[i][1] for i in indices]
    tr_idx, tmp_idx = train_test_split(indices, test_size=0.3, stratify=labels, random_state=args.seed)
    tmp_labels = [labels[i] for i in tmp_idx]
    val_idx, te_idx = train_test_split(tmp_idx, test_size=0.5, stratify=tmp_labels, random_state=args.seed)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(tr_idx), num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, sampler=SubsetRandomSampler(val_idx), num_workers=0)

    # This will now correctly use "mps", "cuda", or "cpu"
    device = torch.device(args.device)
    print(f"--- Using device: {device} ---")

    # Note: You had this line twice, I removed the duplicate
    num_classes = len(train_dataset.classes)
    model = HyperVisionNet(
        num_classes=num_classes,
        in_chans=3,
        patch_embed_dim=args.patch_dim,
        gnn_hidden=args.hidden,
        gnn_layers=args.layers,
        k=args.k,
        dilation=args.dilation,
        use_hyperedges=args.use_hyperedges,
        num_clusters=args.num_clusters,
        hyper_threshold=args.hyper_threshold,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    wandb_ref = maybe_init_wandb(args, model)
    best_val = float("inf")
    print("--- Starting Training ---")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, optimizer, device, num_classes, wandb_ref)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device, num_classes, split="val", wandb_ref=wandb_ref)
        scheduler.step(val_loss)
        if wandb_ref:
            wandb_ref.log({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]})
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f} | {time.time()-t0:.1f}s")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_hypervision.pt")

    print("--- Training Finished ---")

if __name__ == "__main__":
    main()