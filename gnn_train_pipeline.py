"""
Phase A GNN Training: DataLoader + Model + Training Loop
============================================================

Continues from load_graphs.py, which already produced:
    train.pt, val.pt, test.pt   (stratified 80/10/10 split)

This script covers the remaining 3 pieces:
    3. DataLoader        - batches variable-sized graphs together
    4. GNN model          - GraphSAGE encoder + classification head
    5. Training loop      - trains on train.pt, evaluates on val.pt each
                             epoch, and reports final test.pt performance

NOTE: This is the PLAIN (non-private, non-adversarial) version -- i.e.
just the "GraphSAGE / GAT Encoder" bullet trained end-to-end on the
actual labels. Differential privacy (bullet 2) and adversarial debiasing
(bullet 3) get layered on top of this same model/loop once this baseline
is confirmed working correctly.

Usage:
    python gnn_train_pipeline.py --data_dir ./processed --epochs 20
"""

import argparse
import json

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool


# ======================================================================
# FEATURE NORMALIZATION
# ======================================================================
def compute_normalization_stats(train_graphs):
    """
    Computes per-feature mean and std from the TRAINING set only (never
    from val/test -- using val/test stats would leak information about
    those sets into how we scale the data, which is a subtle form of
    data leakage).

    WHY THIS IS NEEDED: our node features mix very different scales --
    is_root is 0 or 1, but subtree_size can be in the hundreds or
    thousands for large cascades. Feeding raw, unnormalized features
    like this into a neural network causes unstable training (huge
    initial gradients, loss spikes, then the network collapsing to
    predicting the same output for everything). Standardizing every
    feature to mean=0, std=1 (using TRAIN set statistics) fixes this.

    Additionally: count-based features (in_deg, out_deg, subtree_size)
    are heavily right-skewed (most nodes have small values, a few have
    huge values) -- we apply log1p (log(1+x)) BEFORE standardizing,
    which compresses those large outlier values and makes the
    distribution much more learnable.
    """
    all_x = torch.cat([g.x for g in train_graphs], dim=0)  # [total_nodes, 7]

    # log1p on the count-like columns: in_deg(0), out_deg(1), subtree_size(4)
    # (is_root, depth, avg/min_response_time are left as-is before standardizing)
    log_cols = [0, 1, 4]
    all_x = all_x.clone()
    for c in log_cols:
        all_x[:, c] = torch.log1p(all_x[:, c])

    mean = all_x.mean(dim=0)
    std = all_x.std(dim=0)
    std[std == 0] = 1.0  # avoid division by zero for any constant column

    return {"mean": mean, "std": std, "log_cols": log_cols}


def apply_normalization(graphs, stats):
    """Applies the log1p + standardization computed above to a list of graphs,
    IN PLACE (modifies each graph's .x tensor)."""
    for g in graphs:
        x = g.x.clone()
        for c in stats["log_cols"]:
            x[:, c] = torch.log1p(x[:, c])
        x = (x - stats["mean"]) / stats["std"]
        g.x = x
    return graphs


# ======================================================================
# 3. DATALOADER
# ======================================================================
def build_dataloaders(data_dir, batch_size):
    """
    Loads the three pre-split .pt files, NORMALIZES node features (fit on
    train, applied to all three splits), and wraps each in a PyG DataLoader.

    WHY A SPECIAL DATALOADER (not the plain PyTorch one): our graphs have
    different numbers of nodes/edges each. PyG's DataLoader knows how to
    "batch" several differently-sized graphs together into one combined
    block, while keeping track of which nodes belong to which original
    graph (via the `batch` vector) -- this is what global_mean_pool later
    uses to un-mix them back into one embedding per graph.
    """
    train_graphs = torch.load(f"{data_dir}/train.pt", weights_only=False)
    val_graphs = torch.load(f"{data_dir}/val.pt", weights_only=False)
    test_graphs = torch.load(f"{data_dir}/test.pt", weights_only=False)

    # Fit normalization stats on TRAIN only, apply to all three splits
    norm_stats = compute_normalization_stats(train_graphs)
    train_graphs = apply_normalization(train_graphs, norm_stats)
    val_graphs = apply_normalization(val_graphs, norm_stats)
    test_graphs = apply_normalization(test_graphs, norm_stats)

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_graphs)} graphs -> {len(train_loader)} batches")
    print(f"Val  : {len(val_graphs)} graphs -> {len(val_loader)} batches")
    print(f"Test : {len(test_graphs)} graphs -> {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


# ======================================================================
# 4. GNN MODEL
# ======================================================================
class GraphSAGEClassifier(torch.nn.Module):
    """
    Same GraphSAGE encoder structure from gnn_encoder.py, now with a
    classification head attached so it can actually be trained end-to-end
    against the real true/false/unverified/non-rumor labels.

        Input graph
            -> SAGEConv (in_channels -> hidden) + ReLU + Dropout
            -> SAGEConv (hidden -> hidden) + ReLU + Dropout
            -> global_mean_pool           (produces h_g, one vector per graph)
            -> Linear (hidden -> embedding_dim)   = h_g
            -> Linear (embedding_dim -> num_classes)  = logits

    Dropout added: randomly zeroes out some activations during training
    only. This prevents the model from over-relying on any single feature
    or path, which helps stabilize training (the bouncy val accuracy we
    saw last run is a classic sign of a model that could use this).
    """

    def __init__(self, in_channels, hidden_channels=64, embedding_dim=32,
                 num_classes=4, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.project = torch.nn.Linear(hidden_channels, embedding_dim)
        self.classifier = torch.nn.Linear(embedding_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)   # -> [num_graphs_in_batch, hidden_channels]
        h_g = self.project(x)            # -> [num_graphs_in_batch, embedding_dim]
        logits = self.classifier(h_g)    # -> [num_graphs_in_batch, num_classes]
        return logits, h_g


class GATClassifier(torch.nn.Module):
    """
    Alternative encoder using GAT (Graph Attention) instead of GraphSAGE.

    KEY DIFFERENCE FROM GRAPHSAGE: GraphSAGE aggregates neighbor
    information with a simple (mean-like) aggregation -- every neighbor
    contributes equally. GAT instead LEARNS an attention weight per
    neighbor, so the model can learn "this particular neighbor's
    engagement matters more than that one" rather than treating all
    neighbors identically. This can help when some users in a cascade
    are more influential/informative than others.

    Uses multi-head attention (heads=4) in the first layer for a richer
    representation, concatenated back down to hidden_channels for the
    second layer.
    """

    def __init__(self, in_channels, hidden_channels=64, embedding_dim=32,
                 num_classes=4, dropout=0.3, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=1, dropout=dropout)
        self.project = torch.nn.Linear(hidden_channels, embedding_dim)
        self.classifier = torch.nn.Linear(embedding_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        h_g = self.project(x)
        logits = self.classifier(h_g)
        return logits, h_g


# ======================================================================
# 5. TRAINING LOOP
# ======================================================================
def run_epoch(model, loader, optimizer, loss_fn, device, train=True):
    """
    Runs one full pass over a DataLoader.
    If train=True: updates model weights.
    If train=False: just evaluates (used for val/test), no gradient updates.
    Returns (average_loss, accuracy) for this pass.
    """
    model.train() if train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)

            if train:
                optimizer.zero_grad()

            logits, _h_g = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(logits, batch.y)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            preds = logits.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total += batch.num_graphs

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Directory containing train.pt, val.pt, test.pt")
    parser.add_argument("--label_map_path", type=str, default=None,
                         help="Optional label_map.json, for readable output")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                         help="L2 regularization strength -- penalizes large weights, "
                              "helps reduce the bouncy/unstable val accuracy we saw")
    parser.add_argument("--dropout", type=float, default=0.3,
                         help="Dropout probability applied after each conv layer")
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--model_type", type=str, default="sage", choices=["sage", "gat"],
                         help="'sage' = GraphSAGE (equal-weighted neighbor aggregation), "
                              "'gat' = Graph Attention (learns which neighbors matter more)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- 3. DataLoader ---
    train_loader, val_loader, test_loader = build_dataloaders(args.data_dir, args.batch_size)

    # Figure out dimensions from the data itself, rather than hardcoding
    sample_graph = train_loader.dataset[0]
    in_channels = sample_graph.x.shape[1]
    all_labels = [g.y.item() for g in train_loader.dataset] + \
                 [g.y.item() for g in val_loader.dataset] + \
                 [g.y.item() for g in test_loader.dataset]
    num_classes = max(all_labels) + 1
    print(f"\nNode feature dim: {in_channels}, Num classes: {num_classes}, Model: {args.model_type}\n")

    # --- 4. Model ---
    if args.model_type == "gat":
        model = GATClassifier(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            embedding_dim=args.embedding_dim,
            num_classes=num_classes,
            dropout=args.dropout,
        ).to(device)
    else:
        model = GraphSAGEClassifier(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            embedding_dim=args.embedding_dim,
            num_classes=num_classes,
            dropout=args.dropout,
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Reduces the learning rate when val_loss plateaus, instead of staying fixed the
    # whole run -- helps the bouncy accuracy settle down in later epochs.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                             factor=0.5, patience=3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # --- 5. Training loop ---
    print("=" * 60)
    print("Training")
    print("=" * 60)
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer, loss_fn, device, train=False)
        scheduler.step(val_loss)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = "  <- best so far"

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={current_lr:.5f}{marker}")

    # --- Final test evaluation, using the BEST validation checkpoint ---
    print("\n" + "=" * 60)
    print("Final Test Evaluation (using best val checkpoint)")
    print("=" * 60)
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc = run_epoch(model, test_loader, optimizer, loss_fn, device, train=False)
    print(f"Test loss: {test_loss:.4f}  |  Test accuracy: {test_acc:.4f}")

    if args.label_map_path:
        with open(args.label_map_path, "r") as f:
            label_map = json.load(f)
        print(f"\nLabel mapping used: {label_map}")


if __name__ == "__main__":
    main()