"""
Load and access all graphs from graphs.pt
============================================

Simple utility to load the 1490 graph objects saved by
build_twitter15_dataset.py, demonstrates how to access/iterate them, and
creates a stratified 80/10/10 train/val/test split.

Usage:
    # Just inspect the dataset:
    python load_graphs.py --graphs_path ./processed/graphs.pt --label_map_path ./processed/label_map.json

    # Inspect AND create the train/val/test split files:
    python load_graphs.py --graphs_path ./processed/graphs.pt --label_map_path ./processed/label_map.json --make_split --out_dir ./processed
"""

import argparse
import json
from collections import Counter

import torch
from sklearn.model_selection import train_test_split


def make_stratified_split(graphs, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Splits the list of graphs into train/val/test sets, STRATIFIED by label.

    "Stratified" means: each split (train/val/test) preserves roughly the
    SAME proportion of each class (true/false/unverified/non-rumor) as the
    full dataset. This matters because a plain random split could, by
    chance, put almost all of one rare class into the test set only --
    stratification prevents that.

    Implementation: sklearn's train_test_split only splits into 2 parts at
    a time, so we do it in two steps:
        Step 1: split off the TEST set first (10%)
        Step 2: split the remaining 90% into TRAIN (80% of total) and
                 VAL (10% of total) -- i.e. VAL is 1/9 of the remaining 90%
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train/val/test ratios must sum to 1.0"

    labels = [g.y.item() for g in graphs]

    # Step 1: carve out the test set (10%), stratified by label
    train_val_graphs, test_graphs, train_val_labels, _test_labels = train_test_split(
        graphs, labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Step 2: split the remaining 90% into train (80%) and val (10%)
    # val_ratio_of_remaining = val_ratio / (train_ratio + val_ratio)
    val_ratio_of_remaining = val_ratio / (train_ratio + val_ratio)
    train_graphs, val_graphs, _train_labels, _val_labels = train_test_split(
        train_val_graphs, train_val_labels,
        test_size=val_ratio_of_remaining,
        stratify=train_val_labels,
        random_state=seed,
    )

    return train_graphs, val_graphs, test_graphs


def print_split_summary(name, split_graphs, idx_to_label=None):
    labels = [g.y.item() for g in split_graphs]
    if idx_to_label:
        counts = Counter(idx_to_label[l] for l in labels)
    else:
        counts = Counter(labels)
    print(f"{name:<6}: {len(split_graphs):>5} graphs  |  label distribution: {dict(counts)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_path", type=str, required=True,
                         help="Path to graphs.pt")
    parser.add_argument("--label_map_path", type=str, default=None,
                         help="Optional path to label_map.json, to print human-readable labels")
    parser.add_argument("--make_split", action="store_true",
                         help="If set, also creates a stratified 80/10/10 train/val/test split "
                              "and saves train.pt, val.pt, test.pt to --out_dir")
    parser.add_argument("--out_dir", type=str, default="./processed",
                         help="Directory to save train.pt / val.pt / test.pt (used only with --make_split)")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for the split, for reproducibility")
    args = parser.parse_args()

    # --- Load all graphs ---
    graphs = torch.load(args.graphs_path, weights_only=False)
    print(f"Loaded {len(graphs)} graphs from {args.graphs_path}\n")

    # --- Optional: load label map so we can decode y -> label string ---
    idx_to_label = None
    if args.label_map_path:
        with open(args.label_map_path, "r") as f:
            label_map = json.load(f)  # e.g. {"false": 0, "non-rumor": 1, "true": 2, "unverified": 3}
        idx_to_label = {v: k for k, v in label_map.items()}  # reverse it: int -> string

    # --- Access a SINGLE graph by index ---
    print("=" * 60)
    print("Example: accessing graph at index 0")
    print("=" * 60)
    g0 = graphs[0]
    print(f"root_id     : {g0.root_id}")
    print(f"num_nodes   : {g0.num_nodes}")
    print(f"num_edges   : {g0.num_edges}")
    print(f"x shape     : {g0.x.shape}          (node features: in-deg, out-deg, is_root)")
    print(f"edge_index  : {g0.edge_index.shape} (2 x num_edges)")
    print(f"edge_attr   : {g0.edge_attr.shape}  (time delta per edge)")
    print(f"y (label id): {g0.y.item()}")
    if idx_to_label:
        print(f"y (label)   : {idx_to_label[g0.y.item()]}")
    print(f"root_idx    : {g0.root_idx}")
    print(f"root_text   : {g0.root_text[:80]!r}")

    # --- Access a graph by its root_id instead of list index ---
    print("\n" + "=" * 60)
    print("Example: looking up a graph by its root_id")
    print("=" * 60)
    root_id_to_graph = {g.root_id: g for g in graphs}  # build a lookup dict once
    target_root_id = g0.root_id  # just reusing the first one's id as a demo
    found = root_id_to_graph.get(target_root_id)
    print(f"Found graph for root_id={target_root_id}: {found.num_nodes} nodes, {found.num_edges} edges")

    # --- Iterate over ALL graphs (this is the pattern you'll use in training loops) ---
    print("\n" + "=" * 60)
    print("Example: iterating over all graphs (showing first 5 only)")
    print("=" * 60)
    for i, g in enumerate(graphs[:5]):
        label_str = idx_to_label[g.y.item()] if idx_to_label else g.y.item()
        print(f"[{i}] root_id={g.root_id:<20} nodes={g.num_nodes:<5} edges={g.num_edges:<5} label={label_str}")

    # --- Basic dataset statistics (often useful for your paper's Data section) ---
    print("\n" + "=" * 60)
    print("Dataset statistics across all graphs")
    print("=" * 60)
    node_counts = [g.num_nodes for g in graphs]
    edge_counts = [g.num_edges for g in graphs]
    print(f"Total graphs        : {len(graphs)}")
    print(f"Nodes per graph     : min={min(node_counts)}, max={max(node_counts)}, "
          f"avg={sum(node_counts)/len(node_counts):.1f}")
    print(f"Edges per graph     : min={min(edge_counts)}, max={max(edge_counts)}, "
          f"avg={sum(edge_counts)/len(edge_counts):.1f}")

    if idx_to_label:
        label_counts = Counter(idx_to_label[g.y.item()] for g in graphs)
        print(f"Label distribution  : {dict(label_counts)}")
    else:
        label_counts = Counter(g.y.item() for g in graphs)
        print(f"Label distribution (raw ids): {dict(label_counts)}")

    # --- Stratified train/val/test split (only runs if --make_split is passed) ---
    if args.make_split:
        print("\n" + "=" * 60)
        print("Creating stratified 80/10/10 train/val/test split")
        print("=" * 60)

        train_graphs, val_graphs, test_graphs = make_stratified_split(
            graphs, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=args.seed
        )

        print_split_summary("Train", train_graphs, idx_to_label)
        print_split_summary("Val", val_graphs, idx_to_label)
        print_split_summary("Test", test_graphs, idx_to_label)

        assert len(train_graphs) + len(val_graphs) + len(test_graphs) == len(graphs), \
            "Split sizes don't add back up to the original total -- something went wrong"

        import os
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(train_graphs, f"{args.out_dir}/train.pt")
        torch.save(val_graphs, f"{args.out_dir}/val.pt")
        torch.save(test_graphs, f"{args.out_dir}/test.pt")

        print(f"\nSaved: {args.out_dir}/train.pt, {args.out_dir}/val.pt, {args.out_dir}/test.pt")
        print(f"(seed={args.seed} -- keep this fixed across runs so your split stays reproducible "
              f"and you never accidentally leak test graphs into training)")


if __name__ == "__main__":
    main()