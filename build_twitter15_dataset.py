"""
Twitter15/16 -> PyTorch Geometric dataset builder
====================================================

Parses the standard Twitter15/16 rumor-detection folder structure:

    twitter15/
        label.txt          - one line per tree: "<label>:<root_tweet_id>"
        source_tweets.txt   - one line per tree: "<root_tweet_id>\t<raw text>"
        tree/
            <root_tweet_id>.txt   - one file per tree, propagation edges:
                ['parent_uid', 'parent_tweetid', 'parent_time']->['child_uid', 'child_tweetid', 'child_time']

Builds one torch_geometric.data.Data graph per tree:
    - nodes          = users who appear in the propagation tree
    - edges          = parent_user -> child_user (retweet/reply direction)
    - edge_attr      = time delta of the engagement (float, hours since root post)
    - x (node feat)  = currently a simple placeholder (in/out degree, depth) —
                        replace/extend with real user features if you have them
    - y (graph label)= mapped from label.txt (string label -> integer class)
    - root_text      = raw text of the source tweet, stored as a graph-level
                        attribute (NOT a per-node feature, since only the root
                        tweet has real text in this dataset)
    - root_idx       = index of the root node within this graph, so downstream
                        code (Phase B text embedding attachment) knows which
                        node to attach h_t to if it wants a per-node fusion
                        instead of a graph-level one.

Usage:
    python build_twitter15_dataset.py --data_dir /path/to/twitter15 --out_dir ./processed

Output:
    <out_dir>/graphs.pt   - a list[Data] object, directly loadable with torch.load
    <out_dir>/label_map.json - the string label -> int mapping used
"""

import argparse
import ast
import json
import os
import re
from pathlib import Path

import torch
from torch_geometric.data import Data


def parse_tree_file(path):
    """
    Parse one tree/<root_id>.txt file into:
        - list of (parent_uid, child_uid, time_delta) edges
        - root_uid (the user id of the root/source tweet author)

    Line format:
        ['parent_uid', 'parent_tweetid', 'parent_time']->['child_uid', 'child_tweetid', 'child_time']
    The very first line always has parent_uid == 'ROOT'; the child of that
    line is the root tweet's author.
    """
    edges = []
    root_uid = None
    node_re = re.compile(r"\[.*?\]")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "->" not in line:
                continue

            left_str, right_str = line.split("->")
            # ast.literal_eval safely parses "['a', 'b', 'c']" into a real list
            try:
                parent = ast.literal_eval(left_str.strip())
                child = ast.literal_eval(right_str.strip())
            except (ValueError, SyntaxError):
                # Fallback: manual bracket strip + split, in case of malformed quotes
                parent = [x.strip(" '\"") for x in left_str.strip().strip("[]").split(",")]
                child = [x.strip(" '\"") for x in right_str.strip().strip("[]").split(",")]

            parent_uid, _parent_tweetid, _parent_time = parent
            child_uid, _child_tweetid, child_time = child

            if parent_uid == "ROOT":
                root_uid = child_uid
                continue  # don't add a "ROOT" node into the graph itself

            try:
                t = float(child_time)
            except ValueError:
                t = 0.0

            edges.append((parent_uid, child_uid, t))

    return edges, root_uid


def build_graph(root_id, edges, root_uid, label_str, label_map, text):
    """
    Convert parsed edges into a PyG Data object.

    Node features (7 dims, richer than the original 3):
        0. in_degree    - how many direct engagements point TO this user
        1. out_degree   - how many direct engagements point FROM this user
        2. is_root      - 1.0 if this is the original tweet's author, else 0.0
        3. depth        - hops from the root along the propagation tree
                           (root=0, direct repliers=1, etc.)
        4. subtree_size - how many users are downstream of this node
                           (i.e. how much of the cascade this user "triggered")
        5. avg_response_time - average time delta of this node's OUTGOING
                           edges (how quickly people reacted to THIS user)
        6. min_response_time - fastest reaction time to this user (0.0 if
                           this node has no outgoing edges / is a leaf)

    WHY THESE: with only degree + is_root (the original 3), the model was
    stuck at chance-level accuracy. Depth and subtree_size capture the
    SHAPE of the cascade (how deep/branching it goes), and the response
    time features capture the SPEED of propagation -- both are
    structure-only signals (no text used) that prior rumor-detection
    literature (e.g. BiGCN) has found informative, since rumors and
    verified news tend to propagate differently in shape and speed.
    """
    users = set()
    for p, c, _t in edges:
        users.add(p)
        users.add(c)
    if root_uid is not None:
        users.add(root_uid)

    if not users:
        return None  # empty/degenerate tree, skip

    users = sorted(users)  # deterministic ordering
    uid_to_idx = {u: i for i, u in enumerate(users)}

    src, dst, edge_time = [], [], []
    for p, c, t in edges:
        src.append(uid_to_idx[p])
        dst.append(uid_to_idx[c])
        edge_time.append(t)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_time, dtype=torch.float).view(-1, 1)

    n = len(users)
    in_deg = torch.zeros(n)
    out_deg = torch.zeros(n)
    for s, d in zip(src, dst):
        out_deg[s] += 1
        in_deg[d] += 1

    root_idx = uid_to_idx.get(root_uid, 0)
    is_root = torch.zeros(n)
    is_root[root_idx] = 1.0

    # --- Build adjacency list (parent -> children) for BFS-based features ---
    children = {i: [] for i in range(n)}
    for s, d, t in zip(src, dst, edge_time):
        children[s].append((d, t))

    # --- Depth from root (BFS) ---
    depth = torch.zeros(n)
    visited = {root_idx}
    queue = [(root_idx, 0)]
    qi = 0
    while qi < len(queue):
        node, d = queue[qi]
        qi += 1
        for child, _t in children.get(node, []):
            if child not in visited:
                visited.add(child)
                depth[child] = d + 1
                queue.append((child, d + 1))

    # --- Subtree size (how many descendants each node has) ---
    subtree_size = torch.zeros(n)
    for node in range(n):
        seen = set()
        stack = [c for c, _t in children.get(node, [])]
        while stack:
            cur = stack.pop()
            if cur not in seen:
                seen.add(cur)
                stack.extend(c for c, _t in children.get(cur, []))
        subtree_size[node] = len(seen)

    # --- Response time stats per node (based on OUTGOING edge times) ---
    avg_response_time = torch.zeros(n)
    min_response_time = torch.zeros(n)
    for node in range(n):
        times = [t for _c, t in children.get(node, [])]
        if times:
            avg_response_time[node] = sum(times) / len(times)
            min_response_time[node] = min(times)

    x = torch.stack([in_deg, out_deg, is_root, depth, subtree_size,
                      avg_response_time, min_response_time], dim=1)  # [n, 7]

    y = torch.tensor([label_map[label_str]], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.root_id = str(root_id)
    data.root_idx = int(root_idx)
    data.root_text = text if text is not None else ""
    data.num_users = n

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path to twitter15 (or twitter16) folder containing "
                              "label.txt, source_tweets.txt, and tree/")
    parser.add_argument("--out_dir", type=str, default="./processed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_path = data_dir / "label.txt"
    text_path = data_dir / "source_tweets.txt"
    tree_dir = data_dir / "tree"

    assert label_path.exists(), f"Missing {label_path}"
    assert text_path.exists(), f"Missing {text_path}"
    assert tree_dir.exists(), f"Missing {tree_dir}"

    # --- Load labels ---
    id_to_label = {}
    all_labels = set()
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label_str, root_id = line.split(":")
            id_to_label[root_id] = label_str
            all_labels.add(label_str)

    label_map = {lbl: i for i, lbl in enumerate(sorted(all_labels))}
    print(f"Found {len(label_map)} label classes: {label_map}")

    # --- Load source text ---
    id_to_text = {}
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            root_id, text = parts
            id_to_text[root_id] = text

    # --- Build graphs ---
    graphs = []
    skipped = 0
    missing_text = 0

    for root_id, label_str in id_to_label.items():
        tree_file = tree_dir / f"{root_id}.txt"
        if not tree_file.exists():
            skipped += 1
            continue

        edges, root_uid = parse_tree_file(tree_file)
        text = id_to_text.get(root_id)
        if text is None:
            missing_text += 1

        g = build_graph(root_id, edges, root_uid, label_str, label_map, text)
        if g is None:
            skipped += 1
            continue

        graphs.append(g)

    print(f"Built {len(graphs)} graphs. Skipped {skipped} (missing/empty tree files). "
          f"{missing_text} graphs had no matching source text.")

    torch.save(graphs, out_dir / "graphs.pt")
    with open(out_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"Saved to {out_dir / 'graphs.pt'} and {out_dir / 'label_map.json'}")

    # Quick sanity print of the first graph
    if graphs:
        g0 = graphs[0]
        print("\nSample graph:")
        print(f"  root_id: {g0.root_id}")
        print(f"  num_nodes: {g0.num_nodes}, num_edges: {g0.num_edges}")
        print(f"  label: {g0.y.item()}")
        print(f"  root_text: {g0.root_text[:80]!r}")


if __name__ == "__main__":
    main()