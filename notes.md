# FedGraphTox-MOON — Project Notes

**Status as of this document:** Step 1 (Data Collection) complete. Step 2 (Phase A), Sub-component 1 of 4 (Interaction Graph Construction) complete.

---

## Step 1: Raw Multi-Platform Social Stream (Data Collection)

### Dataset chosen
**Twitter15**, from the standard rumor-detection benchmark released alongside:
- Ma, Gao, Wong (2017) — *Detecting Rumors from Microblogs with Recurrent Neural Networks*
- Bian et al. (2020) — *Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks* (AAAI 2020), whose public data release (`rumdetect2017.zip`) is the specific copy used here.

### Why this dataset
- Live collection from X/Twitter API is no longer practically accessible for bulk historical/graph data at academic scale; Instagram has no public bulk API for this use case; Koo shut down in 2024. Twitter15 is a well-established, citable benchmark that provides both text and propagation-graph structure, which is what Phase A + Phase B jointly require.
- This substitution/limitation is to be explicitly documented in the paper's Data section, rather than implying live multi-platform scraping occurred.

### Files obtained
Folder: `twitter15/`
| File | Contents | Rows/Files |
|---|---|---|
| `label.txt` | `<label>:<root_tweet_id>` — ground truth class per tweet cascade | 1490 lines |
| `source_tweets.txt` | `<root_tweet_id>\t<raw tweet text>` — text of the **root tweet only** | 1490 lines |
| `tree/<root_id>.txt` | Propagation structure per cascade: `['parent_uid','parent_tweetid','parent_time']->['child_uid','child_tweetid','child_time']` | 1490 files |

### Labels
Four classes, standard to Twitter15/16:
- `true` — claim verified true
- `false` — claim verified false (misinformation)
- `unverified` — claim not yet confirmed either way
- `non-rumor` — ordinary tweet, not a disputed claim

### Known limitation (documented, not fixed)
- Raw text is available **only for the root tweet** of each cascade, not for intermediate reply/retweet nodes. This shapes how text (Phase B) is later attached to the graph (see Step 2 notes below).

---

## Step 2: Phase A — Privacy-Preserving GNN Structural Encoder

### Sub-component 1 of 4: Build interaction graph (User nodes + Reply/Mention/Tag edges)

**Status: Complete** (with documented scope limitation, see below)

#### What was built
A parser (`build_twitter15_dataset.py`) that converts the raw Twitter15 folder into a list of PyTorch Geometric `Data` graph objects — one graph per tweet cascade — saved as a single file: **`graphs.pt`**.

#### How each file was processed

1. **`label.txt` parsed** into a `root_id → label string` lookup, then all distinct label strings are sorted and mapped to integers (`label_map.json`), since models require numeric targets, not text.

2. **`source_tweets.txt` parsed** into a `root_id → raw text` lookup.

3. **Each `tree/<root_id>.txt` file parsed** line by line:
   - Each line follows `[parent_uid, parent_tweetid, parent_time] -> [child_uid, child_tweetid, child_time]`
   - The first line of every file (`parent_uid == 'ROOT'`) identifies the root tweet's author (`root_uid`)
   - All other lines are treated as directed edges: `parent_uid → child_uid`, meaning "child engaged in response to / downstream of parent"
   - The `child_time` value (hours since root post) is kept as an edge attribute (not yet used downstream, but preserved)

4. **Graph construction per tree:**
   - **Nodes** = every distinct user ID appearing in that tree (as parent, child, or root)
   - **Edges** = every parent→child relationship found, directional
   - **Node features (`x`)** = 3 simple numeric values per user: in-degree, out-degree, is-root flag (placeholder features — no real user profile data was available/used)
   - **Edge features (`edge_attr`)** = time delta of the engagement
   - **Graph-level label (`y`)** = the integer-mapped class from `label.txt`
   - **Graph-level text (`root_text`)** = the raw source tweet text, attached once per graph (not per node), since only the root has real text
   - **`root_idx`** = stored separately, so later fusion steps know which specific node is the root if per-node (rather than graph-level) text attachment is ever needed

#### Result

- **1490 total graphs**, one per tweet cascade, saved in `graphs.pt`
- Verified on real data: e.g., one sample graph had 145 nodes and 147 edges for a single root tweet, confirming cascades vary substantially in size (a few users up to 100+), and are correctly parsed as independent graphs
- Verified `label_map.json` correctly reflects all 4 classes found in the real 1490-row `label.txt`
- Confirmed `len(graphs) == 1490` after loading `graphs.pt`, i.e. no trees were silently dropped

#### Scope limitation (to state explicitly in documentation)

The diagram specifies edges as "Reply/Mention/Tag." The Twitter15 tree files only encode **reply/retweet propagation edges** — there is no parsed `@mention` or `#hashtag` information available in this dataset to construct separate mention/tag edges. This is consistent with how this dataset is used across the rumor-detection literature (e.g., BiGCN, Dynamic-GCN, HPCL all use retweet/reply-only propagation graphs from this same dataset). Mention/tag edges were not constructed and would require additional raw-text parsing of `source_tweets.txt` if added later — treated as a documented dataset limitation, not an implementation gap.

#### Files produced so far
- `build_twitter15_dataset.py` — the parser script (later updated, see sub-component 4 below)
- `graphs.pt` — 1490 PyG `Data` graph objects
- `label_map.json` — string label → integer class mapping

---

## Step 2, Sub-component 2: Train/Val/Test Split

**Status: Complete**

- Added a stratified 80/10/10 split (`load_graphs.py`, `--make_split` flag) using scikit-learn's `train_test_split`, done in two steps (test carved off first, then train/val split from the remainder) since sklearn only splits two ways at a time.
- **Stratified by label** — ensures each split preserves the same class proportions as the full dataset, rather than risking a random split that under-represents a class in test/val by chance.
- Verified on real data: 1192 train / 149 val / 149 test, with balanced label distribution confirmed in each split (e.g. train: non-rumor 299, unverified 299, true 298, false 296).
- Seed fixed at 42 for reproducibility across all future runs.
- Output: `train.pt`, `val.pt`, `test.pt` saved alongside `graphs.pt`.

---

## Step 2, Sub-component 4: GraphSAGE / GAT Encoder (Output: h_g)

**Status: Complete.** Built, tested, debugged, and tuned across several iterations before reaching a defensible result. Documenting the full journey here since the debugging process itself is a legitimate part of the methodology.

*(Note: tackled before sub-component 2/Differential Privacy and sub-component 3/Adversarial Debiasing, since those wrap around a trained encoder — it made sense to first confirm the base encoder learns something real before layering privacy/debiasing on top of it.)*

### Pipeline components built
Three pieces on top of the graph construction (sub-component 1) and split (sub-component 2, train/val/test):
- **DataLoader** (`gnn_train_pipeline.py: build_dataloaders`) — loads `train.pt`/`val.pt`/`test.pt`, wraps each in a PyG `DataLoader` capable of batching variable-sized graphs together (tracked via a `batch` index vector).
- **Model** — a 2-layer graph encoder (GraphSAGE or GAT, see below) + global mean pooling (produces `h_g`, one fixed-size vector per graph, regardless of how many nodes the original cascade had) + a linear classification head for end-to-end training.
- **Training loop** (`run_epoch` + `main`) — trains per epoch, evaluates on val after each epoch, keeps the best-validation-accuracy checkpoint (not just the last epoch), and reports final held-out test performance using that checkpoint.

### Iteration 1: Minimal features (in-degree, out-degree, is_root) — 3 dims
- Result: **~29-31% test accuracy**, essentially indistinguishable from chance (25% for 4 balanced classes; confirmed classes are balanced: ~296-299 each in the 1192-graph training set, ruling out majority-class guessing as the explanation).
- Conclusion: minimal degree-based structural features alone do not carry enough signal to separate the 4 classes.

### Iteration 2: Richer structural features — 7 dims
Added, still using zero text (structure-only, consistent with this sub-component's scope):
- `depth` — hops from root (via BFS over the propagation tree)
- `subtree_size` — number of downstream descendants per node (how much of the cascade a user triggered)
- `avg_response_time` / `min_response_time` — derived from the previously-unused `edge_attr` time-delta field, capturing propagation speed per node

Feature correctness was manually verified against raw tree data before training (hand-checked depth, subtree size, and response time values against a known 3-node sample tree).

- **First attempt result: training collapsed.** Loss started anomalously high (~24.6, vs. the expected ~1.39 for 4-class cross-entropy at initialization), then collapsed back to chance-level accuracy (~24-25%) for the rest of training.
- **Diagnosis:** feature scale mismatch. `is_root` ranges 0-1, while `subtree_size` ranges up to ~2990 (the largest graph in the dataset has 2990 nodes). Feeding unnormalized features spanning such different scales into the network caused unstable gradients at the start of training, after which the network collapsed to predicting near-uniform output.
- **Fix:** added a normalization step (`compute_normalization_stats` / `apply_normalization` in `gnn_train_pipeline.py`):
  - `log1p` transform applied to the three skewed count-based columns (`in_degree`, `out_degree`, `subtree_size`) to compress large outliers
  - Standardization (mean=0, std=1) applied to all 7 features, with statistics computed **only from the training set** and then applied unchanged to val/test (avoiding data leakage from val/test statistics)
- Fix was verified on a controlled synthetic dataset (label deliberately tied to a large-scale feature, mimicking the real scale problem) before being trusted on real data — confirmed loss started sane and the model successfully learned (85-95% val accuracy on the synthetic case).
- **Result after normalization, on real data: ~38% test accuracy** (GraphSAGE, default hyperparameters) — meaningfully above chance (+13 points), confirming richer structural features do carry real signal, though training curves were still somewhat unstable/bouncy epoch-to-epoch.

### Iteration 3: Training stabilization + model comparison
To address the bounciness in validation accuracy from Iteration 2, added:
- **Dropout** (0.3 default) after each graph conv layer
- **Weight decay** (L2 regularization, 5e-4 default) on the optimizer
- **ReduceLROnPlateau learning rate scheduler** — halves the learning rate when validation loss plateaus for 3 epochs, rather than using one fixed rate throughout
- **GAT (Graph Attention Network) added as an alternative encoder**, alongside the original GraphSAGE, for direct comparison. Key difference: GraphSAGE aggregates all neighbors equally; GAT learns per-neighbor attention weights, letting the model learn that some users in a cascade (e.g. ones that triggered large downstream bursts) matter more than others.

**Final comparison, both trained 30 epochs on real data:**

| Model | Best val accuracy | Test accuracy |
|---|---|---|
| GraphSAGE | 43.6% | 44.3% |
| **GAT** | **47.0%** | **53.0%** |

### Decision: GAT adopted as the Phase A encoder going forward
GAT outperformed GraphSAGE by a clear margin (53.0% vs 44.3% test accuracy, both well above the 25% chance baseline for 4 classes) and produced visibly more stable training curves. This is the encoder that will carry forward into sub-components 2 (Differential Privacy) and 3 (Adversarial Debiasing), and ultimately into the fused Step 5/6 pipeline.

### Interpretation for the paper
This iterative result is itself a legitimate, reportable finding: standalone structural signal (no text) is informative but limited (~53% on a 4-class task, well above chance but far from strong), which directly motivates and justifies the architecture's decision to fuse Phase A's graph embedding with Phase B's text/emotion signal (Step 5) rather than relying on graph structure alone.

### Files produced in this sub-component
- `gnn_encoder.py` — initial minimal forward-pass-only encoder (superseded, kept for reference/teaching purposes)
- `gnn_train_pipeline.py` — full DataLoader + model (GraphSAGE and GAT) + normalization + training loop, current working version
- `build_twitter15_dataset.py` — updated to produce the 7-dim enriched node features (supersedes the original 3-dim version)

---
