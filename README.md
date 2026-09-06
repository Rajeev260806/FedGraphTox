# FedGraphTox-MOON — Project Notes

## Step 1: Raw Multi-Platform Social Stream

### Dataset chosen
**Twitter15**, from the standard rumor-detection benchmark released alongside:
- Ma, Gao, Wong (2017) — Detecting Rumors from Microblogs with Recurrent Neural Networks
- Bian et al. (2020) — Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks (AAAI 2020), whose public data release (`rumdetect2017.zip`) is the specific copy used here.

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

### One Limitation Faced 
- Raw text is only available for root tweet not for the reply/retweet

---

## Step 2: Phase A — Privacy-Preserving GNN Structural Encoder

### Sub-task 1 of 4 in Step 2: Build interaction graph (User nodes + Reply/Mention/Tag edges)

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
 
The diagram specifies edges as "Reply/Mention/Tag." The Twitter15 tree files only encode **reply/retweet propagation edges** — there is no parsed `@mention` or `#hashtag` information available in this dataset to construct separate mention/tag edges. 

