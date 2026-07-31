# Transparent chess self-learning pipeline

This package continues from the supervised bootstrap checkpoint and implements:

1. policy/value-guided PUCT search;
2. batched full-game self-play;
3. sparse on-disk replay storage;
4. candidate fine-tuning from self-play plus optional human-game rehearsal;
5. paired-color arena evaluation;
6. automatic promotion or rejection of each candidate.

The default model is the current 12-block, 128-channel network. Checkpoint metadata remains the source of truth when loading a model.

## Install

From this directory:

```powershell
pip install -e .
```

Confirm CUDA and `bulletchess`:

```powershell
python -c "import torch, bulletchess; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(len(bulletchess.Board().legal_moves()))"
```

## Configure

Edit `selflearn_config.yaml`:

```yaml
run:
  root: "S:/data/chess/self_learning_12x128"
  initial_checkpoint: "S:/data/chess/bootstrap_runs/resnet12x128/best.pt"
```

The initial checkpoint can be the best bootstrap checkpoint or any later
champion checkpoint with the same model architecture.

## Mandatory smoke test

Run this before generating hundreds of games:

```powershell
python -m chess_selflearn.smoke_test `
  --checkpoint "S:/data/chess/bootstrap_runs/resnet12x128/best.pt"
```

It verifies the 34-plane state encoding, unique legal action indexes, checkpoint
compatibility, neural inference, and an eight-simulation MCTS search.

Also run the CPU-only optimized-search structural test:

```powershell
python -m chess_selflearn.optimization_smoke_test
```

It checks apply/undo board restoration, in-place previous-position encoding,
multiple leaves per tree, virtual-reservation cleanup, and exact visit totals.

## Low-cost first iteration

Before using the full defaults, temporarily set:

```yaml
self_play:
  games_per_iteration: 8
  concurrent_games: 8
  simulations: 32
  inference_batch_size: 32
  leaves_per_tree: 2

training:
  steps_per_iteration: 100
  checkpoint_every_steps: 50

arena:
  games: 4
  concurrent_games: 4
  simulations: 32
```

Then run:

```powershell
python -m chess_selflearn.iterate --config selflearn_config.yaml --iterations 1
```

After the smoke iteration succeeds, restore the supplied defaults.

## Full iteration

```powershell
python -m chess_selflearn.iterate `
  --config selflearn_config.yaml `
  --iterations 1
```

The orchestrator performs:

```text
champion.pt
    -> self-play games
    -> replay HDF5
    -> candidate training
    -> candidate.pt
    -> paired arena
    -> promote or reject
```

Run several sequential iterations with:

```powershell
python -m chess_selflearn.iterate `
  --config selflearn_config.yaml `
  --iterations 5
```

## Run stages separately

Generate replay:

```powershell
python -m chess_selflearn.self_play `
  --config selflearn_config.yaml `
  --iteration 1 `
  --checkpoint "S:/data/chess/self_learning_12x128/champion.pt"
```

Train a candidate:

```powershell
python -m chess_selflearn.train_candidate `
  --config selflearn_config.yaml `
  --iteration 1 `
  --champion "S:/data/chess/self_learning_12x128/champion.pt"
```

Evaluate it:

```powershell
python -m chess_selflearn.arena `
  --config selflearn_config.yaml `
  --iteration 1 `
  --champion "S:/data/chess/self_learning_12x128/champion.pt" `
  --candidate "S:/data/chess/self_learning_12x128/iterations/iteration_000001/candidate.pt"
```

## Output layout

```text
S:/data/chess/self_learning_12x128/
├── champion.pt
├── run_state.json
├── history.json
├── replay_manifest.json
└── iterations/
    └── iteration_000001/
        ├── selfplay.h5
        ├── games.jsonl
        ├── selfplay_summary.json
        ├── candidate_step_001000.pt
        ├── candidate.pt
        ├── candidate_metrics.jsonl
        ├── candidate_tensorboard/
        ├── arena_games.jsonl
        └── arena_summary.json
```

## Replay format

Each self-play position stores:

```text
states                 uint8    [N, 34, 8, 8]
policy_indices         uint16   [N, 256]
policy_probabilities   float16  [N, 256]
policy_count           uint16   [N]
values                 int8     [N]
plies                  uint16   [N]
game_ids               uint64   [N]
```

The policy target is sparse. Only legal root actions and their normalized MCTS
visit counts are retained. Unused action slots contain the `65535` sentinel and
zero probability.

Inspect accumulated replay files with:

```powershell
python -m chess_selflearn.inspect_replay `
  --run-root "S:/data/chess/self_learning_12x128"
```

## Search behavior

Self-play defaults:

```yaml
simulations: 256
concurrent_games: 96
inference_batch_size: 128
leaves_per_tree: 4
virtual_loss: 1.0
dirichlet_alpha: 0.30
dirichlet_epsilon: 0.25
temperature_moves: 20
temperature: 1.0
```

Each tree can reserve several leaves per inference round using virtual visits
and virtual loss. The scheduler fills batches across 64-128 active games, while
each tree traverses one reusable `bulletchess.Board` with apply/undo rather than
copying a board and move history for every simulation. Exact duplicate
state/action pairs are evaluated once. Legal policy indexes are gathered on the
GPU, so dense 4,672-logit policy tensors are never copied back to the CPU.
After a move, the selected child becomes the next root and its subtree is
retained.

Arena search uses no root noise and deterministic maximum-visit selection.
Every opening is played twice with candidate colors reversed.


## Optimized MCTS hot path

Version 0.3 replaces the original latency-bound search path with:

- a rolling active-game pool that replenishes completed games immediately;
- 96 concurrent games by default, intended to be benchmarked at 64/96/128;
- one reusable mutable board per tree with apply/undo traversal;
- in-place previous-position encoding with undo/apply, with no history copy;
- vectorized conversion of 12 or 24 bitboards into tensor planes;
- cached PUCT parent exploration constants;
- several virtually reserved leaves per tree and inference batch;
- GPU-side legal-action gather before device-to-host transfer;
- self-play JSON metrics for simulations/sec, batch sizes, duplicates, and
  selection/evaluation/backup time.

The replay and checkpoint formats are unchanged.

## Candidate training

Candidate training starts from the current champion, not from random weights.
The default source mixture is:

```yaml
bootstrap_mix_ratio: 0.25
```

This means approximately 75 percent self-play batches and 25 percent original
human-game batches. The human data acts as rehearsal and reduces catastrophic
forgetting while the replay set is still small.

Self-play policy loss is cross-entropy against the sparse MCTS visit
distribution. Bootstrap policy loss remains ordinary one-hot cross-entropy.
Both use the final game result from the side-to-move perspective for the value
head.

## Promotion

The supplied defaults use:

```yaml
arena:
  games: 40
  simulations: 400
  promotion_score: 0.55
```

Candidate score is:

```text
(wins + 0.5 * draws) / games
```

A promoted `candidate.pt` atomically replaces `champion.pt`. Rejected
candidates and their replay data remain available for inspection.

Forty arena games are suitable for development but noisy. Increase to 80-200
once self-play is stable and promotion decisions matter more than iteration
speed.

## 4060 Ti tuning

Start with the supplied values. Change one item at a time.

The supplied starting point is:

```yaml
self_play:
  concurrent_games: 96
  inference_batch_size: 128
  leaves_per_tree: 4
  virtual_loss: 1.0
```

Measure 64, 96, and 128 active games on the actual RTX 4060 Ti rather than
choosing from utilization alone:

```powershell
python -m chess_selflearn.benchmark_search `
  --config selflearn_config.yaml `
  --checkpoint "S:/data/chess/bootstrap_runs/resnet12x128/best.pt" `
  --concurrency 64 96 128 `
  --simulations 64 `
  --plies 6 `
  --output "S:/data/chess/search_benchmark.json"
```

Select the setting with the highest sustained `simulations_per_second`, while
also checking mean requested/unique inference batch sizes. If self-play runs
out of memory, reduce `inference_batch_size` first, then concurrent games.

If candidate training runs out of memory:

```yaml
training:
  batch_size: 384
```

then 256 if required.

If replay reading starves the GPU:

```yaml
training:
  num_workers: 4
  read_chunk_size: 8192
```

## Intentional version-one choices

- No resignation during self-play.
- Draw claims are accepted automatically.
- No transposition table yet.
- No opening book is used for self-play; Dirichlet noise produces diversity.
- Arena uses a small fixed set of paired opening prefixes.
- Multiple leaves are batched across trees in one Python process rather than
  distributed across actor processes.
- Arena trees are rebuilt every move because candidate and champion are
  different evaluators; reusing a subtree evaluated by the other model would
  contaminate the comparison.

These choices keep the implementation readable and auditable. Later speed work
can add multiprocessing actors, a transposition table, CUDA graphs, larger
networks, and asynchronous replay generation without changing the replay or
checkpoint concepts.

# LAN chess website and training dashboard

This package now includes a FastAPI website that:

- lists every readable checkpoint under configured roots;
- lets each LAN user start an independent game against any listed model;
- validates legal moves and displays SAN notation;
- uses legal-move-masked neural MCTS for model moves;
- supports White or Black and configurable search strength;
- reads bootstrap TensorBoard scalars;
- reads self-play, candidate-training, and arena statistics;
- requires no browser CDN or internet connection.

The policy head in this package uses the corrected square-major layout:

```python
policy = self.policy_head(x).permute(0, 2, 3, 1).flatten(start_dim=1)
```

## Configure paths

Edit `web_config.yaml`. The supplied defaults expect:

```yaml
models:
  roots:
    - "S:/data/chess/bootstrap_runs"
    - "S:/data/chess/self_learning_12x128"

stats:
  bootstrap_roots:
    - "S:/data/chess/bootstrap_runs"
  self_learning_roots:
    - "S:/data/chess/self_learning_12x128"
```

Roots may be exact run folders or parents containing several runs. Checkpoint discovery is recursive.

## Install and run

From PowerShell in this directory:

```powershell
pip install -e .
python -m chess_web --config web_config.yaml
```

Equivalent installed command:

```powershell
chess-lan-web --config web_config.yaml
```

The terminal prints both the loopback URL and the computer's LAN URL. The server binds to `0.0.0.0`, so another device can browse to:

```text
http://YOUR-PC-IP:8000
```

Find the Windows IPv4 address with:

```powershell
ipconfig
```

If Windows Firewall blocks other devices, run an elevated PowerShell window once:

```powershell
New-NetFirewallRule `
  -DisplayName "Neural Chess LAN" `
  -Direction Inbound `
  -Protocol TCP `
  -LocalPort 8000 `
  -Action Allow
```

## Pages

- `/` — play against a checkpoint.
- `/stats` — bootstrap and self-learning statistics.
- `/docs` — FastAPI's generated API documentation.

## Runtime behavior

- Models load lazily when first selected.
- The model cache retains up to `max_loaded_models` models and evicts least-recently-used models.
- AI requests are serialized by default with `max_concurrent_ai: 1`, which is appropriate for an RTX 4060 Ti.
- A browser request waits for its model move; other LAN users remain connected and queue for the GPU.
- Games are held in server memory. Restarting the server clears active games.
- There is no login or encryption. Use it only on a trusted LAN and do not port-forward it to the internet.

## Checkpoint compatibility

The loader supports both bootstrap and self-learning payloads when they contain:

```text
model
config.model
```

The checkpoint's model configuration determines channel count and residual-block count. An unreadable or incompatible `.pt` remains visible as an error entry but cannot be selected.

Old checkpoints trained before the corrected policy permutation have matching tensor shapes and will load, but their learned policy layout is incompatible. Keep those outside the configured roots or clearly segregate them.
