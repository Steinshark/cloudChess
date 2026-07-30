# Transparent chess self-learning pipeline

This package continues from the supervised bootstrap checkpoint and implements:

1. policy/value-guided PUCT search;
2. batched full-game self-play;
3. sparse on-disk replay storage;
4. candidate fine-tuning from self-play plus optional human-game rehearsal;
5. paired-color arena evaluation;
6. automatic promotion or rejection of each candidate.

The model architecture and checkpoint keys match the earlier 8-block,
128-channel bootstrap package.

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
  root: "S:/data/chess/self_learning"
  initial_checkpoint: "S:/data/chess/bootstrap_runs/resnet8x128/best.pt"
```

The initial checkpoint can be the best bootstrap checkpoint or any later
champion checkpoint with the same model architecture.

## Mandatory smoke test

Run this before generating hundreds of games:

```powershell
python -m chess_selflearn.smoke_test `
  --checkpoint "S:/data/chess/bootstrap_runs/resnet8x128/best.pt"
```

It verifies the 34-plane state encoding, unique legal action indexes, checkpoint
compatibility, neural inference, and an eight-simulation MCTS search.

## Low-cost first iteration

Before using the full defaults, temporarily set:

```yaml
self_play:
  games_per_iteration: 8
  concurrent_games: 8
  simulations: 32

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
  --checkpoint "S:/data/chess/self_learning/champion.pt"
```

Train a candidate:

```powershell
python -m chess_selflearn.train_candidate `
  --config selflearn_config.yaml `
  --iteration 1 `
  --champion "S:/data/chess/self_learning/champion.pt"
```

Evaluate it:

```powershell
python -m chess_selflearn.arena `
  --config selflearn_config.yaml `
  --iteration 1 `
  --champion "S:/data/chess/self_learning/champion.pt" `
  --candidate "S:/data/chess/self_learning/iterations/iteration_000001/candidate.pt"
```

## Output layout

```text
S:/data/chess/self_learning/
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
  --run-root "S:/data/chess/self_learning"
```

## Search behavior

Self-play defaults:

```yaml
simulations: 256
concurrent_games: 32
dirichlet_alpha: 0.30
dirichlet_epsilon: 0.25
temperature_moves: 20
temperature: 1.0
```

One leaf is selected from each active game per search round. Those leaf states
are evaluated together on the GPU. Exact duplicate tensors are evaluated only
once per batch. After a move, the selected child becomes the next root, so its
searched subtree is retained.

Arena search uses no root noise and deterministic maximum-visit selection.
Every opening is played twice with candidate colors reversed.

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

If self-play GPU utilization is low:

```yaml
self_play:
  concurrent_games: 48
  inference_batch_size: 96
```

If self-play runs out of memory, reduce concurrent games first. Model inference
is small; most memory pressure generally comes from the batch and two arena
models residing on the GPU.

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
- Trees are batched across games rather than distributed across multiple Python
  processes.
- Arena trees are rebuilt every move because candidate and champion are
  different evaluators; reusing a subtree evaluated by the other model would
  contaminate the comparison.

These choices keep the implementation readable and auditable. Later speed work
can add multiprocessing actors, a transposition table, CUDA graphs, larger
networks, and asynchronous replay generation without changing the replay or
checkpoint concepts.
