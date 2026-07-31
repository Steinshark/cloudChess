# Self-play optimization patch 0.3

## Hot-path changes

- `self_play.concurrent_games` now defaults to 96. The intended comparison set is 64, 96, and 128.
- Active games are replenished immediately as games finish, avoiding the long low-utilization tail of fixed batches.
- MCTS no longer checks every unchanged root for terminal status during every scheduler pass.
- PUCT's parent log/square-root term is cached by effective parent visit count and reused for every child score.
- Search uses one mutable board per tree. Every simulation applies a selected path and undoes it before returning; no board or repetition-counter copy is made per simulation.
- Previous-position planes use one temporary `undo()`/`apply()` pair and no board-history copy.
- Piece bitboards are converted 12 or 24 at a time with NumPy broadcasting.
- Each tree can reserve several leaves with virtual visits and virtual loss, allowing one inference batch to contain more states than the number of active games.
- The evaluator gathers only legal policy indexes on CUDA. Dense 4,672-value policies are not copied to the CPU.

## New measurements

`selfplay_summary.json` now includes nested `search` and `evaluator` records:

- simulations per second;
- neural leaves per second;
- mean and maximum requested inference batch;
- mean unique inference batch;
- duplicate-state fraction;
- selection, evaluation, and backup wall time;
- total legal logits transferred back to the CPU.

Run the dedicated comparison:

```powershell
python -m chess_selflearn.benchmark_search `
  --config selflearn_config.yaml `
  --checkpoint "S:/data/chess/bootstrap_runs/resnet12x128/best.pt" `
  --concurrency 64 96 128 `
  --simulations 64 `
  --plies 6 `
  --output "S:/data/chess/search_benchmark.json"
```

Choose the concurrency with the highest sustained `search.simulations_per_second`, not necessarily the highest GPU utilization.

## Validation commands

```powershell
python -m chess_selflearn.optimization_smoke_test
python -m chess_selflearn.smoke_test `
  --checkpoint "S:/data/chess/bootstrap_runs/resnet12x128/best.pt"
```

The first test requires `bulletchess` but not CUDA. The second performs a real checkpoint and CUDA inference test.

## Compatibility

Replay HDF5 and checkpoint formats are unchanged. Existing 12-block corrected-policy checkpoints remain loadable. Existing YAML files also remain loadable because all new configuration fields have defaults.
