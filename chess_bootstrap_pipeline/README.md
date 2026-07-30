# Chess bootstrap pipeline

This trains the supervised policy/value model from the sharded HDF5 dataset.

## Install

Install a CUDA-enabled PyTorch build first, then:

```powershell
pip install -r requirements.txt
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Audit

```powershell
python -m chess_bootstrap.inspect_dataset --dataset-root "S:/data/chess/dataset"
```

## Train

Edit `config.yaml`, then:

```powershell
python -m chess_bootstrap.train --config config.yaml
```

TensorBoard:

```powershell
tensorboard --logdir "S:/data/chess/bootstrap_runs/resnet8x128/tensorboard"
```

## Resume

```powershell
python -m chess_bootstrap.train --config config.yaml --resume "S:/data/chess/bootstrap_runs/resnet8x128"
```

## Test split

```powershell
python -m chess_bootstrap.evaluate `
  --config config.yaml `
  --checkpoint "S:/data/chess/bootstrap_runs/resnet8x128/best.pt" `
  --split test `
  --batches 1000
```

## 4060 Ti starting settings

Start at batch size 512 with FP16 and channels-last tensors. If CUDA runs out
of memory, reduce to 384 or 256. If GPU utilization is low, increase
`num_workers` from 2 to 4 and then try `read_chunk_size: 16384`.

The training set is streamed by contiguous HDF5 chunks, shuffled in RAM, and
split by a deterministic hash of `game_id`, so positions from one game cannot
cross between train, validation, and test.

The current bootstrap stage does not apply legal-move masks because the stored
records do not include FENs or legal masks. The later inference/MCTS layer will
mask illegal actions before softmax.
