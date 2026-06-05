# CellEdit

CellEdit edits Cell Painting images at the sample level. The model learns a
conditional latent flow over VAE latents and applies FlowEdit-style velocity
differences to transform a control image toward a target compound-dose
condition while preserving the spatial context of the source image.

## Layout

```text
CellEdit/
  train.py                 # training entry point
  infer.py                 # counterfactual editing entry point
  configs/
    rxrx3.yaml
    rxrx19b.yaml
  celledit/                # FlowEdit editing utilities
  external/
    latent_flow2/          # latent flow model
    adapter_vae/           # 6-channel VAE model/preprocessing
```

## Installation

```bash
pip install -r requirements.txt
```

Set the Python path from the repository root:

```bash
export PYTHONPATH="$PWD/CellEdit:$PWD/CellEdit/external:${PYTHONPATH:-}"
```

## Data

The configs assume the following local layout.

```text
data/
  rxrx3/
    meta/metadata.csv
    images/
    oph_patchwise/
    kpgt/
    vae_latents/
  rxrx19b/
    meta/metadata.csv
    images/
    oph_patchwise/
    kpgt/
    vae_latents/
```

## Train

Single GPU:

```bash
python CellEdit/train.py --mode irepa --config CellEdit/configs/rxrx3.yaml
```

Multi-GPU:

```bash
torchrun --nproc_per_node=4 CellEdit/train.py \
  --mode irepa \
  --config CellEdit/configs/rxrx3.yaml
```

For RxRx19b, replace the config:

```bash
torchrun --nproc_per_node=4 CellEdit/train.py \
  --mode irepa \
  --config CellEdit/configs/rxrx19b.yaml
```

## Inference

```bash
python CellEdit/infer.py \
  --ckpt outputs/rxrx3/checkpoints/step_0200000.pt \
  --train_config outputs/rxrx3/config.yaml \
  --use_ema \
  --vae_model hybrid \
  --vae_ckpt checkpoints/vae/best_final.pt \
  --vae_config checkpoints/vae/config.yaml \
  --pretrained_vae checkpoints/sd-vae-ft-mse \
  --meta_csv data/rxrx3/meta/metadata.csv \
  --image_root data/rxrx3/images \
  --kpgt_dir data/rxrx3/kpgt \
  --tgt_treatment Flavopiridol \
  --tgt_concentration 2.5 \
  --split test \
  --steps 150 \
  --n_max 120 \
  --n_min 5 \
  --n_avg 1 \
  --cfg_src 4 \
  --cfg_tgt 6 \
  --out_dir outputs/edits/flavopiridol
```

Run help for all options:

```bash
python CellEdit/infer.py --help
```
