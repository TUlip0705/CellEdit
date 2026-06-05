#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from celledit.common.conditions import (
    cond_inputs_from_meta_index_bins,
    cond_inputs_from_meta_index_cont_film,
    dose_id_from_concentration_bins,
    ensure_empty_control_ids,
    load_kpgt_table,
)
from celledit.common.meta_ext import load_meta_ext
from celledit.common.model_loader import cond_vec_uncond_batch, load_latent_flow_bundle
from celledit.common.save_utils import SaveSpec, now_run_id, sanitize_filename, save_gray_channels, save_rgb_preview, save_triplet, write_json
from celledit.common.selectors import CounterfactualPair, pick_counterfactual_pair, pick_empty_for_target
from celledit.common.vae_io import (
    build_preprocessor_from_vae_config,
    load_raw_6ch_npy,
    load_vae,
    preprocess_raw_6ch,
    vae_decode,
    vae_encode,
)
from celledit.flowedit.flowedit import FlowEditConfig, flowedit_latent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("CellEdit counterfactual editing")

    # Model (latent_flow2)
    p.add_argument("--ckpt", type=str, required=True, help="latent_flow2 checkpoint (step_XXXX.pt)")
    p.add_argument("--train_config", type=str, required=True, help="latent_flow2 resolved config.yaml used for training")
    p.add_argument("--use_ema", action="store_true", help="Use EMA weights")

    # VAE + preprocessing
    p.add_argument("--vae_model", type=str, default="hybrid", choices=["hybrid", "native"])
    p.add_argument("--vae_ckpt", type=str, required=True)
    p.add_argument("--vae_config", type=str, required=True, help="adapter_vae run config.yaml (for preprocessing)")
    p.add_argument("--pretrained_vae", type=str, default=None, help="Hybrid VAE backbone path")
    p.add_argument("--vae_sample_latent", action="store_true", help="Sample z from posterior (default uses mean for determinism)")

    # Data caches
    p.add_argument("--meta_csv", type=str, default="latent_flow2/cache/meta/metadata.csv")
    p.add_argument(
        "--image_root",
        metavar="IMAGE_ROOT",
        type=str,
        default="data/rxrx3/images",
        help="Root directory for raw 6-channel image arrays",
    )
    p.add_argument("--kpgt_dir", type=str, default="latent_flow2/cache/kpgt")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Default target split")

    p.add_argument("--edits_yaml", type=str, default=None, help="YAML list of edit tasks")
    p.add_argument("--tgt_treatment", type=str, default=None)
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--tgt_concentration", type=float, default=None, help="Target concentration (µM) -> mapped to dose_id")
    g.add_argument("--tgt_dose_id", type=int, default=None, help="Target dose_id (0..7)")

    # FlowEdit hyperparams
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--n_max", type=int, default=120)
    p.add_argument("--n_min", type=int, default=5)
    p.add_argument("--n_avg", type=int, default=1)
    p.add_argument("--cfg_src", type=float, default=1.0, help="CFG scale for src (1.0=off; 0=uncond)")
    p.add_argument("--cfg_tgt", type=float, default=1.0, help="CFG scale for tgt (1.0=off; 0=uncond)")
    p.add_argument("--snapshot_count", type=int, default=0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num_edits", type=int, default=1, help="Number of repeats for single-task mode")

    p.add_argument("--out_dir", type=str, default=None, help="Output dir (default: CellEdit/outputs/<run_id>/...)")

    p.add_argument("--no_npy", action="store_true")
    p.add_argument("--no_source_npy", action="store_true", help="Do not save source_empty_6ch.npy")
    p.add_argument("--no_target_npy", action="store_true", help="Do not save target_real_6ch.npy")
    p.add_argument("--no_gray", action="store_true")
    p.add_argument("--no_rgb", action="store_true")

    p.add_argument(
        "--save_decoded_snapshots",
        action="store_true",
        help="Decode saved FlowEdit snapshots",
    )
    p.add_argument("--decoded_snapshots_dirname", type=str, default="snapshots_decoded")
    p.add_argument("--decoded_snapshots_save_npy", action="store_true", help="Also save decoded 6ch arrays per step (.npy).")
    p.add_argument("--decoded_snapshots_npy_dtype", type=str, default="float16", choices=["float16", "float32"])
    p.add_argument("--decoded_snapshots_gray", action="store_true", help="Save per-channel grayscale PNGs for decoded snapshots")
    p.add_argument("--decoded_snapshots_no_rgb", action="store_true", help="Do not save RGB PNGs for decoded snapshots")

    return p.parse_args()


def _load_edits_yaml(path: str | Path) -> list[dict[str, Any]]:
    obj = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "edits" in obj:
        obj = obj["edits"]
    if not isinstance(obj, list):
        raise ValueError(f"Invalid edits YAML root (expected list or {{edits: [...]}}): {path}")
    out = []
    for i, item in enumerate(obj):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid edits entry at index {i}: expected dict, got {type(item)}")
        out.append(item)
    return out


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    meta = load_meta_ext(args.meta_csv)
    kpgt_table, _, _ = load_kpgt_table(args.kpgt_dir)

    run_id = now_run_id("flowedit")
    out_dir = Path(args.out_dir) if args.out_dir else (Path("CellEdit") / "outputs" / run_id)

    preproc = build_preprocessor_from_vae_config(args.vae_config)
    vae = load_vae(
        vae_model=str(args.vae_model),
        vae_ckpt=args.vae_ckpt,
        device=device,
        pretrained_vae=args.pretrained_vae,
        vae_config_path=args.vae_config,
    )
    bundle = load_latent_flow_bundle(
        ckpt_path=args.ckpt,
        train_config=args.train_config,
        device=device,
        use_ema=bool(args.use_ema),
    )
    if bundle.dose_mode not in ("bins", "cont_film"):
        raise SystemExit(f"Unsupported dose_mode from train_config: {bundle.dose_mode}")

    if args.edits_yaml:
        tasks = _load_edits_yaml(args.edits_yaml)
    else:
        if not args.tgt_treatment:
            raise SystemExit("Missing --tgt_treatment (or provide --edits_yaml).")
        if args.tgt_concentration is None and args.tgt_dose_id is None:
            raise SystemExit("Missing target dose: provide --tgt_concentration or --tgt_dose_id (or --edits_yaml).")
        tasks = [
            {
                "tgt_treatment": str(args.tgt_treatment),
                "tgt_concentration": None if args.tgt_concentration is None else float(args.tgt_concentration),
                "tgt_dose_id": None if args.tgt_dose_id is None else int(args.tgt_dose_id),
                "split": str(args.split),
                "seed": int(args.seed),
                "num_edits": int(args.num_edits),
            }
        ]

    spec = SaveSpec(
        save_npy=not args.no_npy,
        save_source_npy=not args.no_source_npy,
        save_target_npy=not args.no_target_npy,
        save_edited_npy=True,
        save_gray=not args.no_gray,
        save_rgb=not args.no_rgb,
    )

    total_outputs = 0
    for t in tasks:
        n = int(t.get("num_edits", 1))
        total_outputs += max(1, int(n))

    pbar = None
    if total_outputs > 1:
        pbar = tqdm(total=int(total_outputs), desc="FlowEdit edits", unit="img", disable=None)

    out_tasks = []
    for task_idx, task in enumerate(tasks):
        treatment = str(task.get("tgt_treatment", "")).strip()
        if not treatment:
            raise ValueError(f"Task[{task_idx}] missing tgt_treatment")

        split = str(task.get("split", args.split))
        base_seed = int(task.get("seed", int(args.seed) + 10_000 * task_idx))
        num_edits = int(task.get("num_edits", 1))

        if task.get("tgt_dose_id", None) is not None:
            tgt_dose_id = int(task["tgt_dose_id"])
        elif task.get("tgt_concentration", None) is not None:
            tgt_dose_id = dose_id_from_concentration_bins(float(task["tgt_concentration"]), treatment)
        else:
            raise ValueError(f"Task[{task_idx}] needs tgt_dose_id or tgt_concentration")

        src_meta_index = task.get("src_meta_index", None)

        tgt_meta_index = task.get("tgt_meta_index", None)
        if src_meta_index is not None and tgt_meta_index is None:
            raise ValueError(f"Task[{task_idx}] src_meta_index requires tgt_meta_index (to form a valid pair)")

        if src_meta_index is not None and tgt_meta_index is not None:
            src_index = int(src_meta_index)
            tgt_index = int(tgt_meta_index)
            if str(meta.treatment[tgt_index]) != treatment or int(meta.dose_id[tgt_index]) != int(tgt_dose_id) or int(meta.empty_id[tgt_index]) != 0:
                raise ValueError(f"Task[{task_idx}] tgt_meta_index does not match (treatment,dose_id,empty_id=0)")
            allow_cross_plate_source = bool(task.get("allow_cross_plate_source", False))
            if not allow_cross_plate_source:
                if (
                    str(meta.experiment_name[src_index]) != str(meta.experiment_name[tgt_index])
                    or int(meta.plate[src_index]) != int(meta.plate[tgt_index])
                    or str(meta.cell_type[src_index]) != str(meta.cell_type[tgt_index])
                ):
                    raise ValueError(f"Task[{task_idx}] src/tgt meta_index are not on the same plate key (experiment, plate, cell_type)")
            pair = CounterfactualPair(src_index=int(src_index), tgt_index=int(tgt_index))

        elif tgt_meta_index is not None:
            tgt_index = int(tgt_meta_index)
            if str(meta.treatment[tgt_index]) != treatment or int(meta.dose_id[tgt_index]) != int(tgt_dose_id) or int(meta.empty_id[tgt_index]) != 0:
                raise ValueError(f"Task[{task_idx}] tgt_meta_index does not match (treatment,dose_id,empty_id=0)")
            src_index = pick_empty_for_target(
                meta,
                tgt_index=int(tgt_index),
                seed=int(base_seed) + 999_983,
                prefer_same_site=True,
                prefer_split=None,
                include_cell_type_in_plate_key=True,
            )
            pair = CounterfactualPair(src_index=int(src_index), tgt_index=int(tgt_index))
        else:
            pair = pick_counterfactual_pair(
                meta,
                treatment=treatment,
                dose_id=int(tgt_dose_id),
                split=split,
                seed=base_seed,
                prefer_same_site=True,
                prefer_empty_split=None,
                include_cell_type_in_plate_key=True,
            )

        ensure_empty_control_ids(meta, int(pair.src_index))

        src_row = meta.row_dict(int(pair.src_index))
        tgt_row = meta.row_dict(int(pair.tgt_index))

        raw_src = load_raw_6ch_npy(args.image_root, src_row["npy_path"])
        raw_tgt = load_raw_6ch_npy(args.image_root, tgt_row["npy_path"])
        x_src = preprocess_raw_6ch(raw_src, preproc, device=device)
        x_tgt_real = preprocess_raw_6ch(raw_tgt, preproc, device=device)

        with torch.no_grad():
            if bundle.dose_mode == "bins":
                cond_src_in = cond_inputs_from_meta_index_bins(meta, kpgt_table=kpgt_table, index=int(pair.src_index), device=device)
                cond_tgt_in = cond_inputs_from_meta_index_bins(meta, kpgt_table=kpgt_table, index=int(pair.tgt_index), device=device)
                cond_src_vec = bundle.cond(
                    kpgt_fp=cond_src_in.kpgt_fp,
                    dose_id=cond_src_in.dose_id,
                    empty_id=cond_src_in.empty_id,
                    apply_cfg_dropout=False,
                )
                cond_tgt_vec = bundle.cond(
                    kpgt_fp=cond_tgt_in.kpgt_fp,
                    dose_id=cond_tgt_in.dose_id,
                    empty_id=cond_tgt_in.empty_id,
                    apply_cfg_dropout=False,
                )
            else:  # cont_film
                cond_src_in = cond_inputs_from_meta_index_cont_film(meta, kpgt_table=kpgt_table, index=int(pair.src_index), device=device)
                cond_tgt_in = cond_inputs_from_meta_index_cont_film(meta, kpgt_table=kpgt_table, index=int(pair.tgt_index), device=device)
                cond_src_vec = bundle.cond(
                    kpgt_fp=cond_src_in.kpgt_fp,
                    dose_cont=cond_src_in.dose_cont,
                    empty_id=cond_src_in.empty_id,
                    apply_cfg_dropout=False,
                )
                cond_tgt_vec = bundle.cond(
                    kpgt_fp=cond_tgt_in.kpgt_fp,
                    dose_cont=cond_tgt_in.dose_cont,
                    empty_id=cond_tgt_in.empty_id,
                    apply_cfg_dropout=False,
                )
            uncond_vec = cond_vec_uncond_batch(bundle.cond, batch_size=1, device=device, dtype=cond_src_vec.dtype)

        task_dir_name = sanitize_filename(
            f"task{task_idx:03d}_{treatment}_dose{tgt_dose_id}_exp{tgt_row['experiment_name']}_plate{tgt_row['plate']}_site{tgt_row['site']}_tgt{tgt_row['sample_id']}_src{src_row['sample_id']}"
        )
        base_task_dir = out_dir / task_dir_name

        for r in range(num_edits):
            seed_edit = int(base_seed) + int(r)

            g_vae = torch.Generator(device=device)
            g_vae.manual_seed(int(seed_edit))
            enc = vae_encode(vae, x_src, sample_latent=bool(args.vae_sample_latent), generator=g_vae, amp=True)
            z_src = enc.sample

            steps_i = int(task.get("steps", args.steps))
            n_max_i = int(task.get("n_max", args.n_max))
            n_min_i = int(task.get("n_min", args.n_min))
            n_avg_i = int(task.get("n_avg", args.n_avg))
            if steps_i <= 0:
                raise ValueError(f"Invalid FlowEdit steps={steps_i} for task[{task_idx}]")
            if n_max_i < n_min_i:
                raise ValueError(f"Invalid FlowEdit window for task[{task_idx}]: require n_max>=n_min, got n_max={n_max_i} n_min={n_min_i}")

            z_edit, snaps = flowedit_latent(
                sit=bundle.sit,
                z_src=z_src,
                cond_src=cond_src_vec,
                cond_tgt=cond_tgt_vec,
                uncond=uncond_vec,
                cfg=FlowEditConfig(
                    steps=int(steps_i),
                    n_max=int(n_max_i),
                    n_min=int(n_min_i),
                    n_avg=int(n_avg_i),
                    cfg_src=float(task.get("cfg_src", args.cfg_src)),
                    cfg_tgt=float(task.get("cfg_tgt", args.cfg_tgt)),
                    snapshot_count=int(task.get("snapshot_count", args.snapshot_count)),
                    amp=True,
                ),
                seed=int(seed_edit),
            )

            x_edit = vae_decode(vae, z_edit, amp=True, clamp=True)

            out_one = base_task_dir / f"edit_{r:03d}_seed{seed_edit}"
            snapshots_decoded_dir = None
            decoded_snapshot_count = 0
            if args.save_decoded_snapshots:
                snapshots_decoded_dir = str(out_one / str(args.decoded_snapshots_dirname))
            meta_out: dict[str, Any] = {
                "method": "CellEdit",
                "run_id": run_id,
                "device": str(device),
                "seed_select": int(base_seed),
                "seed_edit": int(seed_edit),
                "dose_mode": str(bundle.dose_mode),
                "task": {
                    "tgt_treatment": treatment,
                    "tgt_dose_id": int(tgt_dose_id),
                    "tgt_concentration_input": None if task.get("tgt_concentration", None) is None else float(task.get("tgt_concentration")),
                    "split": split,
                    "task_index": int(task_idx),
                    "repeat_index": int(r),
                },
                "selection_protocol": {
                    "target_split": split,
                    "plate_empty_filter": True,
                    "prefer_same_site_empty": True,
                },
                "src_row": src_row,
                "tgt_row": tgt_row,
                "hyperparams": {
                    "steps": int(task.get("steps", args.steps)),
                    "n_max": int(task.get("n_max", args.n_max)),
                    "n_min": int(task.get("n_min", args.n_min)),
                    "n_avg": int(task.get("n_avg", args.n_avg)),
                    "cfg_src": float(task.get("cfg_src", args.cfg_src)),
                    "cfg_tgt": float(task.get("cfg_tgt", args.cfg_tgt)),
                    "snapshot_count": int(task.get("snapshot_count", args.snapshot_count)),
                },
                "models": {
                    "latent_flow_ckpt": str(args.ckpt),
                    "latent_flow_train_config": str(args.train_config),
                    "latent_flow_sit_state_used": bundle.sit_state_used,
                    "vae_model": str(args.vae_model),
                    "vae_ckpt": str(args.vae_ckpt),
                    "vae_config": str(args.vae_config),
                },
                "preprocessing": {
                    "note": "raw uint8/uint16 is scaled to [0,1], clipped by fixed stats q_low/q_high, then mapped to [-1,1].",
                    "preproc_repr": repr(preproc),
                },
            }

            save_triplet(out_dir=out_one, source=x_src, target_real=x_tgt_real, edited=x_edit, meta=meta_out, spec=spec)
            if snaps is not None and len(snaps) > 0:
                snap_dir = out_one / "snapshots_latent"
                snap_dir.mkdir(parents=True, exist_ok=True)
                for k, z in enumerate(snaps):
                    np.save(snap_dir / f"z_{k:03d}.npy", z.detach().cpu().float().numpy())

                if args.save_decoded_snapshots:
                    dec_dir = out_one / str(args.decoded_snapshots_dirname)
                    dec_dir.mkdir(parents=True, exist_ok=True)
                    dec_meta: dict[str, Any] = {
                        "note": "Each step is a decoded snapshot from FlowEdit latent trajectory. step_000 is the source latent.",
                        "snapshot_count_arg": int(task.get("snapshot_count", args.snapshot_count)),
                        "decoded_snapshots_dirname": str(args.decoded_snapshots_dirname),
                        "decoded_snapshots_save_npy": bool(args.decoded_snapshots_save_npy),
                        "decoded_snapshots_npy_dtype": str(args.decoded_snapshots_npy_dtype),
                        "decoded_snapshots_gray": bool(args.decoded_snapshots_gray),
                        "decoded_snapshots_rgb": not bool(args.decoded_snapshots_no_rgb),
                        "num_snapshots": int(len(snaps)),
                        "files": [],
                    }

                    for k, z in enumerate(snaps):
                        x_k = vae_decode(vae, z, amp=True, clamp=True)
                        row: dict[str, Any] = {"step": int(k)}

                        if not bool(args.decoded_snapshots_no_rgb):
                            rgb_path = dec_dir / f"step_{k:03d}_rgb.png"
                            save_rgb_preview(x_k, out_path=rgb_path)
                            row["rgb"] = str(rgb_path.relative_to(out_one))

                        if bool(args.decoded_snapshots_gray):
                            save_gray_channels(x_k, out_dir=dec_dir, prefix=f"step_{k:03d}", also_grid=True)
                            row["gray_dir"] = str(dec_dir.relative_to(out_one))

                        if bool(args.decoded_snapshots_save_npy):
                            x_save = x_k.detach().cpu()
                            if str(args.decoded_snapshots_npy_dtype) == "float32":
                                x_save = x_save.float()
                            else:
                                x_save = x_save.half()
                            npy_path = dec_dir / f"step_{k:03d}_6ch.npy"
                            np.save(npy_path, x_save.numpy())
                            row["npy"] = str(npy_path.relative_to(out_one))

                        dec_meta["files"].append(row)
                    decoded_snapshot_count = int(len(snaps))
                    write_json(dec_dir / "meta.json", dec_meta)

            if args.save_decoded_snapshots:
                meta_patch = {
                    "snapshots": {
                        "snapshot_count_arg": int(task.get("snapshot_count", args.snapshot_count)),
                        "latent_saved": bool(snaps is not None and len(snaps) > 0),
                        "decoded_saved": bool(args.save_decoded_snapshots),
                        "decoded_snapshots_dir": snapshots_decoded_dir,
                        "decoded_snapshot_count": int(decoded_snapshot_count),
                    }
                }
                meta_path = out_one / "meta.json"
                meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
                meta_obj.update(meta_patch)
                write_json(meta_path, meta_obj)

            out_tasks.append(str(out_one))
            if pbar is not None:
                pbar.set_postfix_str(f"{treatment} dose{tgt_dose_id} task{task_idx} r{r}", refresh=False)
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    print(json.dumps({"run_id": run_id, "out_dir": str(out_dir), "num_outputs": len(out_tasks)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
