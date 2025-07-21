from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from pathlib import Path

import lmdb
import numpy as np
import pyarrow
import soundfile as sf

from mfcc import MFCC

style2onehot = {
'Happy':[1, 0, 0, 0, 0, 0],
'Sad':[0, 1, 0, 0, 0, 0],
'Neutral':[0, 0, 1, 0, 0, 0],
'Old':[0, 0, 0, 1, 0, 0],
'Angry':[0, 0, 0, 0, 1, 0],
'Relaxed':[0, 0, 0, 0, 0, 1],
}

# ---------------------------
# Helpers
# ---------------------------

def flatten_pose(pose: np.ndarray) -> np.ndarray:
    """(T, 9, 7) → (T, 63)"""
    if pose.ndim != 3 or pose.shape[1:] != (9, 7):
        raise ValueError(f"Expected pose shape (T,9,7); got {pose.shape}")
    return pose.reshape(pose.shape[0], -1).astype(np.float32)


def extract_mfcc(wav: np.ndarray, sr: int) -> np.ndarray:
    """Return 13‑D MFCCs using the same settings as the ZEGGS pipeline (20 ms hop)."""
    mfcc_extractor = MFCC(frate=20)  # 20 ms hop ⇒ 50 fps
    # mfcc_extractor.sig2s2mfc_energy expects (wav, None) second arg is dither seeds
    mfcc = mfcc_extractor.sig2s2mfc_energy(wav, None)  # (F, 15)
    mfcc = mfcc[:, :-2]  # keep first 13 coeffs
    return mfcc.astype(np.float32)


# ---------------------------
# Main builder
# ---------------------------

def build_dataset(
    npy_dir: Path,
    wav_dir: Path,
    out_root: Path,
    split_ratio: float = 0.9,
    target_sr: int = 16_000,
    map_size_mb: int = 512,
):
    """Scan *npy_dir* and *wav_dir*, write LMDB & stats under *out_root*."""
    print(f"npy_dir {npy_dir}, wav_dir {wav_dir}, out_root {out_root}")
    npy_dir = npy_dir.expanduser().resolve()
    wav_dir = wav_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(wav_dir.glob("*.wav"))
    clip_pairs: list[tuple[Path, Path]] = []
    for wav_path in wav_files:
        npy_path = npy_dir / f"{wav_path.stem}.npy"
        if npy_path.exists():
            clip_pairs.append((wav_path, npy_path))
        else:
            print(f"[WARN] Missing pose for {wav_path.name}; skipped.")

    if not clip_pairs:
        print("[ERR] No matching wav / npy pairs found. Abort.")
        sys.exit(1)

    # ───────────────────────────────────────────────────────── Split train / valid
    random.shuffle(clip_pairs)
    split_idx = int(len(clip_pairs) * split_ratio)
    splits = {
        "train": clip_pairs[:split_idx],
        "valid": clip_pairs[split_idx:],
    }

    # ───────────────────────────────────────────────────────── Prepare LMDB envs
    map_size = map_size_mb << 20  # MB → bytes
    envs = {
        split: lmdb.open(str(out_root / f"{split}_lmdb"), map_size=map_size)
        for split in splits
    }

    # Clean existing contents
    for env in envs.values():
        with env.begin(write=True) as txn:
            txn.drop(env.open_db())

    # For mean/std computation
    all_poses: list[np.ndarray] = []

    # ───────────────────────────────────────────────────────── Iterate clips
    for split, pairs in splits.items():
        env = envs[split]
        for idx, (wav_path, npy_path) in enumerate(pairs):
            clip_id = wav_path.stem
            print(f"[{split}] {clip_id}")

            name = os.path.split(npy_path)[1][:-4]
            if name.split('_')[1] in style2onehot:
                style = style2onehot[name.split('_')[1]]
            else:
                continue


            # --- Load pose
            pose_raw = np.load(npy_path) # shape (7799, 9, 7)
            poses = flatten_pose(pose_raw)  # (T, 63) > (7799, 63)
            all_poses.append(poses)

            # --- Load audio & MFCC
            wav, sr = sf.read(wav_path)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)  # mono
            if sr != target_sr:
                raise ValueError(f"Sample‑rate mismatch for {wav_path}: {sr} ≠ {target_sr}")
            mfcc_raw = extract_mfcc(wav, sr)  # (F, 13)

            # --- Build record
            record = {
                "vid": clip_id, # '001_Neutral_0_x_1_0'
                "clips": [{
                    "poses": poses, # (7286, 63)
                    "audio_raw": wav.astype(np.float32),
                    "mfcc_raw": mfcc_raw, # (2429, 13)
                    #"style_raw": np.zeros(6, dtype=np.float32),  # placeholder
                    "style_raw":  np.array(style)  # placeholder
                }],
            }

            # --- Write to LMDB
            with env.begin(write=True) as txn:
                k = f"{idx:010d}".encode("ascii")
                v = pyarrow.serialize(record).to_buffer()
                txn.put(k, v)

    # ───────────────────────────────────────────────────────── Save stats
    all_poses_stacked = np.vstack(all_poses)
    pose_mean = all_poses_stacked.mean(axis=0, dtype=np.float64)
    pose_std = all_poses_stacked.std(axis=0, dtype=np.float64)

    np.savez_compressed(out_root / "mean.npz", mean=pose_mean)
    np.savez_compressed(out_root / "std.npz", std=pose_std)

    # Close envs
    for env in envs.values():
        env.sync()
        env.close()

    print("\nDone ✔")
    print(f"Saved LMDBs & stats under {out_root}")


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Convert wav+npy dataset to LMDB for DiffuseStyleGesture")
    ap.add_argument("--npy_dir", default="/home/challenge-user/challenge-audio-to-gesture/datasets/npy", type=Path, help="Folder with *.npy pose files")
    ap.add_argument("--wav_dir", default="/home/challenge-user/challenge-audio-to-gesture/datasets/wav", type=Path, help="Folder with matching *.wav files")
    ap.add_argument("--out_lmdb", default="/home/challenge-user/challenge-audio-to-gesture/DiffuseStyleGesture/sarper_lmdb/processed_lmdb", type=Path, help="Output folder (will be created)")
    ap.add_argument("--split", type=float, default=0.9, help="Train split ratio (default 0.9)")
    ap.add_argument("--map_size_mb", type=int, default=512, help="LMDB map size in MB (default 512)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("done")
    build_dataset(
        npy_dir=args.npy_dir,
        wav_dir=args.wav_dir,
        out_root=args.out_lmdb,
        split_ratio=args.split,
        map_size_mb=args.map_size_mb,
    )
