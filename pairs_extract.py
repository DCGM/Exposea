#!/usr/bin/env python3
"""
extract_quality_pairs.py
========================
Extracts "GOOD vs BAD" patch pairs from Exposea source fragments for
use in a perceptual quality user study.

THEORY
------
Each source image is mapped to the output mosaic via a 3x3 homography H.
At every output pixel, the local Jacobian determinant of the *inverse* mapping
measures how much the source was geometrically stretched:

  |det J| >> 1  ->  source was *compressed* -> high effective resolution <- WINNER
  |det J| << 1  ->  source was *stretched*  -> low effective resolution  <- LOSER

Patches are cropped from the original source images warped into the output
coordinate space:
  * GOOD.png = The winner fragment (highest resolution)
  * BAD.png  = The loser fragment (lowest resolution)

INPUT METADATA FORMAT (--meta PKL)
----------------------------------
A pickle file containing a dictionary where keys are image filenames and values
are the 3x3 homography numpy arrays mapping source -> output.
Example: {"frag_01.jpg": array([[...]]), "frag_02.jpg": array([[...]])}

USAGE
-----
  python extract_quality_pairs.py \
      --stitch     result/stitch.png \
      --meta       cache/homogs/H_expname_timestamp.pkl \
      --n_pairs    30 \
      --patch_size 256
"""

import argparse
import json
import os
import pickle
import random
import sys
from os import makedirs

import cv2 as cv
from pathlib import Path
from omegaconf import OmegaConf

import cv2
import numpy as np
from trimesh.repair import stitch


def jacobian_det_map(H_src_to_out, out_h, out_w):
    Hinv = np.linalg.inv(H_src_to_out)
    ys, xs = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    ones = np.ones_like(xs)
    pts = np.stack([xs, ys, ones], axis=-1)

    def project(H, p):
        q = p @ H.T
        return q[..., :2] / q[..., 2:3]

    dx = np.zeros_like(xs);
    dx[:, 1:-1] = 0.5
    dy = np.zeros_like(ys);
    dy[1:-1, :] = 0.5
    z = np.zeros_like(xs)

    src_xp = project(Hinv, pts + np.stack([dx, z, z], -1))
    src_xm = project(Hinv, pts + np.stack([-dx, z, z], -1))
    src_yp = project(Hinv, pts + np.stack([z, dy, z], -1))
    src_ym = project(Hinv, pts + np.stack([z, -dy, z], -1))

    dsx_dx = (src_xp[..., 0] - src_xm[..., 0]) / (2 * dx + 1e-9)
    dsy_dx = (src_xp[..., 1] - src_xm[..., 1]) / (2 * dx + 1e-9)
    dsx_dy = (src_yp[..., 0] - src_ym[..., 0]) / (2 * dy + 1e-9)
    dsy_dy = (src_yp[..., 1] - src_ym[..., 1]) / (2 * dy + 1e-9)

    det = np.abs(dsx_dx * dsy_dy - dsy_dx * dsx_dy).astype(np.float32)

    det[:, 0] = det[:, 1]
    det[:, -1] = det[:, -2]
    det[0, :] = det[1, :]
    det[-1, :] = det[-2, :]
    return det


def coverage_mask(H_src_to_out, src_h, src_w, out_h, out_w):
    corners = np.array(
        [[0, 0], [src_w - 1, 0], [src_w - 1, src_h - 1], [0, src_h - 1]],
        dtype=np.float32).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners, H_src_to_out).reshape(-1, 2)
    mask = np.zeros((out_h, out_w), np.uint8)
    cv2.fillConvexPoly(mask, np.round(proj).astype(np.int32), 255)
    return mask.astype(bool)


def warp_to_output(src_img, H, out_h, out_w):
    return cv2.warpPerspective(
        src_img, H, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0)


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Input directory same as for exposea")
    ap.add_argument("--meta", required=True, help="Pickle file with homographies dict: {filename: H_matrix}")
    ap.add_argument("--n_pairs", type=int, default=500)
    ap.add_argument("--patch_size", type=int, default=512)
    ap.add_argument("--output_dir", default="pairs")
    ap.add_argument("--min_overlap", type=float, default=0.8)
    ap.add_argument("--max_det_winner", type=float, default=1.5)
    ap.add_argument("--min_det_ratio", type=float, default=0.91)
    ap.add_argument("--spread", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()

    config = OmegaConf.load(os.path.join(args.input, "config.yaml"))

    random.seed(args.seed)
    np.random.seed(args.seed)
    stitch_dir = os.path.join(args.input, "images")
    os.makedirs(args.output_dir, exist_ok=True)

    out_h, out_w = config.final_res
    print(f"[INFO] Canvas size: {out_w} x {out_h} px")

    # Load metadata from pickle
    with open(args.meta, "rb") as f:
        norm_homog = pickle.load(f)

    # Convert dict into the list format expected by the rest of the script
    sources = []
    for filename, H_matrix in norm_homog.items():
        sources.append({
            "image": filename,
            "H": H_matrix
        })
    n_src = len(sources)
    print(f"[INFO] Sources loaded from pickle: {n_src}")

    # Per-source: load image, mask, det-J map
    Hs, src_imgs, masks, jmaps = [], [], [], []

    for i, entry in enumerate(sources):
        img_path = entry["image"]
        # If the filename in the pickle is just the basename, append the stitch directory
        if not os.path.isabs(img_path):
            img_path = os.path.join(stitch_dir, img_path)

        img = cv2.imread(img_path)
        if img is None:
            sys.exit(f"[ERROR] Cannot read source image: {img_path}")

        H = np.array(entry["H"], dtype=np.float64)
        sh, sw = img.shape[:2]

        mask = coverage_mask(H, sh, sw, out_h, out_w)
        jmap = jacobian_det_map(H, out_h, out_w)
        jmap[~mask] = np.inf  # outside coverage: never wins

        Hs.append(H)
        src_imgs.append(img)
        masks.append(mask)
        jmaps.append(jmap)
        print(f"  src[{i}] {os.path.basename(img_path)}  ({sw} x {sh})")

    # Winner / coverage maps
    jstack = np.stack(jmaps, axis=0)  # (N, out_h, out_w)



    n_cover = np.sum(np.isfinite(jstack), axis=0)  # number of sources at each px
    # winner = np.argmin(jstack, axis=0)  # best-resolution source per px
    winner = np.abs(np.log(np.abs(jstack) + 1e-8)).argmin(axis=0)
    winner_idx = np.abs(np.log(np.abs(jstack) + 1e-8)).argmin(axis=0)
    cv.imwrite(f"pairs/best_frag.jpg", winner_idx.astype(np.uint8) * 20)

    P = args.patch_size
    half = P // 2
    min_gap = int(P * args.spread)

    # Enumerate candidate patches
    print("[INFO] Scanning for candidate patches ...")
    step = max(P // 4, 32)
    candidates = []

    for cy in range(half, out_h - half, step):
        for cx in range(half, out_w - half, step):
            y0, y1 = cy - half, cy + half
            x0, x1 = cx - half, cx + half

            if np.mean(n_cover[y0:y1, x0:x1] >= 2) < args.min_overlap:
                continue

            w_patch = winner[y0:y1, x0:x1]
            counts = np.bincount(w_patch.ravel(), minlength=n_src)
            w_idx = int(np.argmax(counts))
            winner_mask = w_patch == w_idx
            if np.mean(winner_mask) < args.min_overlap:
                continue

            w_vals = jstack[w_idx, y0:y1, x0:x1][winner_mask]
            w_detj = float(np.mean(w_vals))
            # if w_detj > args.max_det_winner:
            #     continue

            best_l_idx = None
            best_l_detj = -1.0
            for li in range(n_src):
                if li == w_idx:
                    continue
                li_finite = np.isfinite(jstack[li, y0:y1, x0:x1])
                co_cover = np.mean(li_finite & winner_mask)
                if co_cover < args.min_overlap:
                    continue
                avg_j = float(np.mean(jstack[li, y0:y1, x0:x1][li_finite & winner_mask]))
                if avg_j > best_l_detj:
                    best_l_detj = avg_j
                    best_l_idx = li

            if best_l_idx is None:
                continue
            ratio = best_l_detj / (w_detj + 1e-9)
            if ratio > args.min_det_ratio:
                continue

            candidates.append(dict(
                cy=cy, cx=cx,
                winner=w_idx, loser=best_l_idx,
                winner_detj=round(w_detj, 4),
                loser_detj=round(best_l_detj, 4),
                det_ratio=round(ratio, 3),
            ))

    if not candidates:
        print("[ERROR] No valid pairs found.")
        return

    candidates.sort(key=lambda c: -c["det_ratio"])

    # Spatially spread selected patches
    chosen, used_pos = [], []
    for c in candidates:
        if any(abs(c["cx"] - px) < min_gap and abs(c["cy"] - py) < min_gap for px, py in used_pos):
            continue
        chosen.append(c)
        used_pos.append((c["cx"], c["cy"]))
        if len(chosen) >= args.n_pairs: break

    if len(chosen) < args.n_pairs:
        shuffled = candidates[:]
        random.shuffle(shuffled)
        for c in shuffled:
            if c not in chosen: chosen.append(c)
            if len(chosen) >= args.n_pairs: break

    print(f"[INFO] Extracting {len(chosen)} pairs ...")

    warp_cache = {}

    def get_warped(idx):
        if idx not in warp_cache:
            warp_cache[idx] = warp_to_output(src_imgs[idx], Hs[idx], out_h, out_w)
        return warp_cache[idx]

    manifest = []
    saved = 0
    makedirs(os.path.join(args.output_dir, 'comparison'), exist_ok=True)

    for c in chosen:
        y0 = c["cy"] - half;
        y1 = c["cy"] + half
        x0 = c["cx"] - half;
        x1 = c["cx"] + half

        good_patch = get_warped(c["winner"])[y0:y1, x0:x1].copy()
        bad_patch = get_warped(c["loser"])[y0:y1, x0:x1].copy()

        # Skip if either patch is mostly black/empty space
        if np.mean(bad_patch == 0) > 0.15 or np.mean(good_patch == 0) > 0.15:
            continue

        stem = f"pair_{saved:03d}"
        cv2.imwrite(os.path.join(args.output_dir, f"{stem}_GOOD.png"), good_patch)
        cv2.imwrite(os.path.join(args.output_dir, f"{stem}_BAD.png"), bad_patch)
        cv2.imwrite(os.path.join(args.output_dir, 'comparison', f"{stem}_COMPARE_{c["det_ratio"]}.png"),
                    np.concatenate([good_patch, bad_patch], axis=1))

        manifest.append(dict(
            pair_id=saved,
            output_x=x0, output_y=y0,
            patch_size=P,
            winner_src=sources[c["winner"]]["image"],
            loser_src=sources[c["loser"]]["image"],
            winner_detj=c["winner_detj"],
            loser_detj=c["loser_detj"],
            det_ratio=c["det_ratio"],
            good_file=f"{stem}_GOOD.png",
            bad_file=f"{stem}_BAD.png",
        ))

        print(f"  #{saved:03d}  "
              f"winner={sources[c['winner']]['image']} detJ={c['winner_detj']:.3f}  "
              f"loser={sources[c['loser']]['image']}  detJ={c['loser_detj']:.3f}  "
              f"ratio={c['det_ratio']:.2f}")
        saved += 1
        if saved >= args.n_pairs: break

    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[DONE] Saved {saved} pairs to '{args.output_dir}/'")


if __name__ == "__main__":
    main()