"""EMDB detector swap, step 1: re-detect 2D keypoints with RTMPose-l (COCO-17).

Uses the same per-frame person boxes as the original pipeline (EMDB pkl
'bboxes', lurb = xyxy) so the detector is the only changed variable.
Output per sequence: <out>/<Px_seq>.npz with kp2d (F,17,3) full-image
[x, y, score], matching the emdb_vit_v4.pt kp2d convention.

Run: python tools/emdb_rtmpose_extract.py [--smoke]
"""
import argparse, pickle, time
from pathlib import Path
import numpy as np
import cv2

RAW = Path("inputs/EMDB/raw")
OUT = Path("inputs/EMDB/rtm_kp2d")


def seq_dirs():
    for pkl in sorted(RAW.glob("P*/*/*_data.pkl")):
        yield pkl.parent


def load_boxes(pkl_path):
    annot = pickle.load(open(pkl_path, "rb"))
    boxes = annot["bboxes"]["bboxes"].astype(np.float64)  # (F,4) lurb
    F = annot["n_frames"]
    assert len(boxes) == F, (pkl_path, len(boxes), F)
    # fill invalid rows with nearest valid box
    valid = np.isfinite(boxes).all(1) & ((boxes[:, 2] - boxes[:, 0]) > 1)
    if not valid.all():
        idx = np.where(valid)[0]
        near = idx[np.abs(idx[None, :] - np.arange(F)[:, None]).argmin(1)]
        boxes = boxes[near]
    return boxes, F, annot["name"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="first sequence only")
    args = ap.parse_args()
    from rtmlib import RTMPose
    pose = RTMPose(
        onnx_model="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
        "onnx_sdk/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.zip",
        model_input_size=(288, 384), backend="onnxruntime", device="cuda",
    )
    OUT.mkdir(exist_ok=True)
    for sd in seq_dirs():
        name = f"{sd.parent.name}_{sd.name}"  # e.g. P0_09_outdoor_walk
        out_fp = OUT / f"{name}.npz"
        if out_fp.exists():
            print(f"skip {name} (exists)", flush=True)
            continue
        pkl = next(sd.glob("*_data.pkl"))
        boxes, F, pkl_name = load_boxes(pkl)
        assert pkl_name == name, (pkl_name, name)
        imgs = sorted((sd / "images").glob("*.jpg"))
        assert len(imgs) == F, (name, len(imgs), F)
        kp = np.zeros((F, 17, 3), np.float32)
        t0 = time.time()
        for i, fp in enumerate(imgs):
            img = cv2.imread(str(fp))
            k, s = pose(img, bboxes=[boxes[i]])
            kp[i, :, :2] = k[0]
            kp[i, :, 2] = s[0]
        np.savez_compressed(out_fp, kp2d=kp)
        print(f"{name}: {F} frames in {time.time()-t0:.0f}s "
              f"({F/(time.time()-t0):.0f} fps), conf_mean={kp[:,:,2].mean():.3f}", flush=True)
        if args.smoke:
            break
    print("RTM_EXTRACT_DONE", time.strftime("%F %T"), flush=True)


if __name__ == "__main__":
    main()
