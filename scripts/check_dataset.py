"""
Keep & synchronize ONLY successful trajectories across:
  - the shard JSON (episodes metadata)
  - the shard .h5 (traj_<episode_id> groups)
  - the videos (numbered by success-order: video 0 = 1st success, etc.)

Produces a new synchronized set with consistent 0..K-1 numbering:
  <out>/trajectory.h5      (traj_0 .. traj_{K-1})
  <out>/trajectory.json    (episodes 0 .. K-1)
  <out>/videos/<i>.mp4      renamed to match
"""

import os
import json
import shutil
import glob
import h5py

# ---- CONFIG: point these at ONE shard's matching trio ----
SHARD_DIR   = "demos/PlaceSphere-v1/motionplanning"
JSON_PATH   = os.path.join(SHARD_DIR, "20260724_193812.json")   # a shard json
H5_PATH     = os.path.join(SHARD_DIR, "20260724_193812.h5")     # its shard h5
VIDEO_DIR   = SHARD_DIR                    # where the .mp4s live
OUT_DIR     = "demos/PlaceSphere-v1/successful"

# How videos are numbered. From your findings: "success_order".
# If instead your videos are named by raw episode_id, set to "episode_id".
VIDEO_NUMBERING = "success_order"


def find_video(video_dir, number):
    """Find a video file whose name ends in the given integer, robustly."""
    # match things like '..._0.mp4', '0.mp4', 'video_0.mp4', etc.
    candidates = []
    for pat in (f"*_{number}.mp4", f"*[!0-9]{number}.mp4", f"{number}.mp4"):
        candidates += glob.glob(os.path.join(video_dir, pat))
    # de-dup and prefer an exact-number stem match
    candidates = sorted(set(candidates))
    for c in candidates:
        stem = os.path.splitext(os.path.basename(c))[0]
        # last integer token in the stem
        tail = "".join(ch if ch.isdigit() else " " for ch in stem).split()
        if tail and int(tail[-1]) == number:
            return c
    return candidates[0] if candidates else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_video_dir = os.path.join(OUT_DIR, "videos")
    os.makedirs(out_video_dir, exist_ok=True)

    # 1) Load JSON metadata
    with open(JSON_PATH) as fh:
        meta = json.load(fh)
    episodes = meta["episodes"]

    # 2) Successful episodes, in original order (this order == video success-order)
    successful = [e for e in episodes if e.get("success", False)]
    print(f"Found {len(successful)} successful / {len(episodes)} total episodes")

    # 3) Open source h5 and create the filtered output h5
    with h5py.File(H5_PATH, "r") as src, \
         h5py.File(os.path.join(OUT_DIR, "trajectory.h5"), "w") as dst:

        new_episodes = []
        for new_idx, ep in enumerate(successful):
            ep_id = ep["episode_id"]

            # --- h5: copy traj_<ep_id> -> traj_<new_idx> ---
            src_key = f"traj_{ep_id}"
            if src_key not in src:
                print(f"  WARNING: {src_key} missing in h5, skipping ep {ep_id}")
                continue
            src.copy(src_key, dst, name=f"traj_{new_idx}")

            # --- video: map to source video number, then rename/copy ---
            if VIDEO_NUMBERING == "success_order":
                src_video_num = new_idx          # video already numbered by success order
            else:  # "episode_id"
                src_video_num = ep_id

            vpath = find_video(VIDEO_DIR, src_video_num)
            if vpath is not None:
                dst_vpath = os.path.join(out_video_dir, f"{new_idx}.mp4")
                shutil.copy2(vpath, dst_vpath)
            else:
                print(f"  WARNING: no video found for number {src_video_num} (ep {ep_id})")

            # --- json: keep the record, renumbered ---
            rec = dict(ep)
            rec["original_episode_id"] = ep_id
            rec["episode_id"] = new_idx
            new_episodes.append(rec)

    # 4) Write the synchronized JSON
    out_meta = dict(meta)
    out_meta["episodes"] = new_episodes
    out_meta.setdefault("filtering", {})
    out_meta["filtering"]["kept"] = "successful_only"
    out_meta["filtering"]["source_json"] = os.path.basename(JSON_PATH)
    with open(os.path.join(OUT_DIR, "trajectory.json"), "w") as fh:
        json.dump(out_meta, fh, indent=4)

    print(f"\nDone. Synchronized successful set written to: {OUT_DIR}")
    print(f"  trajectory.h5   -> traj_0 .. traj_{len(new_episodes)-1}")
    print(f"  trajectory.json -> {len(new_episodes)} episodes")
    print(f"  videos/         -> 0.mp4 .. {len(new_episodes)-1}.mp4")


if __name__ == "__main__":
    main()
