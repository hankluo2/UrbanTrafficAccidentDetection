from pathlib import Path
import numpy as np
from vidtools import extract_frames


def extract_frames_from_dataset_source_videos(video_dir, dest_dir, fps=None):
    src_videos = sorted(list(Path(video_dir).glob("*.mp4")))
    for v in src_videos:
        dest_frm_dir = Path(dest_dir) / v.stem
        dest_frm_dir.mkdir(parents=True, exist_ok=True)

        extract_frames(str(v), str(dest_frm_dir), fps, fill_zeros=5)
