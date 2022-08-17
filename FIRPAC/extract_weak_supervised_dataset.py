from tempfile import TemporaryDirectory
import shutil
from pathlib import Path
import i3dextractor
import numpy as np
import scipy.signal as signal

# from GANFuturePred import evaluate
from GANFuturePred import filter_peak_centered_frames
import tools

# options
CLIP_FRAME_NUM = 32

ffp_ckpt = "GANFuturePred/weights/ckpt_2000_0.02878069539864858_0.2501498025655746.pth"
i3d_ckpt = "GANFuturePred/pretrained/i3d_r50_nl_kinetics.pth"


def get_synthetic_dataset_features(dataset_name, weather_name):
    modes = ['test', 'train']  # opt
    keymap = {'train': 'normal', 'test': 'anomaly'}
    for mode in modes:
        # vid_dir = f"datasets/{dataset_name}/added_videos/{weather_name}_{keymap[mode]}"
        # video directory to normal and anomaly
        # vid_dir = f"datasets/{dataset_name}/video/{keymap[mode]}"
        vid_dir = f"datasets/{dataset_name}/segment_videos"

        psnrs_dir = f"pre_data/{dataset_name}/{mode}/psnrs"
        peaks_dir = f"pre_data/{dataset_name}/{mode}/peaks"
        feature_dir = f"datasets/{dataset_name}/features/train/{keymap[mode]}"

        with TemporaryDirectory() as tmpdir:
            dest_dir = f"{tmpdir}/{mode}"  # temp dir
            split_save_dir = f"{tmpdir}/fine-grained/{keymap[mode]}"  # temp dir

            Path(dest_dir).mkdir(parents=True, exist_ok=True)

            fps = 20  # CADP only
            # extract frames from videos
            tools.extract_frames_from_dataset_source_videos(vid_dir, dest_dir, fps)

            # predict psnr errors with trained FFP model
            filter_peak_centered_frames(
                dest_dir,
                CLIP_FRAME_NUM,
                split_save_dir,  # filtered frames segments
                psnrs_dir,
                peaks_dir,
                ffp_ckpt,
                test_mode=True)

            # extract i3d features of the fine-grained clips
            i3dextractor.extract_i3d_feature(split_save_dir,
                                             CLIP_FRAME_NUM,
                                             feature_dir,
                                             i3d_ckpt,
                                             mode="frames")  # extract in frames format

        break


def get_real_dataset_features(dataset_name):
    # modes = ['test', 'train']
    modes = ['test']
    keymap = {'train': 'normal', 'test': 'anomaly'}
    for mode in modes:
        # src_root = Path(f"datasets/{dataset_name}/train/{keymap[mode]}")
        src_root = Path(f"datasets/{dataset_name}/test")

        # psnrs_dir = f"pre_data/{dataset_name}/{mode}/psnrs"
        psnrs_dir = f"pre_data/{dataset_name}/test/{keymap[mode]}/psnrs"
        Path(psnrs_dir).mkdir(parents=True, exist_ok=True)
        # peaks_dir = f"pre_data/{dataset_name}/{mode}/peaks"
        peaks_dir = f"pre_data/{dataset_name}/test/{keymap[mode]}/peaks"
        Path(peaks_dir).mkdir(parents=True, exist_ok=True)

        # feature_dir = f"datasets/{dataset_name}/features/test/{keymap[mode]}"
        feature_dir = f"datasets/{dataset_name}/features/test"
        with TemporaryDirectory() as tmpdir:
            split_save_dir = Path(tmpdir) / f"fine-grained/{keymap[mode]}"
            split_save_dir.mkdir(parents=True, exist_ok=True)

            # predict psnr errors with trained FFP model
            filter_peak_centered_frames(
                str(src_root),
                CLIP_FRAME_NUM,
                split_save_dir,  # filtered frames segments
                psnrs_dir,
                peaks_dir,
                ffp_ckpt,
                test_mode=True)

            # extract i3d features of the fine-grained clips
            i3dextractor.extract_i3d_feature(split_save_dir,
                                             CLIP_FRAME_NUM,
                                             feature_dir,
                                             i3d_ckpt,
                                             mode="frames")  # extract in frames format


if __name__ == "__main__":
    # get_synthetic_dataset_features("CTADv", 'rain')
    get_synthetic_dataset_features("CADP", 'rain')
    # real_datasets = ['TAD']
    # real_datasets = ['suzhou']
    # for rd in real_datasets:
    #     get_real_dataset_features(rd)
