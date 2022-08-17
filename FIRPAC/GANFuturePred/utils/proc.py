from pathlib import Path
import shutil
import json
import numpy as np
import torch
import scipy.signal as signal

from ..eval import evaluate


# customized parsing function
def parse_via_json_annotations(file_name):
    """
    It opens the file, reads the data, and returns the start and end times
    The json file to parse is generated by VIA(VGG Image Annotator).

    :param file_name: the name of the file you want to parse
    :return: The start and end time of the video in seconds.
    """
    with open(file_name, 'r') as reader:
        data = json.load(reader)

    for key in data["metadata"]:  # only one key
        start_sec, end_sec = data[key]["z"]

    return start_sec, end_sec


def split_video2clips_by_peaks(video_root, peaks_root, save_dir, clip_frame_num):
    """
    For each video, we split it into clips, each clip is centered at a peak point
    
    :param video_root: the directory where the videos are stored
    :param peaks_root: the directory where the peak files are stored
    :param save_dir: the directory where you want to save the clips
    """
    video_root_path = Path(video_root)
    peaks_root_path = Path(peaks_root)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # for each pair of video and peaks file, the names between them
    # should be identical.
    common_names = [path.name for path in sorted(list(video_root_path.glob("*")))]
    for name in common_names:
        frames = sorted(list((video_root_path / name).glob("*")))
        # print(frames)
        peaks_data = np.load(peaks_root_path / (name + ".npy"))
        # print(peaks_data.shape)
        for i, zero_point in enumerate(peaks_data):
            # print(peak)
            center = zero_point + 5  # shift 5 frames
            if center - clip_frame_num < 0:
                center = clip_frame_num
            if center + clip_frame_num >= len(frames):
                center = len(frames) - clip_frame_num - 1

            clip = frames[center - clip_frame_num:center + clip_frame_num]
            # print(len(clip))
            # save each clip into a new directory (save as new videos)
            for img in clip:
                # print(img, Path(save_dir) / (name + '_' + str(i).zfill(2) + img.suffix))
                subdir = Path(save_dir) / (name + '_' + str(i).zfill(2))
                subdir.mkdir(parents=True, exist_ok=True)
                print(img, subdir / img.name)
                shutil.copy(img, subdir / img.name)


def gen_clip_level_gt(peaks_dir, frames_gt_dir, save_dir, clip_frame_num):
    """
    For each file in `peaks_dir`, load the corresponding file in `frames_gt_dir`, and for each peak in
    the file, check if there is a 1 in the corresponding 64 frames. If there is, append 1 to `clip_gt`,
    else append 0

    :param peaks_dir: directory where the peak files are stored
    :param frames_gt_dir: directory of ground truth frames
    :param save_dir: the directory where the generated clip-level ground truth will be saved
    """
    # file names under peaks_dir & frames_gt_dir are identical.
    # Both directories store *.npy files
    center_paths = sorted(list(Path(peaks_dir).glob("*.npy")))
    # file_names = [path.name for path in center_paths]
    frames_gt_dirpath = Path(frames_gt_dir)

    for centers in center_paths:
        file_name = centers.name
        centers = np.load(centers, allow_pickle=True)

        frame_gt = frames_gt_dirpath / file_name
        frame_gt = np.load(frame_gt, allow_pickle=True)

        clip_gt = []
        for center in centers:
            if center - clip_frame_num // 2 < 0:
                center = clip_frame_num // 2
            if center + clip_frame_num // 2 >= len(frame_gt):
                center = len(frame_gt) - clip_frame_num // 2 - 1

            if 1 in frame_gt[center - clip_frame_num // 2:center + clip_frame_num // 2]:
                clip_gt.append(1)
            else:
                clip_gt.append(0)

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        np.save(Path(save_dir) / file_name, clip_gt)


def filter_peak_centered_frames(src_frames_root,
                                clip_frame_num,
                                filtered_frames_save_dir,
                                psnrs_dir,
                                peaks_dir,
                                ffp_ckpt,
                                test_mode=False):
    """
    `filter_peak_centered_frames` takes in a directory of frames, and for each video, it finds the peak
    of the PSNR curve, and then saves the frames centered around the peak into a new directory
    
    :param src_frames_root: the directory where the frames of the original videos are stored
    :param clip_frame_num: the number of frames in each clip
    :param filtered_frames_save_dir: the directory where the filtered frames will be saved
    :param psnrs_dir: the directory to save the psnr values of each video
    :param peaks_dir: the directory to save the peak frames of each video
    :param ffp_ckpt: the path to the checkpoint file of the ffp model
    :param test_mode: if True, save the psnr and peak data for each video, defaults to False (optional)
    """
    src_frames_root = Path(src_frames_root)
    video_dirs = sorted(list(src_frames_root.glob("*")))  # frames dir of each norm video
    peak_centered_clip_save_dir = filtered_frames_save_dir
    for video in video_dirs:
        psnr = evaluate(video, ffp_ckpt)
        psnr = (psnr - np.min(psnr)) / (np.max(psnr) - np.min(psnr))
        if test_mode:
            Path(psnrs_dir).mkdir(parents=True, exist_ok=True)
            np.save(Path(psnrs_dir) / (video.stem + ".npy"), psnr)

        try:
            smooth_data = signal.savgol_filter(psnr, clip_frame_num - 1, 3)
            peak = signal.find_peaks(smooth_data, distance=clip_frame_num)[0]  # 1d-array
            if test_mode:
                Path(peaks_dir).mkdir(parents=True, exist_ok=True)
                np.save(Path(peaks_dir) / (video.stem + ".npy"), peak)
        except:
            continue

        frames = sorted(list(video.glob("*")))
        for i, zero_point in enumerate(peak):
            # print(peak)
            center = zero_point + 5  # shift 5 frames
            if center - clip_frame_num // 2 < 0:
                center = clip_frame_num // 2
            if center + clip_frame_num // 2 >= len(frames):
                center = len(frames) - clip_frame_num // 2 - 1

            clip = frames[center - clip_frame_num // 2:center + clip_frame_num // 2]
            # print(len(clip))
            # save each clip into a new directory (save as new videos)
            subdir = Path(peak_centered_clip_save_dir) / (video.name + '_' + str(i).zfill(2))
            subdir.mkdir(parents=True, exist_ok=True)
            for img in clip:
                # print(img, Path(save_dir) / (name + '_' + str(i).zfill(2) + img.suffix))
                print(img, subdir / img.name)
                shutil.copy(img, subdir / img.name)
