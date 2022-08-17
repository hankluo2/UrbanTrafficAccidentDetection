import streamlit as st
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
from scipy.signal import savgol_filter, find_peaks

from .utils import *


def visualize_prediction(record_dir, video_frames_dir, show_org=False):
    """
    It loads the data, plots the data, and prints out the evaluation metrics

    :param record_dir: The directory where the prediction results are stored
    and it contains both prediction and ground-truths.
    """
    # parse test dataset
    samples = sorted(list(Path(record_dir).glob("*.npy")))

    sample_num = len(samples)
    index = st.slider("Sample No.#:", 0, sample_num - 1)
    selected_data = str(samples[index])

    st.write("Selected", selected_data)

    selected_vid_name = samples[index].stem  # [xx].npy
    selected_vid_dirpath = Path(video_frames_dir) / selected_vid_name
    frames = sorted(list(selected_vid_dirpath.glob("*")))[5:]
    frames_index = st.slider("Frame No.#:", 0, len(frames) - 1)
    st.write(f"Showing {str(frames[frames_index])}")

    img = Image.open(str(frames[frames_index]))
    st.image(img, caption=str(frames[frames_index]))

    data = np.load(selected_data, allow_pickle=True)
    gt = data[:, 0]
    pred = data[:, 1]
    chart_data = pd.DataFrame(data, columns=['ground-truth', 'prediction'])

    st.line_chart(chart_data)

    # predict local
    fpr, tpr, _ = roc_curve(gt, pred)
    rec_auc = auc(fpr, tpr)
    gauc, gprecision, grecall, gf1, thresh = evaluate_dataset(record_dir)  # global
    _precision_score = precision_score(gt, pred > thresh, average='weighted')
    _recall = recall_score(gt, pred > thresh, average='weighted')
    f1 = f1_score(gt, pred > thresh, average='weighted')

    # report video
    st.write("Video predictions:")
    st.write(f"AUC = {rec_auc:.2f}")
    st.write(f"f1_score = {f1:.2f}, precision = {_precision_score:.2f}, recall = {_recall:.2f}")

    # report test dataset
    st.write("\n")
    st.write("Global evaluation:")
    st.write(f"AUC = {gauc:.2f}")
    st.write(f"f1_score = {gf1:.2f}, precision = {gprecision:.2f}, recall = {grecall:.2f}")
    st.write(f"threshold = {thresh:.2f}")

    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    st.button("Re-run")


def vis_psnr_curve(video_root_dir, psnr_root_dir, accident_score_dir, gt_dir, show_org=False):
    video_dirs = sorted(list(Path(video_root_dir).glob("*")))
    psnr_paths = sorted(list(Path(psnr_root_dir).glob("*.npy")))
    accident_score_paths = sorted(list(Path(accident_score_dir).glob("*.npy")))
    gt_paths = sorted(list(Path(gt_dir).glob("*.npy")))

    video_index = st.slider("Video No.:", 0, len(video_dirs))  # select a video
    st.write(f"Current video is {str(video_dirs[video_index])}")
    # psnr scores
    psnr_file = psnr_paths[video_index]  # predict npy file
    psnr_data = np.load(psnr_file, allow_pickle=True)

    # accident scores
    acc_score_file = accident_score_paths[video_index]
    acc_score_data = np.load(acc_score_file, allow_pickle=True)
    # smooth_acc_score = savgol_filter(acc_score_data, 63, 3)
    smooth_acc_score = savgol_filter(acc_score_data, 63, 6)
    # print(acc_score_data)

    # ground-truth: labels
    gt_file = gt_paths[video_index]
    gt_data = np.load(gt_file, allow_pickle=True)[5:]
    # print(gt_data.shape)

    frames = list(sorted(video_dirs[video_index].glob("*")))[5:]  # video frames, shift 5
    frame_index = st.slider("Frame No.:", 0, len(frames) - 1)

    img = Image.open(str(frames[frame_index]))
    st.image(img, caption=str(frames[frame_index]))

    psnr_data = (psnr_data - np.min(psnr_data)) / (np.max(psnr_data) - np.min(psnr_data))
    # smooth_psnr = savgol_filter(psnr_data, 63, 3)
    smooth_psnr = savgol_filter(psnr_data, 63, 6)

    # compute mAP
    # ======================================================
    time_err = 3
    fps = 12

    peaks = signal.find_peaks(smooth_psnr, distance=30)[0]  # 1d-array
    scores = [smooth_psnr[peak] for peak in peaks]
    # fps = 10
    shift = time_err * fps
    starts = [peak - shift if peak - shift > 0 else 0 for peak in peaks]
    ends = [peak + shift for peak in peaks]
    preds = list(zip(scores, starts, ends))
    preds.sort()
    preds.reverse()  # target result
    st.write("PSNR score mAP: {:.4f}".format(get_mAP2(gt_data, preds)))

    peaks = signal.find_peaks(smooth_acc_score, distance=30)[0]  # 1d-array
    scores = [smooth_acc_score[peak] for peak in peaks]
    # fps = 10
    shift = time_err * fps
    starts = [peak - shift if peak - shift > 0 else 0 for peak in peaks]
    ends = [peak + shift for peak in peaks]
    preds = list(zip(scores, starts, ends))
    preds.sort()
    preds.reverse()  # target result
    st.write("Accident score mAP: {:.4f}".format(get_mAP2(gt_data, preds)))
    # ======================================================

    psnr_data = psnr_data[:, np.newaxis]
    psnr_smooth_data = smooth_psnr[:, np.newaxis]
    acc_score_data = acc_score_data[:, np.newaxis]
    acc_smooth_data = smooth_acc_score[:, np.newaxis]

    gt_data = gt_data[:, np.newaxis]

    if show_org:
        data = np.concatenate(
            (psnr_data, psnr_smooth_data, acc_score_data, acc_smooth_data, gt_data), axis=1)
        data = pd.DataFrame(data,
                            columns=[
                                "psnr score", "smoothed psnr", "accident score",
                                "smoothed accident score", "ground-truth"
                            ])
    else:
        data = np.concatenate((psnr_smooth_data, acc_smooth_data, gt_data), axis=1)
        data = pd.DataFrame(data, columns=["psnr", "accident score", "ground-truth"])

    st.line_chart(data)
    demo_data = np.concatenate((psnr_data), axis=0)
    demo_data = pd.DataFrame(demo_data, columns=['psnr'])
    st.line_chart(demo_data)

    # Add statistics below: Global
    report = evaluate_dataset(accident_score_dir, gt_dir, "acc")
    st.write("Global evaluation on accident score:")
    st.write(report)

    report = evaluate_dataset(psnr_root_dir, gt_dir, "psnr")
    st.write("Global evaluation on PSNR:")

    st.write(report)
