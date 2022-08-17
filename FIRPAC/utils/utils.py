from typing import List
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
import scipy.signal as signal


def get_segments_start_end_score(frame_wise_data: np.ndarray, is_pred=False, thresh=None):
    """
    It takes a list of labels and returns a list of start and end indices of segments
    
    :param frame_wise_data: the output of the model, which is a numpy array of shape (num_frames, 1)
    :type frame_wise_data: np.ndarray
    :param is_pred: whether the input is prediction or ground truth, defaults to False (optional)
    :param thresh: the threshold for the prediction
    """
    if is_pred:
        # generate prediction video segments
        assert thresh is not None
        # thresholding
        org_data = frame_wise_data
        frame_wise_data = org_data
        frame_wise_data[frame_wise_data >= thresh] = 1.
        frame_wise_data[frame_wise_data < thresh] = 0.

        scores = np.array([])

    labels, starts, ends = np.array([]), np.array([]), np.array([])

    last_index = 0
    last_label = frame_wise_data[last_index]
    labels = np.append(labels, frame_wise_data[last_index])
    starts = np.append(starts, last_index)

    for i in range(frame_wise_data.shape[0]):
        if last_label != frame_wise_data[i]:
            ends = np.append(ends, i)  # save last

            if is_pred:
                scores = np.append(scores, np.amax(org_data[last_index:i]))

            starts = np.append(starts, i)  # update current
            labels = np.append(labels, frame_wise_data[i])

            last_label = frame_wise_data[i]  # update
            last_index = i

    ends = np.append(ends, i + 1)  # update the last
    if is_pred:
        scores = np.append(scores, np.amax(org_data[last_index:i + 1]))
    else:
        scores = labels

    return starts, ends, labels, scores


def merge_data(data_dir, is_frame_gt=False):
    data_files = sorted(list(Path(data_dir).glob("*.npy")))  # for 1D arrays
    merged_data = np.array([])

    for data_file in data_files:
        if is_frame_gt:
            data = np.load(data_file, allow_pickle=True)[5:]
        else:
            data = np.load(data_file, allow_pickle=True)
        merged_data = np.append(merged_data, data)

    return merged_data


def compute_IoU(y_stat, p_stat):
    y_start, y_end, y_label, _ = y_stat
    p_start, p_end, p_label, _ = p_stat

    # tp = 0
    # fp = 0
    # hits = np.zeros(len(y_label))
    # overlap = 0.1

    for i in range(len(p_label)):
        # min_end_py = np.minimum(p_end[j], y_end)
        # max_start_py = np.maximum(p_start[j], y_start)
        # max_end_py = np.maximum(p_end[j], y_end)
        # min_start_py = np.minimum(p_start[j], y_start)
        intersection = np.minimum(p_end[i], y_end) - np.maximum(p_start[i], y_start)
        union = np.maximum(p_end[i], y_end) - np.minimum(p_start[i], y_start)
        IoU = (1.0 * intersection /
               union) * ([p_label[i] == y_label[x] for x in range(len(y_label))])
        print(IoU)


def get_segment_label(frame_wise_label: np.ndarray, pred_data):
    starts, ends, y_pred, y_scores = pred_data
    starts = starts.astype(np.int32)
    ends = ends.astype(np.int32)
    y_true = []
    for i in range(len(starts)):
        # i for segment index
        if frame_wise_label[starts[i]:ends[i]].__contains__(1.):
            y_true.append(1.)
        else:
            y_true.append(0.)

    y_true = np.array(y_true)
    return y_true, y_pred


def roc_metrics(gt, pred, average='weighted'):
    fpr, tpr, _ = roc_curve(gt, pred)
    rec_auc = auc(fpr, tpr)
    thresh = np.argmax(tpr - fpr) / len(tpr - fpr)
    _precision_score = precision_score(gt, pred > thresh, average=average)
    _recall = recall_score(gt, pred > thresh, average=average)
    f1 = f1_score(gt, pred > thresh, average=average)

    return rec_auc, _precision_score, _recall, f1, thresh


def get_mAP(gt_label: np.ndarray, sorted_segments):
    hits = 0
    ap = 0
    for i, (score, p_class, start, end) in enumerate(sorted_segments):
        start = start.astype(np.int32)
        end = end.astype(np.int32)
        if int(p_class) == 1 and gt_label[start:end].__contains__(1.):  # hit
            hits += 1
            ap += hits / (i + 1)
    # print(i)
    mAP = ap / hits
    return mAP


def get_mAP2(gt_label: np.ndarray, sorted_segments):
    hits = 0
    ap = 0
    for i, (score, start, end) in enumerate(sorted_segments):
        # start = start.astype(np.int32)
        # end = end.astype(np.int32)
        if gt_label[start:end].__contains__(1.):  # hit
            hits += 1
            ap += hits / (i + 1)
    # print(i)
    mAP = ap / hits if hits != 0 else 0
    return mAP


def mAP_on_dataset(frames_gt_dir, pred_score_dir, fps, time_err):
    gt_dirpath = Path(frames_gt_dir)
    acc_dirpath = Path(pred_score_dir)

    gt_paths = sorted(list(gt_dirpath.glob("*.npy")))
    vid_names = [path.name for path in gt_paths]
    mAP_sum = 0
    for vid_name in vid_names:
        gt_label = np.load(gt_dirpath / vid_name, allow_pickle=True)[5:]  # target result

        pred_data = np.load(acc_dirpath / vid_name, allow_pickle=True)
        # smoothing data
        smooth_data = signal.savgol_filter(pred_data, 31, 3)
        peaks = signal.find_peaks(smooth_data, distance=30)[0]  # 1d-array
        scores = [smooth_data[peak] for peak in peaks]
        # fps = 10
        shift = time_err * fps
        starts = [peak - shift if peak - shift > 0 else 0 for peak in peaks]
        ends = [peak + shift for peak in peaks]
        preds = list(zip(scores, starts, ends))
        preds.sort()
        preds.reverse()  # target result

        mAP_sum += get_mAP2(gt_label, preds)

        # print(vid_name, get_mAP2(gt_label, preds))

    return mAP_sum / len(gt_paths)


def report_roc(y_true, y_pred):
    metric_keys = ["rec_auc", "prec", "recall", "f1", "thresh"]
    _metrics = roc_metrics(y_true, y_pred, average='weighted')

    dic = dict(zip(metric_keys, _metrics))
    # print(dic)

    return dic


def evaluate_dataset(inference_dir, gt_dir, method: str):
    inf_samples = sorted(list(Path(inference_dir).glob("*.npy")))
    gt_samples = sorted(list(Path(gt_dir).glob("*.npy")))
    gt = np.load(gt_samples[0], allow_pickle=True)[5:]
    pred = np.load(inf_samples[0], allow_pickle=True)

    for i in range(1, len(inf_samples)):
        _gt = np.load(gt_samples[i], allow_pickle=True)[5:]
        _pred = np.load(inf_samples[i], allow_pickle=True)

        gt = np.concatenate((gt, _gt))
        pred = np.concatenate((pred, _pred))

    # _metrics = roc_metrics(gt, pred, average='weighted')  # global metrics
    dic = report_roc(gt, pred)
    np.save(gt_samples[0].parent.parent.name + "-gt.npy", gt)
    np.save(inf_samples[0].parent.parent.parent.name + f"-{method}-pred.npy", pred)
    # fpr, tpr, _ = roc_curve(gt, pred)
    # draw_roc_auc(fpr, tpr)
    return dic
