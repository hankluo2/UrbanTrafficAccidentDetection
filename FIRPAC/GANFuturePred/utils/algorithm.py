from pathlib import Path
from queue import PriorityQueue as PQ
from typing import List, Tuple
import numpy as np
import torch

from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score

# from config import args


def select_topK_maximum_windows(data: np.ndarray, window_size: int, k: int) -> List[Tuple]:
    """
    We first calculate the sum of each window, and then we use a priority queue to store the top K
    windows

    :param data: the data to be processed
    :type data: np.ndarray
    :param window_size: the size of the window
    :type window_size: int
    :param k: the number of windows you want to select
    :type k: int
    :return: A list of tuples, where each tuple is a window of size window_size, and the score of that
    window.
    """

    pq = PQ()
    data_sum = [0] * data.shape[0]
    for i in range(0, data.shape[0]):
        if i == 0:
            data_sum[i] = data[i]
        else:
            data_sum[i] = data_sum[i - 1] + data[i]

    for i in range(len(data_sum) - 1, window_size, -1):
        window_sum = data_sum[i] - data_sum[i - window_size]
        pq.put((-window_sum, i - window_size))

    topKwindows = []
    last_index = -10000
    count = 0
    while True:
        if pq.empty() or count == k:
            break
        score, index = pq.get()
        score *= -1
        if index - last_index > window_size * 0.8:  # Non-overlapping
            last_index = index
        topKwindows.append([index, index + window_size, score / window_size])
        count += 1

    return topKwindows


def psnr_error(generated: torch.Tensor, ground_truth: torch.Tensor):
    """
    It computes the PSNR error between two images

    :param generated: The generated image shaped of [batch, 3, height, width]
    :type generated: torch.Tensor
    :param ground_truth: The ground truth image shaped of [batch, 3, height, width]
    :type ground_truth: torch.Tensor
    """
    shape = generated.shape
    # num_pixels = torch.float(shape[1] * shape[2] * shape[3])
    num_pixels = shape[1] * shape[2] * shape[3]
    # print(generated < 0)  # exist values of some pixels less then 0.
    generated = (generated + 1.0) / 2.0
    ground_truth = (ground_truth + 1.0) / 2.0
    square_diff = torch.square(ground_truth - generated)

    batch_errors = 10 * torch.log10(1. / (1. / num_pixels) * torch.sum(square_diff, [1, 2, 3]))

    return torch.mean(batch_errors)


def metrics(gt, pred, average='weighted'):
    fpr, tpr, _ = roc_curve(gt, pred)
    rec_auc = auc(fpr, tpr)
    thresh = np.argmax(tpr - fpr) / len(tpr - fpr)
    _precision_score = precision_score(gt, pred > thresh, average=average)
    _recall = recall_score(gt, pred > thresh, average=average)
    f1 = f1_score(gt, pred > thresh, average=average)

    return rec_auc, _precision_score, _recall, f1, thresh
