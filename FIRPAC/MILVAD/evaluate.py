import argparse
import random
from tqdm import tqdm
from pathlib import Path
import numpy as np
import sys

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from .utils.loader import FeaturesLoaderVal
from .networks.TorchUtils import TorchModel
from .networks.AnomalyDetectorModel import AnomalyDetector, custom_objective, RegularizedLoss
from .utils.callbacks import DefaultModelCallback, TensorBoardCallback
from .utils.utils import register_logger, get_torch_device


def update_pedict(buffer_data: np.ndarray, pred_clip_score: np.ndarray, center: int,
                  frames_per_clip: int) -> np.ndarray:
    half_len = pred_clip_score.shape[0] // 2 * frames_per_clip
    if center - half_len < 0:  # here 32 = pred_clip_score.shape[0] / 2 * frames_per_clip
        start = 0
    elif center + half_len >= buffer_data.shape[0]:
        start = buffer_data.shape[0] - 2 * half_len - 1
    else:
        start = center - half_len

    cur = start
    for pred in pred_clip_score:
        # print(cur, pred)
        # buffer_data[cur:cur + frames_per_clip] = np.array([pred]).repeat(frames_per_clip)
        buffer_data[cur:cur + frames_per_clip] = pred

        # for idx in range(cur, cur + frames_per_clip):
        #     buffer_data[idx] = pred
        # print(buffer_data)
        cur += frames_per_clip

    return buffer_data


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def predict_clips(args, use_rand=None):
    # sys.path.insert(0, "/root/workspace/video_anomaly_det/FFP")  # Temporary solution
    # args = parse_args()
    # for key in list(vars(args).keys()):
    #     print("%s: %s" % (key, vars(args)[key]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = FeaturesLoaderVal(args.anomaly_feature_dir, args.clip_gt_dir)

    data_iter = DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # 4, # change this part accordingly
        pin_memory=True)

    # model = TorchModel.load_model(args.model_path).to(device).eval()
    # model_dict = torch.load(args.model_path)
    if use_rand is not None:
        setup_seed(use_rand)  # randomly select a random seed.

    network = AnomalyDetector(2048 * 10)
    model = TorchModel(network)
    # params = model.state_dict()
    # print(params)

    # whether to load the model
    if use_rand is None:
        model = model.load_model(args.model_path, network, is_pred=True)
        print(f"Model loaded from {args.model_path}")

    # params = model.state_dict()
    # print(params)
    model = model.to(device).eval()

    cudnn.benchmark = True

    y_trues = np.array([])
    y_preds = np.array([])
    last_peak_file = None
    frame_gt_dir = Path(args.frames_gt_dir)

    # placeholder of caches
    peaks_cache = None
    frame_gt_cache = None
    video_pred_cache = None
    Path(args.anomaly_score_dir).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for features, label, peaks_name, peaks_index in tqdm(data_iter):
            # print(features.shape, label.shape, peaks_name, peaks_index)
            features = features.to(device)
            outputs = model(features).squeeze(-1)
            # print(outputs, label, peaks_name, peaks_index)
            # print(outputs.shape, label.shape, peaks_name, peaks_index)

            peaks_name = peaks_name[0]  # the peaks numpy file name
            # print(peaks_name)
            # pre_data/suzhou/syn/peaks/[01_01.npy]
            peaks_index = peaks_index.cpu().numpy()[0]  # index of each peak in a certain peak file
            # print(peaks_name, peaks_index)
            peaks_path = Path(args.peaks_dir) / peaks_name
            frame_gt_path = frame_gt_dir / peaks_name
            # print(last_peak_file, peaks_name)

            if last_peak_file != peaks_name:
                # initialize a new prediction buffer of a whole video
                # else load the cache file directly

                # save last
                if video_pred_cache is not None:
                    np.save(Path(args.anomaly_score_dir) / last_peak_file, video_pred_cache)

                # load next
                peaks_cache = np.load(peaks_path,
                                      allow_pickle=True)
                # print(peaks_cache)
                frame_gt_cache = np.load(frame_gt_path, allow_pickle=True)
                # print(frame_gt_cache)
                video_pred_cache = np.array([0.] * frame_gt_cache.shape[0])
                # print(video_pred_cache)
                last_peak_file = peaks_name  # update
                # print(last_peak_file)

            center = peaks_cache[peaks_index]

            clip_num = outputs.shape[1]
            # label duplicated
            clip_gt = label.cpu().numpy().repeat(clip_num)
            # print(clip_num)
            scores = outputs[0].cpu().numpy()
            # print(scores)
            video_pred_cache = update_pedict(video_pred_cache, scores, center, args.clip_frame_num)
            # print(video_pred_cache)
            # print(len(video_pred_cache))
            # video_pred_cache = update_pedict(video_pred_cache, scores, center, args.feature_clip_len)  # bad perf.

            y_trues = np.concatenate((y_trues, clip_gt))
            y_preds = np.concatenate((y_preds, scores))

        # print(video_pred_cache.shape)
        np.save(Path(args.anomaly_score_dir) / last_peak_file, video_pred_cache)

    # print(y_trues.shape, y_preds.shape)
    return y_trues, y_preds
