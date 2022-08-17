from typing import List
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, auc
import scipy.signal as signal

from .dataset import TestDataset
from .networks.unet.unet import UNet
from .utils.algorithm import psnr_error
from .config.config import args
from .utils.algorithm import select_topK_maximum_windows

from tqdm import tqdm

import cv2
import imageio


def load_predict(video_root: str) -> List[str]:
    """
    It takes a path to a directory containing subdirectories of videos, and returns a list of paths to
    those subdirectories

    :param video_root: The root directory of the videos
    :type video_root: str
    :return: A list of strings, where each string is the path to a video directory.
    """
    root = Path(video_root)
    video_dir_paths = sorted(list(root.glob("*")))
    return [str(path) for path in video_dir_paths]


def predict_and_save(videos_dir, checkpoint, save_dir, show_heatmap=False):

    generator = UNet(in_channels=args.channels * (args.time_steps - 1), out_channels=args.channels)
    generator.load_state_dict(torch.load(checkpoint)['generator'])
    print('The pre-trained generator has been loaded from ', checkpoint)

    testloader = DataLoader(dataset=TestDataset(channels=args.channels,
                                                size=args.size,
                                                videos_dir=videos_dir,
                                                time_steps=args.time_steps),
                            batch_size=1,
                            shuffle=False,
                            num_workers=0)

    if torch.cuda.is_available() and args.gpu is not None:
        use_cuda = True
        torch.cuda.set_device(args.gpu)

        generator = generator.cuda()
    else:
        use_cuda = False
        print('using CPU, this will be slow')

    if show_heatmap:
        heatmaps = []
        originals = []

    psnrs = []
    with torch.no_grad():
        for i, datas in enumerate(tqdm(testloader)):
            frames, o_frames = datas[0], datas[1]
            generator.eval()
            inputs = frames[:, :args.channels * (args.time_steps - 1), :, :]
            target = frames[:, args.channels * (args.time_steps - 1):, :, :]

            if use_cuda:
                inputs = inputs.cuda()

            generated = generator(inputs)

            if use_cuda:
                generated = generated.cpu()

            # compute real psnr
            psnr_err = psnr_error(generated=generated, ground_truth=target)
            psnrs.append(psnr_err.detach().numpy())

            # ==================== visualization of psnr ==================
            if show_heatmap:
                diffmap = torch.sum(torch.abs(generated - target).squeeze(), 0)
                diffmap -= diffmap.min()
                diffmap /= diffmap.max()
                diffmap *= 255
                diffmap = diffmap.detach().numpy().astype('uint8')

                heatmap = cv2.applyColorMap(diffmap, cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                heatmaps.append(heatmap)

                original = o_frames[-1].squeeze().detach().numpy().astype('uint8')
                original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                originals.append(original)
            # =============== end of visualization of psnr ================

    if show_heatmap:
        imageio.mimsave('results/heatmap-02_01.gif', heatmaps, fps=12.5)
        imageio.mimsave('results/original-02_01.gif', originals, fps=12.5)

    # save psnrs to npy file.
    save_path = Path(save_dir) / (Path(videos_dir).name + ".npy")
    np.save(str(save_path), psnrs)


def pred_and_save_psnrs(pred_save_dir, checkpoint, video_root):
    """
    This function loads the data to be predicted, and then predicts the psnr error score computed by FFP
    model, and saves them into npy files

    :param pred_save_dir: the directory where the predicted PSNRs will be saved
    :param checkpoint: the path to the checkpoint file
    :param video_root: the directory where the videos are stored
    """
    # Path(args.pred_save_dir).mkdir(parents=True, exist_ok=True)
    Path(pred_save_dir).mkdir(parents=True, exist_ok=True)

    predict = False
    # predict psnr error score computed by FFP model, and save them into npy files.
    if predict:
        # data2pred = load_predict(args.video_root)
        data2pred = load_predict(video_root)
        for _dir in data2pred:  # for all videos
            predict_and_save(_dir, checkpoint, pred_save_dir)


def save_peaks(psnr_dir, save_dir):
    """
    It takes in a directory of PSNR values, and saves the peaks of each PSNR value to a directory
    
    :param psnr_dir: the directory where the psnr data is saved
    :param save_dir: the directory to save the peaks
    """
    # select top-k windows of interest, then save them to a directory.
    # pred_data_files = sorted(list(Path(args.pred_save_dir).glob("*.npy")))
    pred_data_files = sorted(list(Path(psnr_dir).glob("*.npy")))
    # K = 10
    # window_size = 50
    # save_dir = args.peaks_save_dir
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    for file in pred_data_files:
        data = np.load(file, allow_pickle=True)
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        # woi = select_topK_maximum_windows(data, window_size, K)
        smooth_data = signal.savgol_filter(data, 53, 3)
        peaks = signal.find_peaks(smooth_data, distance=30)[0]  # 1d-array

        np.save(save_dir_path / file.name, peaks)

        print(file, peaks.shape, '\n')


def save_anomaly_scores(psnr_dir, ascore_dir, accident_score_dir):
    psnr_dir = Path(psnr_dir)
    psnr_paths = sorted(list(psnr_dir.glob('*.npy')))
    file_names = [path.name for path in psnr_paths]
    ascore_dir = Path(ascore_dir)

    Path(accident_score_dir).mkdir(parents=True, exist_ok=True)

    for vid_name in file_names:
        psnr = np.load(psnr_dir / vid_name, allow_pickle=True)
        ascore = np.load(ascore_dir / vid_name, allow_pickle=True)[5:]
        # print(psnr.shape, ascore.shape)
        # deprecated: use assertion instead
        if psnr.shape != ascore.shape:
            print(str(ascore_dir / vid_name))
            print(
                f"expected psnr.shape == ascore.shape, but got psnr.shape = {psnr.shape}, ascore.shape = {ascore.shape}"
            )

            # if psnr.shape[0] < ascore.shape[0]:
            patch_0 = np.array([0.] * 5)
            ascore = ascore[ascore.shape[0] - psnr.shape[0]:]
            ascore = np.concatenate((patch_0, ascore), axis=0)
            np.save(ascore_dir / vid_name, ascore)
            ascore = ascore[5:]

        # ## debug
        # psnr = psnr + 0.001  # epsilon
        # # For distinct dataset, use different f(ascore, psnr) can boost perf.
        # # thrsh = 0.95  # empirical, 0.95 is the best, for UTD
        # thrsh = 0.515  # empirical, 0.95 is the best, for TAD
        # weight = np.log10(1 + thrsh)
        # data = []
        # for i in range(psnr.shape[0]):
        #     if psnr[i] > thrsh:
        #         data.append(ascore[i] + weight * psnr[i] + math.log(psnr[i]))  # TAD
        #         # data.append(ascore[i] + weight * psnr[i])
        #     if psnr[i] < thrsh:
        #         data.append(ascore[i] - weight * psnr[i] + math.log(psnr[i]))  # TAD
        #         # data.append(ascore[i] - weight * psnr[i])
        # data = np.array(data)
        # ##

        data = ascore
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

        np.save(Path(accident_score_dir) / vid_name, data)
