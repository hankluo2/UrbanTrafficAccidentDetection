from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, auc

from .networks.unet.unet import UNet
from .utils.algorithm import psnr_error
from .config.config import args

from .dataset import TestDataset

from tqdm import tqdm

import cv2
import imageio


def evaluate(videos_dir, checkpoint, gt_file=None, show_heatmap=False):
    """
    It takes a directory of videos, a ground truth file, and a checkpoint file, and returns the AUC of
    the ROC curve

    :param videos_dir: the directory where the videos are stored
    :param gt_file: the file containing the ground truth labels
    :param checkpoint: the path to the pre-trained model
    :param show_heatmap: if True, the heatmap of the difference between the generated frame and the
    ground truth frame will be saved as a gif, defaults to False (optional)
    """

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

    psnrs = np.array(psnrs)
    norm_psnrs = (psnrs - np.min(psnrs)) / (np.max(psnrs) - np.min(psnrs))

    if gt_file is not None:
        gt = np.load(gt_file, allow_pickle=True)[args.time_steps:]
        fpr, tpr, thresh = roc_curve(gt, norm_psnrs)
        roc_auc = auc(fpr, tpr)

    # print(roc_auc)

    if show_heatmap:
        imageio.mimsave('results/heatmap-02_01.gif', heatmaps, fps=12.5)
        imageio.mimsave('results/original-02_01.gif', originals, fps=12.5)

    if gt_file is not None:
        return gt, norm_psnrs, roc_auc
    else:
        return norm_psnrs


def main():
    gt, pred, roc_auc = evaluate(args.video_dir, args.gt, args.checkpoint)  # demo
    gt = gt[:, np.newaxis]
    pred = pred[:, np.newaxis]
    result = np.concatenate((gt, pred), axis=1)
    np.save(f"results/predict/{Path(args.gt).name}", result)


if __name__ == "__main__":
    main()
