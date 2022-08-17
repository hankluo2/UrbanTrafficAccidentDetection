"""
The general framework pipeline.
"""
from pathlib import Path
import hydra
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st

from GANFuturePred.signal_processing import save_anomaly_scores, save_peaks
from GANFuturePred.utils.proc import gen_clip_level_gt, split_video2clips_by_peaks
from MILVAD.evaluate import predict_clips
from MILVAD.train import train
from utils import *


def draw_roc_auc(dataset: str):
    gt = np.load(f"{dataset}-gt.npy", allow_pickle=True)
    acc_pred = np.load(f"{dataset}-acc-pred.npy", allow_pickle=True)
    psnr_pred = np.load(f"{dataset}-psnr-pred.npy", allow_pickle=True)
    fig = plt.figure()
    if dataset == 'suzhou':
        dataset = 'UTD'
    plt.title(f'ROC-{dataset}')

    fpr, tpr, _ = roc_curve(gt, acc_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'g', label='AUC_AccPred = %0.4f' % roc_auc)

    fpr, tpr, _ = roc_curve(gt, psnr_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label='AUC_psnr = %0.4f' % roc_auc)

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')

    st.pyplot(fig)
    plt.close()


def parse_args_dataset(args, manual=None):
    if manual is not None:
        args.dataset = manual

    for key in args:
        if isinstance(args[key], str) and 'DATASET' in args[key]:
            args[key] = args[key].replace('DATASET', args.dataset)
    for key in args.train:
        if isinstance(args.train[key], str) and 'DATASET' in args.train[key]:
            args.train[key] = args.train[key].replace('DATASET', args.dataset)

    return args


def eval_vis(args):
    # uncomment below for testing
    # Test: evaluation of mil vad
    # =============================================================================

    y_true_file = f"results/{args.dataset}_gt.npy"
    y_pred_file = f"results/{args.dataset}_pred.npy"
    if not Path(y_true_file).exists():
        y_true, y_pred = predict_clips(args)  # get from milvad
        np.save(y_true_file, y_true)
        np.save(y_pred_file, y_pred)
    else:
        # y_true, y_pred = predict_clips(args, use_rand=1087)  # get from milvad

        y_true, y_pred = predict_clips(args)  # get from milvad
        np.save(y_true_file, y_true)
        np.save(y_pred_file, y_pred)

        y_true = np.load(y_true_file, allow_pickle=True)
        y_pred = np.load(y_pred_file, allow_pickle=True)

    # generate final predictions
    # =============================================================================
    save_anomaly_scores(args.psnrs_dir, args.ascore_dir, args.accident_score_dir)
    #
    # =============================================================================

    # perform visualization  # NOTE: Use streamlit run main.py
    # =============================================================================
    vis_psnr_curve(args.video_root_dir, args.psnrs_dir, args.accident_score_dir, args.frames_gt_dir)
    draw_roc_auc(args.dataset)
    #
    # =============================================================================

    # Dataset-wise evaluation on mAP
    # mAP ======================================================================
    psnr_mAP = mAP_on_dataset(args.frames_gt_dir, args.psnrs_dir, 10, 2)
    refined_mAP = mAP_on_dataset(args.frames_gt_dir, args.accident_score_dir, 10, 2)
    st.write("mAP on psnr = {:.4f}, on accident score = {:.4f}".format(psnr_mAP, refined_mAP))
    #
    # ==========================================================================

    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    st.button("Re-run")


@hydra.main(config_path='config', config_name='mainCfg')
def main(args):
    # select_dataset_name = st.radio("Select a dataset:", args.datasets)
    # args = parse_args_dataset(args, manual=select_dataset_name)

    args = parse_args_dataset(args)

    # only for test/val mode below
    gen_clip_level_gt(args.peaks_dir, args.frames_gt_dir, args.clip_gt_dir, args.clip_frame_num)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        eval_vis(args)


if __name__ == "__main__":
    # Module 1.
    # Train FFP model with traffic videos first [use pretrained here]

    # Module 2.
    # Selective searching: see `extract_weak_supervised_dataset.py`

    # Module 3. MIL training/testing
    main()
