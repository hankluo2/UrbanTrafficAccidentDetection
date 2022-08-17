class args():

    # Model setting
    # checkpoint = 'weights/sample.pth'
    checkpoint = 'weights/ckpt_2000_0.02878069539864858_0.2501498025655746.pth'

    # Dataset setting
    channels = 3
    size = 256
    # videos_dir = 'datasets/Test015'
    video_dir = '/root/workspace/video_anomaly_det/FFP/datasets/test/suzhou/03_07'
    gt = "datasets/frames_gt/03_07.npy"
    time_steps = 5

    # For GPU training
    gpu = 0  # None

    video_root = '/root/workspace/video_anomaly_det/FFP/datasets/CTAD/train'
    pred_save_dir = 'results/raw-predictions/CTAD/psnr/train'
    peaks_save_dir = 'results/raw-predictions/CTAD/peaks/train'
