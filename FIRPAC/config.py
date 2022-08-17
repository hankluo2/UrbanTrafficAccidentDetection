import argparse


def evaluate_parse_args():
    parser = argparse.ArgumentParser(description="Parameters of evaluation.")
    parser.add_argument('--anomaly_feature_dir',
                        type=str,
                        default='datasets/suzhou/features16',
                        help="Directory to extracted i3d features of anomaly videos.")
    parser.add_argument('--clip_gt_dir',
                        type=str,
                        default='datasets/suzhou/peak_clip_gt',
                        help="Directory to annotated labels of test video sets.")
    parser.add_argument('--frame_gt_dir',
                        type=str,
                        default="datasets/suzhou/frames_gt",
                        help="feature-level groundtruth")
    parser.add_argument('--peaks_dir',
                        type=str,
                        default="pre_data/suzhou/syn/peaks",
                        help="peaks information storage directory")
    parser.add_argument('--anomaly_score_dir',
                        type=str,
                        default="pre_data/suzhou/syn/ascore",
                        help="Directory to save anomaly scores.")
    parser.add_argument('--frames_per_clip', type=int, default=16, help="number of frames per clip")
    parser.add_argument('--model_path',
                        type=str,
                        default="exps/models/2000.pt",
                        help="pretrained vad model path")

    return parser.parse_args()
