from pathlib import Path
import numpy as np

from .eval import evaluate
from .config import config


def load_test():
    gt_dir = 'datasets/suzhou/frames_gt'
    test_dir = 'datasets/suzhou/test'

    gts = sorted(list(Path(gt_dir).glob("*.npy")))  # valid ground truth files
    tests = sorted(list(Path(test_dir).glob("*_*")))  # valid test videos

    tests = []
    gt_name = [path.stem for path in gts]

    for exist_name in gt_name:
        test_path = Path(test_dir) / exist_name
        gt_path = Path(gt_dir) / (exist_name + '.npy')

        tests.append({"gt": gt_path, "test": test_path})

    # print(tests)
    return tests


def pred():
    tests = load_test()
    model_ckpt = config.args.checkpoint
    for sample in tests:
        gt_path = str(sample['gt'])
        video_dir = str(sample['test'])
        gt, pred, roc_auc = evaluate(video_dir, gt_path, model_ckpt)
        gt = gt[:, np.newaxis]
        pred = pred[:, np.newaxis]
        result = np.concatenate((gt, pred), axis=1)
        # NOTE: np.save below is optional. However pred should be returned for end2end sake.
        np.save(f"results/predict/suzhou-syn/{Path(gt_path).name}", result)

    # return result  # for reference


if __name__ == "__main__":
    pred()
