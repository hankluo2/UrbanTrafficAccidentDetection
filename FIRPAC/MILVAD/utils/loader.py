import logging
from pathlib import Path
from typing import List, Tuple
import os
import random
import numpy as np
import torch
from torch.utils import data
# from feature_extractor import read_features


class FeaturesLoader:
    def __init__(self,
                 normal_dir,
                 anomaly_dir,
                 bucket_size=64,
                 iterations=20000,
                 ext=False,
                 ex_normal_dir=None,
                 ex_anomaly_dir=None):

        super(FeaturesLoader, self).__init__()
        # self.normal_dir = normal_dir
        # self.anomaly_dir = anomaly_dir
        self.bucket_size = bucket_size
        # load video list
        self.features_list_normal, self.features_list_anomaly = FeaturesLoader._get_features_list(
            normal_dir, anomaly_dir)

        if ext:
            # add extra datasets in training
            ex_features_list_normal, ex_features_list_anomaly = FeaturesLoader._get_features_list(
                ex_normal_dir, ex_anomaly_dir
            )
            self.features_list_normal += ex_features_list_normal
            self.features_list_anomaly += ex_features_list_anomaly

        self.iterations = iterations
        self.features_cache = dict()
        self.i = 0

    def shuffle(self):
        self.features_list_anomaly = np.random.permutation(self.features_list_anomaly)
        self.features_list_normal = np.random.permutation(self.features_list_normal)

    def __len__(self):
        return self.iterations

    def __getitem__(self, index):
        if self.i == len(self):
            self.i = 0
            raise StopIteration

        succ = False
        while not succ:
            try:
                feature, label = self.get_features()
                succ = True
            except Exception as e:
                index = np.random.choice(range(0, self.__len__()))
                logging.warning("VideoIter:: ERROR!! (Force using another index:\n{})\n{}".format(
                    index, e))

        self.i += 1
        return feature, label

    def get_features(self):
        self.shuffle()  # optional
        normal_paths = np.random.choice(self.features_list_normal, size=self.bucket_size)
        abnormal_paths = np.random.choice(self.features_list_anomaly, size=self.bucket_size)
        all_paths = np.concatenate([normal_paths, abnormal_paths])
        features = torch.stack([self.read_features(subpath) for subpath in all_paths])
        labels = [0] * self.bucket_size + [1] * self.bucket_size
        return features, torch.tensor(labels)

    def read_features(self, path):
        feat = np.load(path, allow_pickle=True)
        feat = torch.Tensor(feat)
        feat = feat.flatten(start_dim=1, end_dim=2)
        # print(feat.shape)

        return feat

    @staticmethod
    def _get_features_list(normal_feat_dir, anomaly_feat_dir) -> Tuple[List, List]:
        normal_feat_list = sorted(list(Path(normal_feat_dir).glob("*.npy")))
        anomaly_feat_list = sorted(list(Path(anomaly_feat_dir).glob("*.npy")))

        return normal_feat_list, anomaly_feat_list


class FeaturesLoaderVal(object):
    def __init__(self, anomaly_feature_dir, clip_gt_dir) -> None:
        """
        The function takes in two directories, one containing the features and the other containing the
        ground truth labels. It then creates two lists, one containing the paths to the features and the
        other containing the ground truth labels

        :param anomaly_feature_dir: The directory where the anomaly features are stored
        :param clip_gt_dir: the directory where the ground truth for each clip is stored
        """
        # self.feature_paths = sorted(list(Path(anomaly_feature_dir).glob("*.npy")),
        #                             key=lambda s: int(s.stem.split('_')[-1]))  # NOTE: bug exists
        self.feature_paths = sorted(list(Path(anomaly_feature_dir).glob("*.npy")))
        self.clips_gt = self._get_clip_gts(clip_gt_dir)
        assert len(self.feature_paths) == len(
            self.clips_gt), str(len(self.feature_paths)) + '!=' + str(len(self.clips_gt))
        # print(self.feature_paths, self.clips_gt)

    def _get_clip_gts(self, clip_gt_dir):
        # clip_gts = sorted(list(Path(clip_gt_dir).glob("*.npy")), key=lambda s: int(s.stem.split('_')[-1]))
        clip_gts = sorted(list(Path(clip_gt_dir).glob("*.npy")))
        gts = [np.load(path, allow_pickle=True) for path in clip_gts]
        merged_gt = torch.Tensor(np.concatenate(gts))
        return merged_gt

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, index):
        feature_file = self.feature_paths[index]
        label = self.clips_gt[index]

        # print(feature_file, label)
        feature = np.load(feature_file, allow_pickle=True)
        feature = torch.tensor(feature)
        feature = feature.flatten(start_dim=1, end_dim=2)

        # additionally parse feature file names in order to get zeros of peaks
        peak_name = '_'.join(feature_file.stem.split('_')[-2::-1][::-1]) + ".npy"
        peak_index = int(feature_file.stem.split('_')[-1])

        return feature, label, peak_name, peak_index  # feature for a set of 3D-featured clips
