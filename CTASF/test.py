"""
Environment: Windows
"""
import hydra
from pathlib import Path
import json

from datamaker import *
from preprocess import *


@hydra.main(config_path='./configure', config_name='carla')
def main(cfg):
    # make data
    if cfg.scene_num != 0:
        print("Begin to make raw data ...\n")
        make_data(cfg)
    else:
        print("Data maker unset. Check config file.")

    # organize data
    if cfg.save_tube:
        get_all_merged_tubes_from_scenes(cfg)

    # test tracking bounds
    scene_dir = cfg.save_dir + "\\scene-1"
    with open(scene_dir + "\\tubes.json", 'r') as read:
        tubes = json.load(read)
    tube = tubes[0]  # for test case
    (h1, w1), (h2, w2) = resize_bbox(scene_dir, tube)
    print(h1, w1, h2, w2)


if __name__ == '__main__':
    main()
