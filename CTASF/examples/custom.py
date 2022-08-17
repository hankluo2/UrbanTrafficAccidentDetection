"""
@ filename custom.py
@ author Haohan Luo
@ date 2021/9/28
"""
from pathlib import Path


def exp_dir(prfx: str) -> str:
    """Get output directory string in current workspace.

    Args:
        prfx (str): preffix of the saving derectory.

    Returns:
        str: directory to save output data.
    """
    cwd = Path.cwd()
    # print(cwd)
    dir_list = [str(dir_) for dir_ in cwd.glob(prfx + '*')]
    sorted(dir_list, key=lambda s: int(s.split('-')[-1]))
    # print(dir_list)
    next_exp_num = int(dir_list[-1].split('-')[-1]) + 1
    new_dir = dir_list[-1].split('-')[-2] + '-' + str(next_exp_num)
    return new_dir


if __name__ == "__main__":
    print("Hello world!")
    preffix = '_out'
    exp_dir(preffix)
