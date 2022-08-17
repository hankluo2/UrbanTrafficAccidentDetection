from pathlib import Path
import queue


def exp_dir(prfx='scene', save_dir=None) -> str:
    """Get output directory string in current workspace.

    Args:
        prfx (str): preffix of the saving derectory.
        save_dir (optional, str): directory to save experiment data. Default to None.

    Returns:
        str: directory to save output data.
    """
    save_path = Path(save_dir)
    dir_list = [str(dir_) for dir_ in save_path.glob(prfx + '*')]
    if dir_list != []:
        dir_list.sort(key=lambda s: int(s.split('-')[-1]))
        next_exp_num = int(dir_list[-1].split('-')[-1]) + 1
        new_dir = dir_list[-1].split('-')[-2] + '-' + str(next_exp_num).zfill(2)
    else:
        new_dir = save_dir + f'/{prfx}-01'

    Path(new_dir).mkdir(parents=True, exist_ok=True)
    return new_dir


def retrieve_data(sensor_queue, frame, timeout=5):
    while True:
        try:
            data = sensor_queue.get(True, timeout)  # wait for 5 seconds, if no data receive, throw error.
        except queue.Empty:
            return None
        if data.frame == frame:  # match: whether of the same frame
            return data
