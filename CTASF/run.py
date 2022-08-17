import os
import time

if __name__ == "__main__":
    # generate 5 scenes
    for i in range(1, 6):
        cmd = f"python main.py scene_num={str(i).zfill(2)}"
        print(cmd)
        os.system(cmd)
        time.sleep(20)  # wait for 20 seconds...
