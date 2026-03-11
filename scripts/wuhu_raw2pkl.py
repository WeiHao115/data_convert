import os
import pickle
import numpy as np
import glob
from bisect import bisect_left
from tqdm import tqdm
import pandas as pd


TARGET_FPS = 30.0
CHUNK_SIZE = 2000


DATASET_ROOT_DIR = "/home/ywl/下载/Data"
# 统一的输出目录
OUTPUT_DIR = "/home/ywl/test_data_pkl"


def find_nearest_index(array, value):
    """二分查找最近时间戳索引"""
    if len(array) == 0: 
        return None
    idx = bisect_left(array, value)
    if idx == 0: 
        return 0
    if idx == len(array): 
        return len(array) - 1
    before = array[idx - 1]
    after = array[idx]
    return idx if (after - value < value - before) else idx - 1

def get_image_files(folder_path):
    """索引图像文件并提取时间戳"""
    if not os.path.exists(folder_path):
        return np.array([]), []

    files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")) + 
                   glob.glob(os.path.join(folder_path, "*.png")))
    timestamps = []
    valid_files = []
    
    for f in files:
        try:
            ts = float(os.path.splitext(os.path.basename(f))[0])
            timestamps.append(ts)
            valid_files.append(f)
        except ValueError:
            continue
            
    return np.array(timestamps), valid_files

def load_robot_from_csv(csv_path):
    """从 CSV 读取机器人状态数据"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到文件: {csv_path}")

    df = pd.read_csv(csv_path)
    
    if 'timestamp' not in df.columns:
        raise ValueError("缺少 timestamp 列")
    
    timestamps = df['timestamp'].values
    target_columns = ['ee_x', 'ee_y', 'ee_z', 'ee_qx', 'ee_qy', 'ee_qz', 'ee_qw', 'gripper']
    available_cols = [col for col in target_columns if col in df.columns]
    
    if not available_cols:
        raise ValueError("数据源缺少位姿或夹爪数据列")
        
    poses = df[available_cols].values
    return timestamps, poses



def process_single_episode(episode_dir, output_dir, episode_name):
    """处理单个数据文件夹并导出 pkl"""
    path_csv = os.path.join(episode_dir, "robot_data.csv")
    dir_tl = os.path.join(episode_dir, "tactile_left")
    dir_tr = os.path.join(episode_dir, "tactile_right")
    dir_wr = os.path.join(episode_dir, "wrist_rgb")

    try:
        robot_ts, robot_data = load_robot_from_csv(path_csv)
    except Exception as e:
        print(f"跳过目录 {episode_name}: 状态文件加载异常 ({e})")
        return

    if len(robot_ts) == 0:
        print(f"跳过目录 {episode_name}: 状态数据为空")
        return

    start_ts = robot_ts[0]
    end_ts = robot_ts[-1]
    duration = end_ts - start_ts
    num_samples = int(duration * TARGET_FPS)

    ts_tac_left, files_tac_left = get_image_files(dir_tl)
    ts_tac_right, files_tac_right = get_image_files(dir_tr)
    ts_wrist_rgb, files_wrist_rgb = get_image_files(dir_wr)


    buffer = []
    chunk_idx = 0

    print(f"正在处理 {episode_name} ({num_samples} 帧)...")
    for i in tqdm(range(num_samples), desc=episode_name, leave=False):
        target_time = start_ts + (i / TARGET_FPS)
        
        r_idx = find_nearest_index(robot_ts, target_time)
        cur_state = robot_data[r_idx] if r_idx is not None else None

        path_tl = files_tac_left[find_nearest_index(ts_tac_left, target_time)] if len(ts_tac_left) > 0 else None
        path_tr = files_tac_right[find_nearest_index(ts_tac_right, target_time)] if len(ts_tac_right) > 0 else None
        path_wr = files_wrist_rgb[find_nearest_index(ts_wrist_rgb, target_time)] if len(ts_wrist_rgb) > 0 else None

        sample = {
            "episode_name": episode_name,
            "timestamp": target_time,
            "robot_state": cur_state,
            "tactile_left_path": path_tl,
            "tactile_right_path": path_tr,
            "wrist_rgb_path": path_wr,
        }
        buffer.append(sample)

        if len(buffer) >= CHUNK_SIZE:
            save_path = os.path.join(output_dir, f"{episode_name}_meta_chunk_{chunk_idx:04d}.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(buffer, f)
            buffer = []
            chunk_idx += 1

    if buffer:
        save_path = os.path.join(output_dir, f"{episode_name}_meta_chunk_{chunk_idx:04d}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(buffer, f)


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not os.path.exists(DATASET_ROOT_DIR):
        print(f"根目录不存在: {DATASET_ROOT_DIR}")
        return

    episode_dirs = [os.path.join(DATASET_ROOT_DIR, d) for d in os.listdir(DATASET_ROOT_DIR) 
                    if os.path.isdir(os.path.join(DATASET_ROOT_DIR, d))]
    
    episode_dirs.sort()

    print(f"共发现 {len(episode_dirs)} 个序列目录。开始处理...")

    for ep_dir in episode_dirs:
        episode_name = os.path.basename(ep_dir)
       
        if not os.path.exists(os.path.join(ep_dir, "robot_data.csv")):
            print(f"跳过目录 {episode_name}: 缺少核心文件 robot_data.csv")
            continue
            
        process_single_episode(ep_dir, OUTPUT_DIR, episode_name)

    print("全部数据处理完毕。")

if __name__ == "__main__":
    main()