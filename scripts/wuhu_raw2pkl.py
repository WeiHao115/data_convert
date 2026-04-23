import os
import pickle
import numpy as np
import glob
from bisect import bisect_left
from tqdm import tqdm
import pandas as pd
import cv2
from datetime import timedelta
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata


TARGET_FPS = 30.0
CHUNK_SIZE = 20000
GOPRO_TIMEZONE_OFFSET = 0

DATASET_ROOT_DIR = "/home/ywl/simdata"
# 统一的输出目录
OUTPUT_DIR = "/home/ywl/simdata/pkl"


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

def extract_video_to_images(video_path, output_dir, start_timestamp, target_fps=30.0):
    """/home/ywl/simdata
        video_path (str): 视频文件路径
        output_dir (str): 提取后图像存放的目标文件夹
        start_timestamp (float): 视频第一帧对应的初始时间戳
        target_fps (float): 目标帧率
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频源: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = target_fps

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_ts = start_timestamp + (frame_count / fps)
        img_path = os.path.join(output_dir, f"{current_ts:.6f}.jpg")
        cv2.imwrite(img_path, frame)
        frame_count += 1

    cap.release()

def get_gopro_start_time(video_path):
    """从 GoPro 视频文件中提取创建时间并转换为时间戳"""
    print(f"检查视频路径: {video_path}")
    
    if not os.path.exists(video_path):
        print("错误: 文件不存在。")
        return None
    
    if not os.access(video_path, os.R_OK):
        print("错误: 文件存在，但无读取权限。")
        return None

    try:
        parser = createParser(video_path)
    except Exception as e:
        print(f"解析器异常: {e}")
        return None

    if not parser:
        print("错误: 无法识别此文件格式。")
        return None
    
    metadata = extractMetadata(parser)
    if not metadata: 
        print("错误: 无法提取元数据。")
        parser.close()
        return None
    
    creation_date = metadata.get('creation_date')
    parser.close()
    
    if creation_date:
        timestamp = (creation_date + timedelta(hours=GOPRO_TIMEZONE_OFFSET)).timestamp()
        print(f"成功提取视频时间戳: {timestamp}")
        return timestamp
    else:
        print("警告: 元数据中缺失 creation_date。该文件可能已被重新编码或剪辑。")
        return None


# def get_image_files(folder_path):
#     """索引图像文件并提取时间戳"""
#     if not os.path.exists(folder_path):
#         return np.array([]), []

#     files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")) + 
#                    glob.glob(os.path.join(folder_path, "*.png")))
#     timestamps = []
#     valid_files = []
    
#     for f in files:
#         try:
#             ts = float(os.path.splitext(os.path.basename(f))[0])
#             timestamps.append(ts)
#             valid_files.append(f)
#         except ValueError:
#             continue
            
#     return np.array(timestamps), valid_files


###仅针对虚拟采集数据的处理代码（因为图片命名不是真实时间戳，但是所有传感器同步开始结束），否则复用上面的函数
def get_image_files(folder_path):
    """索引图像文件，提取时间戳并严格按数值排序"""
    if not os.path.exists(folder_path):
        return np.array([]), []

    files = glob.glob(os.path.join(folder_path, "*.jpg")) + \
            glob.glob(os.path.join(folder_path, "*.png"))
    
    data = []
    for f in files:
        try:
            ts = float(os.path.splitext(os.path.basename(f))[0])
            data.append((ts, f))
        except ValueError:
            continue
            
    data.sort(key=lambda x: x[0])
    
    timestamps = [x[0] for x in data]
    valid_files = [x[1] for x in data]
    
    return np.array(timestamps), valid_files





def load_robot_from_txt(txt_path):
    """从 TXT 读取机器人状态数据，处理空格分隔并统一列名"""
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"未找到文件: {txt_path}")

    if 'gripper_state' in df.columns:
        df.rename(columns={'gripper_state': 'gripper'}, inplace=True)

    standard_columns = ['timestamp', 'ee_x', 'ee_y', 'ee_z', 'ee_qx', 'ee_qy', 'ee_qz', 'ee_qw', 'gripper']
    
    # sep='\s+' 支持解析一个或多个空格/制表符
    # comment='#' 自动忽略第一行的表头
    # names=standard_columns 为读取的数据强制覆写兼容的列名
    df = pd.read_csv(txt_path, sep='\s+', comment='#', names=standard_columns)
    
    if 'timestamp' not in df.columns:
        raise ValueError("缺少 timestamp 列")
    
    timestamps = df['timestamp'].values
    
    # 提取位姿与夹爪数据
    pose_columns = ['ee_x', 'ee_y', 'ee_z', 'ee_qx', 'ee_qy', 'ee_qz', 'ee_qw', 'gripper']
    poses = df[pose_columns].values
    
    return timestamps, poses



def load_robot_from_csv(csv_path):
    """从 CSV 读取机器人状态数据"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到文件: {csv_path}")

    df = pd.read_csv(csv_path)
    
    if 'timestamp' not in df.columns:
        raise ValueError("缺少 timestamp 列")
    
    if 'gripper_state' in df.columns:
        df.rename(columns={'gripper_state': 'gripper'}, inplace=True)

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
    path_txt = os.path.join(episode_dir, "robot_data.txt")
    dir_tl = os.path.join(episode_dir, "tactile_left")
    dir_tr = os.path.join(episode_dir, "tactile_right")
    path_video = os.path.join(episode_dir, "gopro_video.MP4")
    dir_wr_extracted = os.path.join(episode_dir, "wrist_rgb_extracted")
    dir_wr_image = os.path.join(episode_dir, "wrist_rgb")

    try:
        if os.path.exists(path_txt):
            robot_ts, robot_data = load_robot_from_txt(path_txt)
        elif os.path.exists(path_csv):    
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

    # ts_tac_left, files_tac_left = get_image_files(dir_tl)
    # ts_tac_right, files_tac_right = get_image_files(dir_tr)

    # if os.path.exists(path_video):   # sep='\s+' 支持解析一个或多个空格/制表符
    # # comment='#' 自动忽略第一行的表头
    # # names=standard_columns 为读取的数据强制覆写兼容的列名
    #     video_start_ts = get_gopro_start_time(path_video)
    #     extract_video_to_images(path_video, dir_wr_extracted, start_timestamp=video_start_ts, target_fps=TARGET_FPS)
    #     ts_wrist_rgb, files_wrist_rgb = get_image_files(dir_wr_extracted)
    # elif os.path.exists(dir_wr_image):
    #     ts_wrist_rgb, files_wrist_rgb = get_image_files(dir_wr_image)
    # else:
    #     ts_wrist_rgb, files_wrist_rgb = np.array([]), []
    #     print("不存在gopro或仿真提供的腕部相机图像，请检查数据是否完整")

    # buffer = []
    # chunk_idx = 0


    ts_tac_left, files_tac_left = get_image_files(dir_tl)
    ts_tac_right, files_tac_right = get_image_files(dir_tr)

    if os.path.exists(path_video): 
        video_start_ts = get_gopro_start_time(path_video)
        extract_video_to_images(path_video, dir_wr_extracted, start_timestamp=video_start_ts, target_fps=TARGET_FPS)
        ts_wrist_rgb, files_wrist_rgb = get_image_files(dir_wr_extracted)
    elif os.path.exists(dir_wr_image):
        ts_wrist_rgb, files_wrist_rgb = get_image_files(dir_wr_image)
    else:
        ts_wrist_rgb, files_wrist_rgb = np.array([]), []
        print("不存在gopro或仿真提供的腕部相机图像，请检查数据是否完整")

    # 时间轴强制对齐逻辑
    def align_sequence_timestamps(ts_array, start_ts, target_fps):
        """如果检测到使用的是 1, 2, 3 等序列号或相对时间，强制将其等距映射到真实物理时间戳上"""
        if len(ts_array) > 0 and ts_array[0] < 1e8:
            return start_ts + (ts_array - ts_array[0]) / target_fps
        return ts_array

    ts_tac_left = align_sequence_timestamps(ts_tac_left, start_ts, TARGET_FPS)
    ts_tac_right = align_sequence_timestamps(ts_tac_right, start_ts, TARGET_FPS)
    ts_wrist_rgb = align_sequence_timestamps(ts_wrist_rgb, start_ts, TARGET_FPS)
    # -----------------------------------

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
       
        # 同时检查 csv 和 txt
        if not (os.path.exists(os.path.join(ep_dir, "robot_data.csv")) or 
                os.path.exists(os.path.join(ep_dir, "robot_data.txt"))):
            print(f"跳过目录 {episode_name}: 缺少核心文件 robot_data.csv 或 robot_data.txt")
            continue
            
        process_single_episode(ep_dir, OUTPUT_DIR, episode_name)

    print("全部数据处理完毕。")

if __name__ == "__main__":
    # main()
    get_gopro_start_time("/media/ywl/PSSD/100GOPRO/GX010037.MP4")


    