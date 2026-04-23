import sys
import os
import glob
import numpy as np
import cv2
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from bisect import bisect_left
from datetime import timedelta
from PIL import Image
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata


LEROBOT_PATH = "/home/k202/lerobot/src" 

if LEROBOT_PATH not in sys.path:
    sys.path.append(LEROBOT_PATH)

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import VideoEncodingManager
except ImportError as e:
    print(f"导入 LeRobot 失败: {e}")
    exit(1)

try:
    from transform_utils import convert_pose_quat2mat, convert_pose_mat2quat
except ImportError as e:
    print(f"导入 transform_utils 失败: {e}")
    exit(1)


# 全局常量与参数配置

class Config:
    # 路径配置
    DATASET_ROOT_DIR = "/home/k202/0408"
    BASE_OUTPUT_DIR = "/home/k202/Insert_Notac"
    os.makedirs(BASE_OUTPUT_DIR, exist_ok = True)
    # 任务配置
    TASK_NAME = "Insert the plug into the power strip"
    TASK_DESC = "Insert the plug into the power strip"
    REPO_ID = f"user/{TASK_NAME}"
    ROBOT_TYPE = "umi"
    
    # 采样与图像配置
    TARGET_FPS = 30
    STATE_DIM = 8
    RESIZE_WRIST = (224, 224)   
    GOPRO_TIMEZONE_OFFSET = 0
    # 标定参数 (UMI to TCP)
    T_ROBOT_CAPTURE = np.array([[ 0.02437661, -0.99968762, -0.00551686,  0.64165423],
                                [ 0.99970222,  0.02437005,  0.00125216,  0.38837618],
                                [-0.00111733, -0.00554574,  0.999984,    0.0126956 ],
                                [ 0.,          0.,          0. ,         1.        ]])
    UMI_POS = np.array([240.99915140650108, 234.05686586094797, 306.7073704633528]) / 1000  
    UMITCP_POS = np.array([336.5609397965651, 232.25531595494857, 214.80757938571125]) / 1000 
    P_UMI_UMITCP = UMITCP_POS - UMI_POS
    T_UMI_UMITCP = np.array([[1, 0, 0, P_UMI_UMITCP[0]],
                             [0, 1, 0, P_UMI_UMITCP[1]],
                             [0, 0, 1, P_UMI_UMITCP[2]],
                             [0, 0, 0, 1]])
    T_UMITCP_TCP = np.array([[ 0,  0,  1,  0],
                             [-1,  0,  0,  0],
                             [ 0, -1,  0,  0],
                             [ 0,  0,  0,  1]])


# 辅助工具函数

def calculate_relative_action(curr_state, next_state):
    pose_curr = curr_state[:7]
    pose_next = next_state[:7]
    gripper_next = next_state[7]

    mat_curr = convert_pose_quat2mat(pose_curr)
    mat_next = convert_pose_quat2mat(pose_next)
    mat_rel = np.linalg.inv(mat_curr) @ mat_next
    action_pose = convert_pose_mat2quat(mat_rel)
    
    action = np.append(action_pose, gripper_next)
    return action.astype(np.float32)

def find_nearest_index(array, value):
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

def safe_load_and_resize_image(path, target_size):
    if path and os.path.exists(path):
        try:
            img_array = np.fromfile(path, dtype=np.uint8)
            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            img_bgr = None
            print(f"读取图像异常 {path}: {e}")

        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            short_edge = min(h, w)
            start_x = (w - short_edge) // 2
            start_y = (h - short_edge) // 2
            img_cropped = img_rgb[start_y:start_y+short_edge, start_x:start_x+short_edge]
            img_resized = cv2.resize(img_cropped, target_size)
            return Image.fromarray(img_resized)
            
    return Image.fromarray(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))


# 视频与图像索引处理

def get_gopro_start_time(video_path):
    if not os.path.exists(video_path) or not os.access(video_path, os.R_OK):
        return None
    try:
        parser = createParser(video_path)
        if not parser: return None
        metadata = extractMetadata(parser)
        if not metadata: 
            parser.close()
            return None
        creation_date = metadata.get('creation_date')
        parser.close()
        if creation_date:
            return (creation_date + timedelta(hours=Config.GOPRO_TIMEZONE_OFFSET)).timestamp()
    except Exception as e:
        print(f"解析视频元数据异常: {e}")
    return None

def extract_video_to_images(video_path, output_dir, start_timestamp, target_fps):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频源: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = target_fps
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        current_ts = start_timestamp + (frame_count / fps)
        img_path = os.path.join(output_dir, f"{current_ts:.6f}.jpg")
        cv2.imwrite(img_path, frame)
        frame_count += 1
    cap.release()

def get_image_files(folder_path):
    if not os.path.exists(folder_path):
        return np.array([]), []
    
    files = glob.glob(os.path.join(folder_path, "*.jpg")) + \
            glob.glob(os.path.join(folder_path, "*.png"))
    
    valid_data = []
    for f in files:
        try:
            # 提取文件名并转换为浮点数
            ts = float(os.path.splitext(os.path.basename(f))[0])
            valid_data.append((ts, f))
        except ValueError:
            continue
            
    # 严格按照时间戳的数值进行升序排序
    valid_data.sort(key=lambda x: x[0])
    
    if not valid_data:
        return np.array([]), []
        
    timestamps = np.array([x[0] for x in valid_data])
    valid_files = [x[1] for x in valid_data]
    
    return timestamps, valid_files


# 运动学解算模块

def process_kinematics(episode_dir):
    """
    检查目录中的预处理文件，如果存在则直接读取；
    否则寻找 UMI 原始文件进行运动学转换并返回数据。
    返回: timestamps (np.array), robot_data (np.array [N, 8])
    """
    path_csv = os.path.join(episode_dir, "robot_data.csv")
    path_txt = os.path.join(episode_dir, "robot_data.txt")
    
    # 优先读取已存在的处理后数据
    if os.path.exists(path_txt):
        cols = ['timestamp', 'ee_x', 'ee_y', 'ee_z', 'ee_qx', 'ee_qy', 'ee_qz', 'ee_qw', 'gripper']
        df = pd.read_csv(path_txt, sep='\s+', comment='#', names=cols)
        return df['timestamp'].values, df.iloc[:, 1:].values
    elif os.path.exists(path_csv):
        df = pd.read_csv(path_csv)
        cols = ['ee_x', 'ee_y', 'ee_z', 'ee_qx', 'ee_qy', 'ee_qz', 'ee_qw', 'gripper']
        return df['timestamp'].values, df[cols].values

    # 若无处理后数据，则在内存中进行原始数据解算
    umi_files = glob.glob(os.path.join(episode_dir, "umi_body_abs.txt"))
    gripper_state_files = glob.glob(os.path.join(episode_dir, "gripper_state_time.txt"))
    # force_files = glob.glob(os.path.join(episode_dir, "force_torque.txt"))


    if not (umi_files and gripper_state_files):
        print(f"目录 {episode_dir} 缺少运动学解算所需的原始数据。")
        return None, None

    # 读取原始数据 
    umi_data = np.loadtxt(umi_files[0])
    timestamps = umi_data[:, 0]
    
    T_capture_UMI_quat = umi_data[:, 1:] / 1000.0
    T_capture_UMI_mat = convert_pose_quat2mat(T_capture_UMI_quat)

    T_robot_TCP_mat = Config.T_ROBOT_CAPTURE[None] @ T_capture_UMI_mat @ Config.T_UMI_UMITCP[None] @ Config.T_UMITCP_TCP[None]
    T_robot_TCP_quat = convert_pose_mat2quat(T_robot_TCP_mat)

    # 处理夹爪状态
    # gripper_data.txt 格式为 [timestamp, gripper_state] 
    gripper_data = np.loadtxt(gripper_state_files[0])
    gripper_timestamps = gripper_data[:, 0]
    raw_gripper_states = gripper_data[:, 1] 



    # 将夹爪状态对齐至主体时间戳
    gripper_states_aligned = np.zeros((len(timestamps), 1))
    for i in range(len(timestamps)):
        time_diff = np.abs(gripper_timestamps - timestamps[i])
        closest_idx = np.argmin(time_diff)
        gripper_states_aligned[i, 0] = raw_gripper_states[closest_idx]


    # 拼接 xyz + 四元数 + 夹爪状态 
    robot_data = np.hstack((T_robot_TCP_quat, gripper_states_aligned))
    print(f"最终机器人状态张量的完整形状: {robot_data}")
    return timestamps, robot_data

# 主控制流

def get_feature_config():
    h_wrist, w_wrist = Config.RESIZE_WRIST[1], Config.RESIZE_WRIST[0]

    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (Config.STATE_DIM,),
            "names": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (Config.STATE_DIM,), 
            "names": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
        },
        "observation.images.gopro": {
            "dtype": "video",
            "shape": (h_wrist, w_wrist, 3),
            "names": ["height", "width", "channels"],
        }
    }

def main():
    output_dir = os.path.join(Config.BASE_OUTPUT_DIR, Config.TASK_NAME)
    if os.path.exists(output_dir):
        print(f"清理旧输出目录: {output_dir}")
        shutil.rmtree(output_dir)

    dataset = LeRobotDataset.create(
        repo_id=Config.REPO_ID,
        fps=Config.TARGET_FPS,
        root=output_dir,
        features=get_feature_config(),
        use_videos=True,
        robot_type=Config.ROBOT_TYPE,
        image_writer_threads=8,
        batch_encoding_size=1
    )

    episode_dirs = sorted([os.path.join(Config.DATASET_ROOT_DIR, d) 
                           for d in os.listdir(Config.DATASET_ROOT_DIR) 
                           if os.path.isdir(os.path.join(Config.DATASET_ROOT_DIR, d))])

    print(f"共发现 {len(episode_dirs)} 个序列目录。开始端到端转换...")

    with VideoEncodingManager(dataset):
            for ep_dir in episode_dirs:
                ep_name = os.path.basename(ep_dir)
                
                # 1. 优先解析图像资源，建立时间轴基准
                path_video = os.path.join(ep_dir, "gopro_video.MP4")
                dir_wr_extracted = os.path.join(ep_dir, "gopro")
                dir_wr_image = os.path.join(ep_dir, "gopro")

                if os.path.exists(path_video):
                    video_start_ts = get_gopro_start_time(path_video)
                    if video_start_ts is not None:
                        extract_video_to_images(path_video, dir_wr_extracted, video_start_ts, Config.TARGET_FPS)
                    ts_wrist_rgb, files_wrist_rgb = get_image_files(dir_wr_extracted)
                elif os.path.exists(dir_wr_image):
                    ts_wrist_rgb, files_wrist_rgb = get_image_files(dir_wr_image)
                else:
                    ts_wrist_rgb, files_wrist_rgb = np.array([]), []
                    
                if len(ts_wrist_rgb) == 0:
                    print(f"跳过目录 {ep_name}: 未找到 GoPro 相机图像作为时间基准。")
                    continue

                # 2. 运动学解算与状态读取
                robot_ts, robot_data = process_kinematics(ep_dir)
                if robot_ts is None or len(robot_ts) < 2:
                    print(f"跳过目录 {ep_name}: 状态数据缺失或帧数过少。")
                    continue



                # 4. 以图像时间为基准进行循环与对齐
                num_samples = len(ts_wrist_rgb)
                print(f"正在处理 Episode [{ep_name}] (以图片时间为基准，共 {num_samples} 帧)...")
                
                for i in tqdm(range(num_samples - 1), desc=ep_name, leave=False):
                    # 严格使用当前图片和下一张图片的时间戳
                    curr_target_time = ts_wrist_rgb[i]
                    next_target_time = ts_wrist_rgb[i + 1]
                    
                    curr_r_idx = find_nearest_index(robot_ts, curr_target_time)
                    next_r_idx = find_nearest_index(robot_ts, next_target_time)
                    
                    # 检查机器人状态索引是否有效
                    if curr_r_idx is None or next_r_idx is None:
                        continue
                        
                    curr_state = robot_data[curr_r_idx]
                    next_state = robot_data[next_r_idx]
                    
                    action = calculate_relative_action(curr_state, next_state)
                    
                    # 当前腕部图像直接按索引获取，其他图像进行时间对齐
                    path_wr = files_wrist_rgb[i]
                    img_wr = safe_load_and_resize_image(path_wr, Config.RESIZE_WRIST)

                    lerobot_frame = {
                        "observation.state": curr_state.astype(np.float32),
                        "action": action,
                        "observation.images.gopro": img_wr,
                        "task": Config.TASK_DESC
                    }
                    
                    dataset.add_frame(lerobot_frame)

                dataset.save_episode()

    dataset.finalize()
    print(f"全部数据转换完毕。数据集已输出至: {output_dir}")

if __name__ == "__main__":
    main()