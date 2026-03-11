import sys
import os
import pickle
import numpy as np
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from transform_utils import convert_pose_mat2quat, convert_pose_quat2mat


LEROBOT_PATH = "/home/ywl/lerobot/src" 
if LEROBOT_PATH not in sys.path:
    sys.path.append(LEROBOT_PATH)

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import VideoEncodingManager
    print("成功导入: lerobot.datasets.lerobot_dataset")
except ImportError as e:
    print(f"导入 LeRobot 失败: {e}")
    exit(1)


TASK_NAME = "tactile_manipulation_test" 
TASK_DESC = "robot manipulation with tactile and wrist camera feedback"

PKL_DATA_DIR = "/home/ywl/test/test_data_pkl" 
BASE_OUTPUT_DIR = "/home/ywl/test/test_data_lerobotdataset"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, TASK_NAME)
REPO_ID = f"user/{TASK_NAME}"

FPS = 30                        
ROBOT_TYPE = "umi"
STATE_DIM = 8               # 7维位姿 + 1维夹爪

RESIZE_WRIST = (224, 224)   
RESIZE_TACTILE = (320, 240) 


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

def safe_load_and_resize_image(path, target_size):
    """安全读取图像并缩放。若路径无效则返回全黑占位图"""
    if path and os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            return cv2.resize(img, target_size)
            
    return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

def get_feature_config():
    h_wrist, w_wrist = RESIZE_WRIST[1], RESIZE_WRIST[0]
    h_tac, w_tac = RESIZE_TACTILE[1], RESIZE_TACTILE[0]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (STATE_DIM,), 
            "names": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
        },
        "observation.images.wrist_rgb": {
            "dtype": "video",
            "shape": (h_wrist, w_wrist, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.tactile_left": {
            "dtype": "video",
            "shape": (h_tac, w_tac, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.tactile_right": {
            "dtype": "video",
            "shape": (h_tac, w_tac, 3),
            "names": ["height", "width", "channels"],
        }
    }
    return features



def convert_to_lerobot():
    # 1. 扫描并按 Episode 分组 pkl 文件
    pkl_paths = sorted(list(Path(PKL_DATA_DIR).glob("*_meta_chunk_*.pkl")))
    if not pkl_paths:
        print(f"目录 {PKL_DATA_DIR} 下未找到索引文件。")
        return
    
    episodes_map = defaultdict(list)
    for p in pkl_paths:
        # 文件名示例: episode_01_meta_chunk_0000.pkl
        ep_name = p.name.split("_meta_chunk_")[0]
        episodes_map[ep_name].append(p)
        
    print(f"扫描到 {len(pkl_paths)} 个索引块，划分为 {len(episodes_map)} 个 Episode。")

    # 2. 初始化 Dataset
    if os.path.exists(OUTPUT_DIR):
        print(f"清理旧输出目录: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
        
    features = get_feature_config()
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        root=OUTPUT_DIR,
        features=features,
        use_videos=True,
        robot_type=ROBOT_TYPE,
        image_writer_threads=8,
        batch_encoding_size=1
    )

    # 3. 逐个 Episode 处理
    with VideoEncodingManager(dataset):
        for ep_name in sorted(episodes_map.keys()):
            chunk_files = sorted(episodes_map[ep_name])
            
            # 合并当前 episode 的所有 chunk
            episode_data = []
            for chunk_file in chunk_files:
                with open(chunk_file, "rb") as f:
                    episode_data.extend(pickle.load(f))
                    
            total_frames = len(episode_data)
            if total_frames < 2:
                print(f"Episode [{ep_name}] 帧数过少 ({total_frames})，跳过。")
                continue
                
            print(f"正在转换 Episode [{ep_name}] (共 {total_frames} 帧)...")
            
            # 遍历当前 episode 的帧
            for i in tqdm(range(total_frames - 1), desc=ep_name, leave=False):
                curr_frame = episode_data[i]
                next_frame = episode_data[i+1]
                
                # 数据完整性检查
                if curr_frame["robot_state"] is None or next_frame["robot_state"] is None:
                    continue
                
                # 计算动作
                curr_state = curr_frame["robot_state"]
                next_state = next_frame["robot_state"]
                relative_action = calculate_relative_action(curr_state, next_state)
                
                # 读取图像
                img_wr = safe_load_and_resize_image(curr_frame["wrist_rgb_path"], RESIZE_WRIST)
                img_tl = safe_load_and_resize_image(curr_frame["tactile_left_path"], RESIZE_TACTILE)
                img_tr = safe_load_and_resize_image(curr_frame["tactile_right_path"], RESIZE_TACTILE)

                # 构建 LeRobot 数据帧
                lerobot_frame = {
                    "observation.state": curr_state.astype(np.float32),
                    "action": relative_action,
                    "observation.images.wrist_rgb": img_wr,
                    "observation.images.tactile_left": img_tl,
                    "observation.images.tactile_right": img_tr,
                    "task": TASK_DESC
                }
                
                dataset.add_frame(lerobot_frame)

            # 完成当前 episode 后保存
            dataset.save_episode()

    dataset.finalize()
    print(f"转换全部完成。数据集已输出至: {OUTPUT_DIR}")

if __name__ == "__main__":
    convert_to_lerobot()