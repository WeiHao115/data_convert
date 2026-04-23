import os
import re
import numpy as np

def align_gripper_to_gopro(gopro_dir, gripper_file, output_file):
    """
    根据 GoPro 图像的时间戳（文件名提取），匹配夹爪的离散状态。
    """
    # 1. 解析夹爪状态文件
    transitions = []
    with open(gripper_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(r'\[(.*?)\].*Status:\s*(\d)', line)
            if match:
                ts = float(match.group(1))
                status = int(match.group(2))
                transitions.append((ts, status))
                    
    if not transitions:
        raise ValueError("未在夹爪状态文件中找到有效的状态转换数据。")

    transitions.sort(key=lambda x: x[0])
    trans_ts = np.array([x[0] for x in transitions])
    trans_status = np.array([x[1] for x in transitions])

    # 2. 读取并解析 GoPro 图像文件的时间戳
    image_files = [f for f in os.listdir(gopro_dir) if f.endswith('.jpg')]
    if not image_files:
        raise ValueError(f"目录 {gopro_dir} 中未找到 JPG 图像文件。")
        
    gopro_ts_list = []
    for f in image_files:
        try:
            ts = float(f.replace('.jpg', ''))
            gopro_ts_list.append(ts)
        except ValueError:
            continue
            
    # 确保目标时间戳升序排序
    gopro_ts = np.array(sorted(gopro_ts_list))
    N = len(gopro_ts)

    # 3. 匹配逻辑
    idx = np.searchsorted(trans_ts, gopro_ts, side='right') - 1

    matched_status = np.zeros(N, dtype=int)
    valid_mask = idx >= 0
    matched_status[valid_mask] = trans_status[idx[valid_mask]]

    # 4. 组装结果并保存
    result = np.column_stack((gopro_ts, matched_status))
    np.savetxt(output_file, result, fmt='%.6f %d')
    print(f"  -> 成功处理 {N} 帧图像。结果保存至: {output_file}")

def batch_process_directories(root_dir):
    """
    遍历指定根目录下的所有子文件夹，自动寻找有效数据并进行处理。
    """
    print(f"开始遍历目录: {root_dir}")
    processed_count = 0
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 判定当前目录是否为有效的数据采集目录
        if "gripper_state.txt" in filenames and "gopro" in dirnames:
            print(f"检测到有效数据目录: {dirpath}")
            
            gopro_dir = os.path.join(dirpath, "gopro")
            gripper_file = os.path.join(dirpath, "gripper_state.txt")
            output_file = os.path.join(dirpath, "gripper_state_time.txt")
            
            try:
                align_gripper_to_gopro(gopro_dir, gripper_file, output_file)
                processed_count += 1
            except Exception as e:
                print(f"  -> 处理失败: {e}")

    print(f"遍历结束，共成功处理 {processed_count} 个文件夹。")

if __name__ == "__main__":
    # 设定包含所有采集批次的父级目录
    # 例如包含 1111, 1212 等子文件夹的上一层目录
    ROOT_DIRECTORY = "/home/k202/0406" 
    
    if os.path.exists(ROOT_DIRECTORY):
        batch_process_directories(ROOT_DIRECTORY)
    else:
        print(f"错误: 根目录 {ROOT_DIRECTORY} 不存在，请检查路径。")