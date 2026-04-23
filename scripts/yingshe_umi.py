import sys
sys.path.append("/home/ywl/rekep_multicam/src/rekep_multicam/scripts")
import sys
import numpy as np
from transform_utils import convert_pose_quat2mat, convert_pose_quat2euler, \
    convert_pose_mat2quat, convert_pose_quat2euler, convert_pose_euler2quat



# 阈值为10mm
def detect_gripper_events_by_accumulation(coords1, coords2, threshold=10):
    """
    通过累加相邻帧距离变化量判断夹爪动作。
    
    参数:
    coords1: np.ndarray, 维度 [N, 3], 夹指1的坐标序列
    coords2: np.ndarray, 维度 [N, 3], 夹指2的坐标序列
    threshold: 累计位移阈值，单位需与输入坐标一致（如 0.01 代表 1cm）
    window_size: 平滑窗口大小，用于降低高频噪声
    
    返回:
    List[List[int, int]]: 状态变化列表 [帧索引, 状态码]
                          状态码 0: 开始闭合 (Grasp)
                          状态码 1: 开始打开 (Release)
    """
    # 计算每一帧的欧氏距离
    distances = np.linalg.norm(coords1 - coords2, axis=1)
    # 计算相邻帧的位移增量 delta_d = d_t - d_{t-1}
    deltas = np.diff(distances)
    
    events = {}
    accumulated_motion = 0.0
    start_frame = 0
    
    # 当前系统所处的状态标识：-1 寻找中, 0 正在闭合, 1 正在打开
    # 为了避免重复触发同一状态，记录上一次触发的动作类型
    last_triggered_state = -1 

    for i, delta in enumerate(deltas):
        # 如果当前的增量方向与累计方向相反，且尚未触发状态，则重新开始累计
        # 判断逻辑：如果 delta 与当前累计值异号，重置起点
        if (delta > 0 and accumulated_motion < 0) or (delta < 0 and accumulated_motion > 0):
            accumulated_motion = delta
            start_frame = i
        else:
            accumulated_motion += delta
        
        # 检查累计量是否超过阈值
        # 累计减少超过阈值 -> 判定为闭合开始
        if accumulated_motion <= -threshold:
            if last_triggered_state != 0:
                events.update({
                    start_frame: 0
                })
                last_triggered_state = 0
            # 触发后重置累计器，准备检测下一个反向动作
            accumulated_motion = 0
            
        # 累计增加超过阈值 -> 判定为打开开始
        elif accumulated_motion >= threshold:
            if last_triggered_state != 1:
                events.update({
                    start_frame: 1
                })
                last_triggered_state = 1
            # 触发后重置累计器
            accumulated_motion = 0
    return events



if __name__ == "__main__":

    ROOT_PATH = '/home/ywl/317data'
    
    T_robot_capture = np.array([[ 0.0081, -0.9998, -0.0203,  0.6511],
                                [ 1.,      0.0081, -0.0002,  0.4073],
                                [ 0.0004, -0.0203,  0.9998,  0.0148],
                                [ 0.,      0. ,     0.,      1.    ]])
    UMI_POS = np.array([331.42073612987366, 327.2589200263196, 291.4834539846908]) / 1000
    UMITCP_POS = np.array([437.79758518341583, 318.0101023486956, 224.30811975625292]) / 1000
    P_UMI_UMITCP = UMITCP_POS - UMI_POS
    T_UMI_UMITCP = np.array([[1, 0, 0, P_UMI_UMITCP[0]],
                             [0, 1, 0, P_UMI_UMITCP[1]],
                             [0, 0, 1,  P_UMI_UMITCP[2]],
                             [0, 0, 0, 1]])
    
    T_capture_UMI = np.loadtxt(ROOT_PATH + "/5/rawdata/test_arr_umi_body_abs_5.txt")[::20][:, 1:] / 1000  # [N 7]
    T_capture_UMI = convert_pose_quat2mat(T_capture_UMI)    # [N 4 4]

    T_UMITCP_TCP = np.array([[0, 0, 1, 0],
                             [-1, 0, 0, 0],
                             [0, -1, 0, 0],
                             [0, 0, 0, 1]])
    
    T_robot_TCP = T_robot_capture[None] @ T_capture_UMI @ T_UMI_UMITCP[None] @ T_UMITCP_TCP[None]
    T_robot_TCP = convert_pose_mat2quat(T_robot_TCP)        # 机械臂位姿

    left_gripper_pos = np.loadtxt(ROOT_PATH + "/5/rawdata/test_gripper_left_5.txt")[::20]
    right_gripprt_ros = np.loadtxt(ROOT_PATH + "/5/rawdata/test_gripper_right_5.txt")[::20]
    min_rows = min(len(left_gripper_pos), len(right_gripprt_ros))
    if len(left_gripper_pos) != len(right_gripprt_ros):
        print(f"[警告] 左右夹爪数据行数不一致，已自动裁剪对齐。")
        left_gripper_pos = left_gripper_pos[:min_rows]
        right_gripprt_ros = right_gripprt_ros[:min_rows]
    left_gripper_xyz = left_gripper_pos[:, 1:4]
    right_gripper_xyz = right_gripprt_ros[:, 1:4]
    gripper_event = detect_gripper_events_by_accumulation(left_gripper_xyz, right_gripper_xyz)
    print(gripper_event)


    timestamps = np.loadtxt(ROOT_PATH + "/5/rawdata/test_arr_umi_body_abs_5.txt")[::20][: , 0:1]
    gripper_timestamps = left_gripper_pos[:, 0]
    num_gripper_frames = len(gripper_timestamps)
    raw_gripper_states = np.zeros(num_gripper_frames)
    
    current_state = 0.0
    for i in range(num_gripper_frames):
        if i in gripper_event:
            # 状态映射：算法输出 0 为闭合，映射为 1.0；输出 1 为打开，映射为 0.0
            if gripper_event[i] == 0:
                current_state = 1.0
            elif gripper_event[i] == 1:
                current_state = 0.0
        raw_gripper_states[i] = current_state

    num_frames = timestamps.shape[0]
    gripper_states = np.zeros((num_frames, 1))
    
    for i in range(num_frames):
        time_diff = np.abs(gripper_timestamps - timestamps[i, 0])
        closest_idx = np.argmin(time_diff)
        gripper_states[i, 0] = raw_gripper_states[closest_idx]

 
    combined_data = np.hstack((timestamps, T_robot_TCP, gripper_states))
    save_path = ROOT_PATH + "/5/predata/robot_data.txt"
    np.savetxt(save_path,
               combined_data,
               fmt='%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %d',
               delimiter=' ',
               header='timestamp x y z qx qy qz qw gripper_state',
               comments='# ')


