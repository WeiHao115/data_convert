import os
os.environ['QT_LOGGING_RULES'] = 'qt.qpa.fonts.warning=false'

import sys
import time
import threading
import cv2
import serial
from datetime import datetime

import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
import re
from geometry_msgs.msg import WrenchStamped

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


sys.path.append("/home/k202/gsmini_ws/src")

try:
    import gelSight_SDK.examples.gsdevice as gsdevice
except ImportError:
    raise ImportError("无法导入 gelSight_SDK, 请检查系统路径是否存在。")




class GelSightManager:
    def __init__(self, 
                 dev1_id="GelSight Mini R0B 2DPF-C3HB:", 
                 dev2_id="GelSight Mini R0B 2DMA-NFYZ"):
        self.dev1 = gsdevice.Camera(dev1_id)
        self.dev2 = gsdevice.Camera(dev2_id)
        self.dev1.connect()
        self.dev2.connect()

        self.frame_1 = None
        self.frame_2 = None
        self.timestamp = 0.0
        
        self.running = True
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        while self.running:
            f1 = self.dev1.get_raw_image()
            f2 = self.dev2.get_raw_image()
            current_time = time.time()

            if f1 is not None and f2 is not None:
                with self.lock:
                    self.frame_1 = f1
                    self.frame_2 = f2
                    self.timestamp = current_time

    def get_tactile_frame(self):
        with self.lock:
            if self.frame_1 is not None and self.frame_2 is not None:
                return self.frame_1.copy(), self.frame_2.copy(), self.timestamp
        return None, None, 0.0

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()


class RealsenseRosManager:
    def __init__(self, topic_name="/camera/color/image_raw", save_dir=""):
        try:
            rospy.init_node("data_record_node", anonymous=True, disable_signals=True)
        except rospy.exceptions.ROSException:
            pass

        self.save_dir = save_dir
        self.bridge = CvBridge()
        
        self.lock = threading.Lock()
        self.current_frame = None
        self.timestamp = 0.0

        # 订阅话题
        self.sub_rs = rospy.Subscriber(topic_name, Image, self._callback, queue_size=10)
        print(f"ROS RealSense 订阅节点初始化完成，监听话题: {topic_name}")

    def _callback(self, msg):
        try:
            # 绕过 cv_bridge，直接解析数据
            img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            
            # ROS 默认通常是 RGB，OpenCV 需要 BGR
            if msg.encoding == "rgb8":
                cv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            else:
                cv_img = img_np # 如果已经是 bgr8 则直接赋值

            t = msg.header.stamp.to_sec()
            
            with self.lock:
                self.current_frame = cv_img.copy()
                self.timestamp = t
        except Exception as e:
            print(f"RealSense 图像转换解析失败: {e}")

    def get_latest_frame(self):
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy(), self.timestamp
        return None, 0.0

    def release(self):
        if hasattr(self, 'sub_rs'):
            self.sub_rs.unregister()




#腕部
class GoproManager:
    def __init__(self, device_id=2, width=1920, height=1080, fps=30):
        cv2.setNumThreads(1)
        self.cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频设备: {device_id}")

        self.running = True
        self.lock = threading.Lock()
        
        self.current_frame = None
        self.kernel_timestamp = 0.0

        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        while self.running:
            grabbed = self.cap.grab()
            if not grabbed:
                continue
            
            v4l2_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp_sec = v4l2_msec / 1000.0
            ret, buffer_frame = self.cap.retrieve()
            
            if ret and buffer_frame is not None:
                with self.lock:
                    self.current_frame = buffer_frame.copy()
                    self.kernel_timestamp = timestamp_sec

    def get_latest_frame(self):
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy(), self.kernel_timestamp
        return None, 0.0

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()

#夹爪状态类 端口读取
class SerialManager:
    def __init__(self, port='/dev/ttyUSB3', baud_rate=115200, save_dir=""):
        self.port = port
        self.baud_rate = baud_rate
        self.log_file = os.path.join(save_dir, "gripper_state.txt")
        self.running = True
        
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=0.1)
            print(f"--- 成功打开串口 {self.port}，日志将保存至 {self.log_file} ---")
            
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"\n--- 新的记录开启时间: {datetime.now()} ---\n")
                
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
        except Exception as e:
            print(f"串口初始化失败 {self.port}: {e}")
            self.ser = None

    def _update_loop(self):
        while self.running and self.ser and self.ser.is_open:
            try:
                raw_line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if not raw_line:
                    continue
                    
                if ',' in raw_line:
                    try:
                        mcu_ms, status = raw_line.split(',')
                        sys_time = time.time()
                        log_entry = f"[{sys_time:.6f}] MCU_Time: {mcu_ms}ms | Status: {status}\n"
                        with open(self.log_file, "a", encoding="utf-8") as f:
                            f.write(log_entry)
                            f.flush()
                    except ValueError:
                        print(f"异常数据格式被丢弃: {raw_line}")
                        continue
            except Exception as e:
                print(f"串口读取异常: {e}")
            
            time.sleep(0.005)

    def release(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()
        if self.ser and self.ser.is_open:
            self.ser.close()

#  ROS 位姿管理类  话题读取
class PoseManager:
    def __init__(self, save_dir=""):
        try:
            rospy.init_node("data_record_node", anonymous=True, disable_signals=True)
        except rospy.exceptions.ROSException:
            pass

        self.file_body = os.path.join(save_dir, "umi_body_abs.txt")
        self.sub_body = rospy.Subscriber("/vrpn_client_node/Body_0/0/pose", PoseStamped, self._callback_body, queue_size=60)   # +30300
        print("ROS UMI 位姿订阅节点初始化完成。")


    def _callback_body(self, msg: PoseStamped):
        t = msg.header.stamp.to_sec()
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        ox = msg.pose.orientation.x
        oy = msg.pose.orientation.y
        oz = msg.pose.orientation.z
        ow = msg.pose.orientation.w
        with open(self.file_body, mode='a+', encoding='utf-8') as f:
            f.write(f"{t} {x} {y} {z} {ox} {oy} {oz} {ow}\n")

    def release(self):
        if hasattr(self, 'sub_body'):
            self.sub_body.unregister()
        
        if not rospy.is_shutdown():
            rospy.signal_shutdown("Program terminated manually")

#六维力类  话题读取
class ForceTorqueManager:
    def __init__(self, topic_name="/force_sensor/wrench", save_dir=""):
        try:
            rospy.init_node("data_record_node", anonymous=True, disable_signals=True)
        except rospy.exceptions.ROSException:
            pass

        self.file_path = os.path.join(save_dir, "force_torque.txt")
        self.sub_ft = rospy.Subscriber(topic_name, WrenchStamped, self._callback, queue_size=60)
        print(f"ROS 六维力位姿订阅节点初始化完成，监听话题: {topic_name}")

    def _callback(self, msg: WrenchStamped):
        t = msg.header.stamp.to_sec()
        fx = msg.wrench.force.x
        fy = msg.wrench.force.y
        fz = msg.wrench.force.z
        tx = msg.wrench.torque.x
        ty = msg.wrench.torque.y
        tz = msg.wrench.torque.z
        
        with open(self.file_path, mode='a+', encoding='utf-8') as f:
            f.write(f"{t:.6f} {fx:.6f} {fy:.6f} {fz:.6f} {tx:.6f} {ty:.6f} {tz:.6f}\n")

    def release(self):
        if hasattr(self, 'sub_ft'):
            self.sub_ft.unregister()


def process_gripper_states(umi_file, gripper_file, output_file):
    """
    根据时间戳将夹爪状态匹配到机械臂位姿数据中。
    
    参数:
    umi_file: 机械臂位姿数据文件路径
    gripper_file: 夹爪状态日志文件路径
    output_file: 匹配结果的输出保存路径
    """
    
    # 1. 解析夹爪状态文件
    transitions = []
    with open(gripper_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 使用正则表达式提取时间戳和状态
            # 匹配格式: [1774850895.424666] MCU_Time: 1626020ms | Status: 1
            match = re.search(r'\[(.*?)\].*Status:\s*(\d)', line)
            if match:
                ts = float(match.group(1))
                status = int(match.group(2))
                transitions.append((ts, status))
                
    if not transitions:
        raise ValueError("未在夹爪状态文件中找到有效的状态转换数据。")

    # 按照时间戳升序排序，确保时序正确
    transitions.sort(key=lambda x: x[0])
    trans_ts = np.array([x[0] for x in transitions])
    trans_status = np.array([x[1] for x in transitions])

    # 2. 读取机械臂位姿文件
    # 假设数据以空格或制表符分隔
    umi_data = np.loadtxt(umi_file)
    umi_ts = umi_data[:, 0]
    N = len(umi_ts)

    # 3. 匹配逻辑 (核心)
    # 使用二分查找获取 umi_ts 在 trans_ts 中的插入位置
    # side='right' 保证取到的是当前时间戳“左侧最近”的一次状态变化
    idx = np.searchsorted(trans_ts, umi_ts, side='right') - 1

    # 初始化状态数组
    # 假设在第一次状态发生改变之前的默认夹爪状态为 0
    matched_status = np.zeros(N, dtype=int)
    
    # 如果索引 >= 0，说明该时间戳在第一次状态变化之后，直接赋予对应的状态
    valid_mask = idx >= 0
    matched_status[valid_mask] = trans_status[idx[valid_mask]]

    # 4. 组装结果并保存为 [N, 2] 格式
    result = np.column_stack((umi_ts, matched_status))
    
    np.savetxt(output_file, result, fmt='%.6f %d')

def unify_quaternion_sign(file_path):
    """
    读取位姿文件，统一所有四元数的实部 w 为正。
    如果 w < 0，则将整个四元数 (qx, qy, qz, qw) 取反。
    """
    try:
        # 加载数据 (timestamp, x, y, z, qx, qy, qz, qw)
        data = np.loadtxt(file_path)
        if data.size == 0:
            return
        
        # 假设 qw 是最后一列 (索引 7)
        # 提取四元数部分
        quats = data[:, 4:8]
        
        # 找到所有实部 qw < 0 的行
        # quats[:, 3] 对应 qw
        neg_indices = quats[:, 3] < 0
        
        # 对这些行进行取反操作 (q 和 -q 等价)
        quats[neg_indices] *= -1
        
        # 写回原数组
        data[:, 4:8] = quats
        
        # 覆盖保存
        np.savetxt(file_path, data, fmt='%.18f')
        print(f" 已统一四元数实部为正: {file_path}")
    except Exception as e:
        print(f" 统一四元数时发生错误: {e}")



def main():
    import pathlib

    base_dir = pathlib.Path("/home/k202/test")
    next_num = max([int(f.name) for f in base_dir.glob('*') if f.name.isdigit()] + [-1]) + 1
    base_dir = base_dir / f"{next_num:06d}"
    base_dir.mkdir(parents=True, exist_ok=True)

    os.makedirs(base_dir, exist_ok=True)
    dir_tac1 = os.path.join(base_dir, "tactile_left")
    dir_tac2 = os.path.join(base_dir, "tactile_right")
    dir_gopro = os.path.join(base_dir, "gopro")
    dir_realsense = os.path.join(base_dir, "realsense")


    for d in [dir_tac1, dir_tac2, dir_gopro, dir_realsense]:
        os.makedirs(d, exist_ok=True)

    print("正在初始化硬件与 ROS 节点...")
    
    tac_manager = GelSightManager()
    gopro_manager = GoproManager(device_id=10, width=224, height=224, fps=30)
    serial_manager = SerialManager(port='/dev/ttyUSB0', baud_rate=115200, save_dir=base_dir)
    pose_manager = PoseManager(save_dir=base_dir)
    # force_manager = ForceTorqueManager(topic_name="/force_sensor/wrench", save_dir=base_dir)
    rs_manager = RealsenseRosManager(topic_name="/camera/color/image_raw", save_dir=base_dir)
    
    target_fps = 15.0
    target_interval = 1.0 / target_fps
    
    print("初始化完成。开始并行记录数据。按 ESC 键退出。")

    try:
        while True:
            loop_start_time = time.time()

            tac_f1, tac_f2, tac_ts = tac_manager.get_tactile_frame()
            gopro_f, gopro_ts = gopro_manager.get_latest_frame()
            rs_f, rs_ts = rs_manager.get_latest_frame()

            
            if tac_f1 is not None and tac_f2 is not None and gopro_f is not None and rs_f is not None:
                unified_timestamp = time.time()
                filename = f"{unified_timestamp:.6f}.jpg"

                cv2.imwrite(os.path.join(dir_tac1, filename), tac_f1)
                cv2.imwrite(os.path.join(dir_tac2, filename), tac_f2)
                cv2.imwrite(os.path.join(dir_gopro, filename), gopro_f)
                cv2.imwrite(os.path.join(dir_realsense, filename), rs_f)

                cv2.imshow("Tactile Left", tac_f1)
                cv2.imshow("Tactile Right", tac_f2)
                cv2.imshow("GoPro", gopro_f)
                cv2.imshow("RealSense", rs_f)

            elif gopro_f is None:
                print("等待gopro相机数据同步中...", end="\r")
            elif rs_f is None:
                print("等待realsense相机数据同步中...", end="\r")


            elif tac_f1 is None:
                 print("等待触觉机数据同步中...", end="\r")

            elapsed = time.time() - loop_start_time
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            key = cv2.waitKey(1)
            if key == 27:
                break
            
    finally:
        print("\n检测到键盘中断。正在停止采集并释放硬件资源...")
        
        # 提前释放所有后台线程与硬件占用，确保系统有足够的内存和IO处理图像
        tac_manager.release()
        gopro_manager.release()
        serial_manager.release()
        pose_manager.release()
        # force_manager.release()
        cv2.destroyAllWindows()
        hardware_released_flag = True 
        
        print("硬件已释放，开始执行数据后处理...")

        try:
            umi_file_path = os.path.join(base_dir, 'umi_body_abs.txt')
            gripper_file_path = os.path.join(base_dir, 'gripper_state.txt')
            output_file_path = os.path.join(base_dir, 'gripper_state_time.txt')

            unify_quaternion_sign(umi_file_path) 
              
            print(" 正在执行夹爪状态与机械臂位姿匹配...")
            process_gripper_states(umi_file_path, gripper_file_path, output_file_path)
            
            print("数据后处理全部完成。")
            
        except Exception as e:
            print(f"数据后处理过程发生错误: {e}")

        # 如果程序非正常中断（非键盘引发），确保资源被释放
        if not locals().get('hardware_released_flag', False):
            print("\n执行异常清理，释放硬件资源...")
            tac_manager.release()
            gopro_manager.release()
            serial_manager.release()
            pose_manager.release()
            # force_manager.release()
            cv2.destroyAllWindows()
        print("程序彻底退出。")


if __name__ == "__main__":
    main()

#查看端口v4l2-ctl --list-devices