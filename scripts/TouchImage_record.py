import sys 

sys.path.append("/home/k202/gsmini_ws/src")

import gelSight_SDK.examples.gsdevice as gsdevice
import cv2
import time
import os

# 设置保存路径 
SAVE_DIR_CAM1 = "test_0318/tactile_left_5"
SAVE_DIR_CAM2 = "test_0318/tactile_right_5"

os.makedirs(SAVE_DIR_CAM1, exist_ok=True)
os.makedirs(SAVE_DIR_CAM2, exist_ok=True)

dev1 = gsdevice.Camera("GelSight Mini R0B 2DAT-2LMZ")
dev2 = gsdevice.Camera("GelSight Mini R0B 2DMA-NFYZ")

dev1.connect()
dev2.connect()

window_w, window_h = 600, 400

print(f"程序已启动，正在连续保存数据至: {SAVE_DIR_CAM1} 和 {SAVE_DIR_CAM2}")
print("按 ESC 键退出")

while True:
   
    f1 = dev1.get_raw_image() # f1.shape = (240,320,3)，uint8
    f2 = dev2.get_raw_image()

    
    # 获取精确时间戳并保存
    timestamp = time.time()
    filename = f"{timestamp:.6f}.jpg"
    
    if f1 is not None and f2 is not None:
        cv2.imwrite(os.path.join(SAVE_DIR_CAM1, filename), f1)
        cv2.imwrite(os.path.join(SAVE_DIR_CAM2, filename), f2)

        cv2.imshow("GelSight Mini R0B 2DAT-DPZE", f1)
        cv2.imshow("GelSight Mini R0B 2DAT-2LMZ", f2)

    if cv2.waitKey(1) & 0xFF == 27:
        print("用户按下ESC键, 退出")
        break

cv2.destroyAllWindows()

