import cv2

#  video0 是主视频节点
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# 强制设置像素格式，解决部分 Linux 环境下的兼容性问题
# HD60 X 支持 MJPG, YUYV, NV12
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("错误：无法访问 /dev/video0。")
    print("请确认 OBS 是否已关闭，或尝试执行：sudo chmod 777 /dev/video0")
    exit()

print("成功连接 GoPro！按 'q' 键退出预览。")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Real-time Recording', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()