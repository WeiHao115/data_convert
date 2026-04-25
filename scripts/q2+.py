import numpy as np
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
        np.savetxt("./www.txt", data, fmt='%.18f')
        print(f" 已统一四元数实部为正: {file_path}")
    except Exception as e:
        print(f" 统一四元数时发生错误: {e}")


if __name__ == "__main__":
    file_path = "/home/k202/0421/000000/umi_body_abs.txt"
    unify_quaternion_sign(file_path)