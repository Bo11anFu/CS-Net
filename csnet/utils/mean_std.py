import os
import numpy as np
import tifffile
from tqdm import tqdm





FILTERED_IMAGE_DIR = r"/home/ubuntu/data/zhengqinxu/object_detection/DATA/test_video_1028_384/img/1"



# --- 参数 (通常无需修改) ---
# 您的数据是 12-bit，最大值是 2^12 - 1 = 4095
MAX_PIXEL_VALUE = 4095.0

# ===================================================================

def calculate_mean_std_for_normalization():
    """
    文档说明：
    此函数会读取一个文件名列表，遍历列表中指定的图像图块，
    计算这些图像在 [0.0, 1.0] 范围内的逐通道均值和标准差，
    用于后续的 torchvision.transforms.Normalize 操作。
    """

    
    target_filenames = []
    for filename in os.listdir(FILTERED_IMAGE_DIR):
        if filename.endswith('.tiff'):
            target_filenames.append(filename)

        if not target_filenames:
            print("目标列表文件没有找到.tiff文件，无法进行计算。")
            return

        print(f"将基于 {len(target_filenames)} 个含有目标的图像图块进行计算。")


    # (B, G, R) or (C1, C2, C3)
    channel_sum = np.zeros(3)
    channel_sum_squared = np.zeros(3)
    pixel_count = 0

    # --- 3. 循环遍历图像并累加 ---
    for filename in tqdm(target_filenames, desc="正在计算统计数据"):
        img_path = os.path.join(FILTERED_IMAGE_DIR, filename)

        # 读取 uint16 格式存储的 uint12 图像
        img_uint16 = tifffile.imread(img_path)

        # 转换为 float32 并缩放到 [0.0, 1.0] 范围
        img_scaled = img_uint16.astype(np.float32) / MAX_PIXEL_VALUE

        # 扩展为3个通道，以匹配模型输入的格式
        img_3channel = np.stack([img_scaled] * 3, axis=-1)
        
        # 累加像素总数
        h, w, c = img_3channel.shape
        pixel_count += h * w

        # 累加每个通道的像素值总和
        # axis=(0, 1) 表示在高度和宽度维度上求和
        channel_sum += np.sum(img_3channel, axis=(0, 1))
        
        # 累加每个通道的像素值平方的总和
        channel_sum_squared += np.sum(np.square(img_3channel), axis=(0, 1))

    # --- 4. 计算最终的均值和标准差 ---
    # E[X] = sum(X) / N
    mean = channel_sum / pixel_count
    
    # Var(X) = E[X^2] - (E[X])^2
    variance = (channel_sum_squared / pixel_count) - np.square(mean)
    
    # StdDev(X) = sqrt(Var(X))
    std = np.sqrt(variance)

    # --- 5. 打印结果 ---
    print("\n计算完成！")
    print("-" * 50)
    print(f"计算得到的均值 (Mean): {mean}")
    print(f"计算得到的标准差 (Std): {std}")
    print("-" * 50)
    # 使用 .6f 格式化以获得足够的小数位数
    print(f"transforms.Normalize(mean=[{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}], std=[{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}])")

if __name__ == '__main__':
    calculate_mean_std_for_normalization()