import os
import cv2
import numpy as np


def convert_masks_to_binary(root_dir):
    """
    遍历root_dir下的所有序列文件夹，将mask文件夹中的语义分割掩码转换为二值掩码(0/255)
    并保存到同级目录的mask_binary文件夹中
    """
    # 遍历每个序列文件夹
    for seq_name in os.listdir(root_dir):
        seq_path = os.path.join(root_dir, seq_name)

        if not os.path.isdir(seq_path):
            continue  # 跳过非文件夹

        mask_dir = os.path.join(seq_path, "mask")
        output_dir = os.path.join(seq_path, "mask_binary")

        # 检查mask文件夹是否存在
        if not os.path.exists(mask_dir):
            print(f"⚠️ 跳过 {seq_name}: 未找到mask文件夹")
            continue

        # 创建输出目录 (如果不存在)
        os.makedirs(output_dir, exist_ok=True)
        processed_count = 0

        # 处理所有PNG文件
        for mask_file in os.listdir(mask_dir):
            if mask_file.lower().endswith('.png'):
                input_path = os.path.join(mask_dir, mask_file)
                output_path = os.path.join(output_dir, mask_file)

                # 读取图像 (灰度模式)
                mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

                if mask is None:
                    print(f"⚠️ 无法读取: {input_path}")
                    continue

                # 创建二值掩码: 所有>0的值设为255
                binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)

                # 保存结果
                cv2.imwrite(output_path, binary_mask)
                processed_count += 1

        print(f"✅ 完成 {seq_name}: 转换了 {processed_count} 张掩码")


if __name__ == "__main__":
    print("=" * 50)
    print("开始转换语义分割掩码 -> 二值掩码(0/255)")
    print("=" * 50)

    root_dir = r'/home/ubuntu/data/zhengqinxu/object_detection/DATA/IRAir/val'
    convert_masks_to_binary(root_dir)

    print("=" * 50)
    print("所有序列处理完成！")
    print("=" * 50)