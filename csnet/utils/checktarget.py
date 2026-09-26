import os
import numpy as np
import tifffile
from glob import glob
from tqdm import tqdm




TARGET_MASK_DIR = r"/home/ubuntu/data/zhengqinxu/object_detection/DATA/test_video384/mask/1" 

# ===================================================================

def find_patches_with_targets():
  
    
    # 检查路径是否存在
    if not os.path.isdir(TARGET_MASK_DIR):
        print(f"错误：目录不存在 -> '{TARGET_MASK_DIR}'")
        print("请确保 TARGET_MASK_DIR 路径正确，并指向包含 .tiff 文件的具体序列文件夹。")
        return

    # 获取所有 mask 图块的路径
    mask_paths = sorted(glob(os.path.join(TARGET_MASK_DIR, '*.tiff')))
    
    if not mask_paths:
        print(f"错误：在 '{TARGET_MASK_DIR}' 目录中没有找到 .tiff 文件。")
        return
        
    print(f"开始检查 {len(mask_paths)} 个 mask 图块...")
    
    patches_with_targets = []
    
    # 循环检查每一个 mask 图块
    for mask_path in tqdm(mask_paths, desc="正在分析 Mask 图块"):
        # 读取 mask 图块
        mask_patch = tifffile.imread(mask_path)
        
        # 核心逻辑：检查是否存在任何一个像素值大于 0
        if np.any(mask_patch > 0):
            # 如果存在，将该图块的文件名（不含路径）添加到列表中
            file_name = os.path.basename(mask_path)
            patches_with_targets.append(file_name)

    # --- 打印和保存结果 ---
    print("\n检查完成！")
    num_total = len(mask_paths)
    num_with_targets = len(patches_with_targets)
    
    print(f"  总图块数量: {num_total}")
    print(f"  含有目标的图块数量: {num_with_targets}")
    print(f"  不含目标的图块数量: {num_total - num_with_targets}")
    
    if num_with_targets > 0:
        # 定义输出文件名
        output_file_path = os.path.join(os.path.dirname(TARGET_MASK_DIR), 'patches_with_targets.txt')
        
        # 将含有目标的文件名列表写入文件，每行一个
        with open(output_file_path, 'w') as f:
            for name in patches_with_targets:
                f.write(name + '\n')
        
        print(f"\n含有目标的文件名列表已保存到: '{output_file_path}'")


if __name__ == '__main__':
    find_patches_with_targets()