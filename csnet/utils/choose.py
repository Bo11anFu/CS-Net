import os
import shutil
from tqdm import tqdm

# ===================================================================
# 1. 配置您的路径
# ===================================================================

# --- 请修改以下三个路径 ---

# 1. 包含 5400 个重叠图块的源数据根目录
#    (即包含 'image' 和 'mask' 文件夹的那个目录)
SOURCE_DATA_DIR = r"/home/ubuntu/data/zhengqinxu/object_detection/DATA/test_video384"

# 2. 您希望保存筛选后数据的新文件夹的路径
#    (脚本会自动创建这个文件夹)
FILTERED_DATA_DIR = r"/home/ubuntu/data/zhengqinxu/object_detection/DATA/test_384tar"

# 3. 指向上一步生成的、包含目标文件名的文本文件路径
#    这个文件应该在您的 mask 目录中
TARGET_LIST_FILE = r"/home/ubuntu/data/zhengqinxu/object_detection/DATA/test_video384/mask/patches_with_targets.txt"

# ===================================================================

def create_filtered_dataset():


   
    if not os.path.isdir(SOURCE_DATA_DIR):
        print(f"错误：源数据目录不存在 -> '{SOURCE_DATA_DIR}'")
        return
    if not os.path.isfile(TARGET_LIST_FILE):
        print(f"错误：目标列表文件不存在 -> '{TARGET_LIST_FILE}'")
        print("请先运行 'check_patches.py' 脚本来生成这个文件。")
        return

    # --- 1. 读取含有目标的文件名列表 ---
    print(f"正在从 '{TARGET_LIST_FILE}' 读取目标列表...")
    with open(TARGET_LIST_FILE, 'r') as f:
        # 使用 line.strip() 来移除每行末尾的换行符
        target_filenames = [line.strip() for line in f if line.strip()]

    if not target_filenames:
        print("目标列表文件为空，没有需要复制的文件。程序退出。")
        return

    print(f"找到 {len(target_filenames)} 个含有目标的图块需要复制。")

    # --- 2. 创建输出目录结构 ---
    # 目录结构将与源数据保持一致
    dest_img_dir = os.path.join(FILTERED_DATA_DIR, "image", "1")
    dest_mask_dir = os.path.join(FILTERED_DATA_DIR, "mask", "1")
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_mask_dir, exist_ok=True)
    print(f"已创建输出目录: '{FILTERED_DATA_DIR}'")

    # --- 3. 循环复制文件 ---
    # 定义源文件的基本路径
    source_img_dir = os.path.join(SOURCE_DATA_DIR, "image", "1")
    source_mask_dir = os.path.join(SOURCE_DATA_DIR, "mask", "1")

    for filename in tqdm(target_filenames, desc="正在复制文件"):
        # 构建源文件和目标文件的完整路径
        source_img_path = os.path.join(source_img_dir, filename)
        source_mask_path = os.path.join(source_mask_dir, filename)
        
        dest_img_path = os.path.join(dest_img_dir, filename)
        dest_mask_path = os.path.join(dest_mask_dir, filename)

        # 检查源文件是否存在，以防万一
        if os.path.exists(source_img_path) and os.path.exists(source_mask_path):
            shutil.copy2(source_img_path, dest_img_path)
            shutil.copy2(source_mask_path, dest_mask_path)
        else:
            print(f"\n警告：找不到源文件 {filename}，已跳过。")
            
    print("\n筛选和复制完成！")
    print(f"所有 {len(target_filenames)} 个含有目标的图像/掩码对已成功复制。")

if __name__ == '__main__':
    create_filtered_dataset()