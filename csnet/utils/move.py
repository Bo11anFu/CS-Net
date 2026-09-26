import os
import shutil


BASE_DIR = '/home/ubuntu/data/zhengqinxu/object_detection/DATA/test_video_1028'

# 图像文件所在的源目录 (img/mask)
source_dir_name ='mask'
source_path = os.path.join(BASE_DIR, source_dir_name)

# 目标目录 (img/1)
target_dir_name = '1'
target_path = os.path.join(source_path, target_dir_name)

# --- 核心逻辑 ---

def move_files_to_subfolder(source_dir, target_dir):
    """
    遍历源文件夹下的所有文件，并将所有 .tiff 文件移动到目标文件夹。
    """
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")

    if not os.path.isdir(source_dir):
        print(f"错误：源目录 '{source_dir}' 不存在。请检查 BASE_DIR 配置是否正确。")
        return

    if not os.path.isdir(target_dir):
        print(f"目标目录 '{target_dir}' 不存在，正在创建...")
        try:
            os.makedirs(target_dir)
            print("创建成功。")
        except OSError as e:
            print(f"错误：创建目标目录失败：{e}")
            return

    moved_count = 0
    for item in os.listdir(source_dir):
        source_item_path = os.path.join(source_dir, item)

        # 确保只处理文件，并且忽略目标目录本身
        if os.path.isfile(source_item_path) and item.lower().endswith('.tiff'):
            # 构建目标路径
            target_item_path = os.path.join(target_dir, item)

            try:
                # 移动文件
                shutil.move(source_item_path, target_item_path)
                print(f"移动文件: {item}")
                moved_count += 1
            except Exception as e:
                print(f"错误：移动文件 {item} 失败：{e}")

    if moved_count > 0:
        print(f"\n操作完成。共移动了 {moved_count} 个 .tiff 文件到 '{target_dir}'。")
    else:
        print("\n操作完成。没有找到需要移动的 .tiff 文件。")

# 执行函数
move_files_to_subfolder(source_path, target_path)