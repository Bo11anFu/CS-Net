import os
import time


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def make_dir(root):
    if not os.path.exists(root):
        os.makedirs(root)


def make_exp_root(root):
    exp_root = os.path.join(root, get_time_str())
    make_dir(exp_root)
    return exp_root


def rename_filename_in_real_dataset(root):
    for seq in os.listdir(root):
        seq_dir = os.path.join(root, seq)
        for filename in os.listdir(seq_dir):
            decompose = filename.split('.')
            new_filename = f"{int(decompose[0].split('(')[1].split(')')[0]):05d}.{decompose[1]}"
            os.rename(os.path.join(seq_dir, filename), os.path.join(seq_dir, new_filename))


if __name__ == '__main__':
    root = '/media/estar/Data1/gr/CSNet/data/IRDST/real/boxes'
    rename_filename_in_real_dataset(root)
