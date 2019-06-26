"""
从old_root 中随机移动 n张到new_root下
"""
import shutil
import os
import random

test_path = './data/train'  # old_root
test_fold = os.listdir(test_path)  #[001, 002, 003 ]
for sub_folder in test_fold:
    file_root = os.path.join(test_path, sub_folder)
    file_name = os.listdir(file_root)
    choice_file = random.sample(file_name, 1000)  # 将1000张移动到新文件夹下
    for i in choice_file:
        root = os.path.join(test_path, sub_folder, i)
        new_root = './data/test'  # new_root
        new_folder = os.path.join(new_root, sub_folder)
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
        root_2 = os.path.join(new_folder, i)
        shutil.move(root, root_2)
        print('comprss:{}'.format(sub_folder))