# import os
# import shutil
#
# # single picture
# root_ = r'E:\datasets\focuslight_results_x1.5'
# folder_new = r'E:\datasets\focuslight_results_x1.5-single'
#
# dirs_ = os.listdir(root_)
# dirs_.sort()
#
# for dir_ in dirs_:
#     folder_ = os.path.join(root_, dir_)
#     files = os.listdir(folder_)
#     files.sort(reverse=True)
#     if not os.path.exists(folder_new):
#         os.makedirs(folder_new)
#     old_name = os.path.join(root_, dir_, files[0])
#     new_name = os.path.join(folder_new, str(int(dir_) - 1).zfill(3) + '_' + files[0])
#
#     shutil.copy(old_name, new_name)


import os
import argparse


def main(args):
    # to list.txt
    root_ = args.folder_dir

    files = []
    for root, dirs, file in os.walk(root_):
        for f in file:
            files.append(os.path.join(root, f))

    f_ = open(args.txt_dir, 'w')
    for file in files:
        line = file + '\n'
        f_.write(line)
    f_.close()

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--folder_dir', type=str, default=r'E:\datasets\san_FaceID-alig')
    parse.add_argument('--txt_dir', type=str, default='data_list/san_FaceID-alig.txt', help='work root is arcface_torch')

    args = parse.parse_args()
    main(args)

