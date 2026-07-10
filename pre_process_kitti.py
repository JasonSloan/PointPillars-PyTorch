import argparse
import pdb
import cv2
import numpy as np
import os
from tqdm import tqdm
import sys
CUR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR)

from utils import read_points, write_points, read_calib, read_label, \
    write_pickle, remove_outside_points, get_points_num_in_bbox, \
    points_in_bboxes_v2


def judge_difficulty(annotation_dict):
    truncated = annotation_dict['truncated']
    occluded = annotation_dict['occluded']
    bbox = annotation_dict['bbox']
    height = bbox[:, 3] - bbox[:, 1]

    MIN_HEIGHTS = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.30, 0.50]
    difficultys = []
    for h, o, t in zip(height, occluded, truncated):
        difficulty = -1
        for i in range(2, -1, -1):
            if h > MIN_HEIGHTS[i] and o <= MAX_OCCLUSION[i] and t <= MAX_TRUNCATION[i]:
                difficulty = i
        difficultys.append(difficulty)
    return np.array(difficultys, dtype=np.int32)


def create_data_info_pkl(data_root, data_type, prefix, label=True, db=False):
    """
    为了方便理解, 可以把cam0_raw当做世界坐标系, 例如一个LiDAR点云向cam2的图像投影, 需要经过以下步骤:
              Tr_velo_to_cam                   R0_rect                             P2
    LiDAR ---------------------> cam0_raw ---------------------> cam0_rect ---------------------> image_2
    为 KITTI 指定数据划分生成 info pkl, 并保存裁剪后的点云文件。

    这个函数会按 ImageSets/{data_type}.txt 中的样本 id 逐帧处理数据:
    1. 读取 image_2 图像, 记录图像尺寸和相对路径。
    2. 读取 calib 标定文件, 记录 P0~P3、R0_rect、Tr_velo_to_cam 等矩阵。
    3. 读取 velodyne 点云, 根据 image_2 的可见视锥过滤掉图像范围外的点,
       并将结果保存到 velodyne_reduced/{id}.bin。
    4. 如果 label=True, 读取 label_2 标注, 计算每个目标的 difficulty,
       统计每个 GT 3D 框内包含的点云数量, 并写入 annos 字段。
    5. 如果 db=True, 额外为每个有效 GT 目标保存局部点云, 生成用于数据增强的
       ground truth database 信息。

    Args:
        data_root (str): KITTI 数据集根目录, 目录下应包含 training/ 和 testing/。
        data_type (str): 数据划分名称, 例如 'train'、'val'、'test'。
            样本 id 优先从 {data_root}/ImageSets/{data_type}.txt 读取;
            如果该文件不存在, 则回退到当前代码仓库的 dataset/ImageSets/。
        prefix (str): 输出 pkl 文件名前缀, 例如 'kitti'。
        label (bool): 当前划分是否包含 label_2 标注。True 时读取 training/;
            False 时读取 testing/。
        db (bool): 是否生成 ground truth database。通常只在 train 划分中开启,
            用于后续训练阶段的数据增强。

    写出的文件:
        - {data_root}/{split}/velodyne_reduced/{id}.bin:
          当前样本过滤图像范围外点之后的点云。
        - {data_root}/{prefix}_infos_{data_type}.pkl:
          当前划分的样本信息字典。
        - {data_root}/{prefix}_gt_database/*.bin:
          db=True 时, 每个 GT 目标框内部的局部点云。
        - {data_root}/{prefix}_dbinfos_train.pkl:
          db=True 时, ground truth database 的索引信息。

    Returns:
        dict: 以 image id 为 key 的样本信息字典。每个 value 包含 image、calib,
        以及在 label=True 时包含 annos 标注信息。
    """
    sep = os.path.sep
    print(f"Processing {data_type} data..")
    split = 'training' if label else 'testing'
    ids_file = os.path.join(data_root, 'ImageSets', f'{data_type}.txt')
    if not os.path.exists(ids_file):
        ids_file = os.path.join(CUR, 'dataset', 'ImageSets', f'{data_type}.txt')
    with open(ids_file, 'r') as f:
        ids = [id.strip() for id in f.readlines()]

    kitti_infos_dict = {}
    if db:
        kitti_dbinfos_train = {}
        db_points_saved_path = os.path.join(data_root, f'{prefix}_gt_database')
        os.makedirs(db_points_saved_path, exist_ok=True)
    for id in tqdm(ids):
        cur_info_dict={}
        img_path = os.path.join(data_root, split, 'image_2', f'{id}.png')
        lidar_path = os.path.join(data_root, split, 'velodyne', f'{id}.bin')
        calib_path = os.path.join(data_root, split, 'calib', f'{id}.txt') 
        cur_info_dict['velodyne_path'] = sep.join(lidar_path.split(sep)[-3:])

        img = cv2.imread(img_path)
        image_shape = img.shape[:2]
        cur_info_dict['image'] = {
            'image_shape': image_shape,
            'image_path': sep.join(img_path.split(sep)[-3:]), 
            'image_idx': int(id),
        }

        # KITTI 这里每个样本都有一个 calib 文件，主要是为了数据使用方便，不代表每一帧都重新标定了一次, 实际只有4套标定数据
        calib_dict = read_calib(calib_path)
        cur_info_dict['calib'] = calib_dict

        # n*4, 4: [x, y, z, intensity]
        lidar_points = read_points(lidar_path)
        reduced_lidar_points = remove_outside_points(
            points=lidar_points, 
            r0_rect=calib_dict['R0_rect'], 
            tr_velo_to_cam=calib_dict['Tr_velo_to_cam'], 
            P2=calib_dict['P2'], 
            image_shape=image_shape)
        saved_reduced_path = os.path.join(data_root, split, 'velodyne_reduced')
        os.makedirs(saved_reduced_path, exist_ok=True)
        saved_reduced_points_name = os.path.join(saved_reduced_path, f'{id}.bin')
        write_points(reduced_lidar_points, saved_reduced_points_name)

        if label:
            label_path = os.path.join(data_root, split, 'label_2', f'{id}.txt')
            annotation_dict = read_label(label_path)
            # 通过框的高度 是否截断 是否遮挡来给没给框一个难度系数
            annotation_dict['difficulty'] = judge_difficulty(annotation_dict)
            annotation_dict['num_points_in_gt'] = get_points_num_in_bbox(
                points=reduced_lidar_points,
                r0_rect=calib_dict['R0_rect'], 
                tr_velo_to_cam=calib_dict['Tr_velo_to_cam'],
                dimensions=annotation_dict['dimensions'],   # 3d标出框的尺寸: [l, h, w]
                location=annotation_dict['location'],       # 目标 3D 框的中心点坐标
                rotation_y=annotation_dict['rotation_y'],   # 绕 y 轴的旋转角
                name=annotation_dict['name'])
            cur_info_dict['annos'] = annotation_dict

            if db:
                indices, n_total_bbox, n_valid_bbox, bboxes_lidar, name = \
                    points_in_bboxes_v2(
                        points=lidar_points,
                        r0_rect=calib_dict['R0_rect'].astype(np.float32), 
                        tr_velo_to_cam=calib_dict['Tr_velo_to_cam'].astype(np.float32),
                        dimensions=annotation_dict['dimensions'].astype(np.float32),
                        location=annotation_dict['location'].astype(np.float32),
                        rotation_y=annotation_dict['rotation_y'].astype(np.float32),
                        name=annotation_dict['name']    
                    )
                for j in range(n_valid_bbox):
                    db_points = lidar_points[indices[:, j]]
                    # 把当前 GT 框里的点云坐标，从全局 LiDAR 坐标系转换成以该 3D bbox 中心为原点的局部坐标系
                    db_points[:, :3] -= bboxes_lidar[j, :3]
                    db_points_saved_name = os.path.join(db_points_saved_path, f'{int(id)}_{name[j]}_{j}.bin')
                    write_points(db_points, db_points_saved_name)

                    db_info={
                        'name': name[j],
                        'path': os.path.join(os.path.basename(db_points_saved_path), f'{int(id)}_{name[j]}_{j}.bin'),
                        'box3d_lidar': bboxes_lidar[j],
                        'difficulty': annotation_dict['difficulty'][j], 
                        'num_points_in_gt': len(db_points), 
                    }
                    if name[j] not in kitti_dbinfos_train:
                        kitti_dbinfos_train[name[j]] = [db_info]
                    else:
                        kitti_dbinfos_train[name[j]].append(db_info)
        
        kitti_infos_dict[int(id)] = cur_info_dict

    saved_path = os.path.join(data_root, f'{prefix}_infos_{data_type}.pkl')
    write_pickle(kitti_infos_dict, saved_path)
    if db:
        saved_db_path = os.path.join(data_root, f'{prefix}_dbinfos_train.pkl')
        write_pickle(kitti_dbinfos_train, saved_db_path)
    return kitti_infos_dict


def main(args):
    data_root = args.data_root
    prefix = args.prefix

    ## 1. train: create data infomation pkl file && create reduced point clouds 
    ##           && create database(points in gt bbox) for data aumentation
    kitti_train_infos_dict = create_data_info_pkl(data_root, 'train', prefix, db=True)

    ## 2. val: create data infomation pkl file && create reduced point clouds
    kitti_val_infos_dict = create_data_info_pkl(data_root, 'val', prefix)
    
    ## 3. trainval: create data infomation pkl file
    kitti_trainval_infos_dict = {**kitti_train_infos_dict, **kitti_val_infos_dict}
    saved_path = os.path.join(data_root, f'{prefix}_infos_trainval.pkl')
    write_pickle(kitti_trainval_infos_dict, saved_path)

    ## 4. test: create data infomation pkl file && create reduced point clouds
    kitti_test_infos_dict = create_data_info_pkl(data_root, 'test', prefix, label=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset infomation')
    parser.add_argument('--data_root', default='/mnt/nfs_docker_volume/training-container-space/mnt/datasets/object-detection-datasets/open-source/KITTI', 
                        help='your data root for kitti')
    parser.add_argument('--prefix', default='kitti', 
                        help='the prefix name for the saved .pkl file')
    args = parser.parse_args()

    main(args)
