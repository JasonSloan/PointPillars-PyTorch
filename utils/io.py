import numpy as np
import os
import pickle


def read_pickle(file_path, suffix='.pkl'):
    assert os.path.splitext(file_path)[1] == suffix
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def write_pickle(results, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)


def read_points(file_path, dim=4):
    suffix = os.path.splitext(file_path)[1] 
    assert suffix in ['.bin', '.ply']
    if suffix == '.bin':
        return np.fromfile(file_path, dtype=np.float32).reshape(-1, dim)
    else:
        raise NotImplementedError


def write_points(lidar_points, file_path):
    suffix = os.path.splitext(file_path)[1] 
    assert suffix in ['.bin', '.ply']
    if suffix == '.bin':
        with open(file_path, 'w') as f:
            lidar_points.tofile(f)
    else:
        raise NotImplementedError


def read_calib(file_path, extend_matrix=True):
    """https://zhuanlan.zhihu.com/p/364423582"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    """
    P0~P3是校正后的相机投影矩阵, 不是单纯的相机内参矩阵。
    投影矩阵的形式为 K [R | t], 第4列编码了相机之间的平移项。

    KITTI通常以cam0对应的cam0_rect坐标系作为参考坐标系。
    因此, 当一个3D点已经被转换到该参考坐标系后, 再投影到cam2(image_2)
    图像上时, 需要使用完整的P2。由于cam2相对于cam0存在安装偏移,
    P2中的平移项不为0。

    为了方便理解, 可以把cam0_raw当做世界坐标系, 例如一个LiDAR点云向cam2的图像投影, 需要经过以下步骤:
              Tr_velo_to_cam                   R0_rect                             P2
    LiDAR ---------------------> cam0_raw ---------------------> cam0_rect ---------------------> image_2
    """
    # 0号灰度相机/左灰度相机的校正后投影矩阵, 用于 cam0_rect -> image_0 像素坐标
    P0 = np.array([item for item in lines[0].split(' ')[1:]], dtype=np.float32).reshape(3, 4)
    # 1号灰度相机/右灰度相机的校正后投影矩阵, 用于 cam0_rect -> image_1 像素坐标
    P1 = np.array([item for item in lines[1].split(' ')[1:]], dtype=np.float32).reshape(3, 4)
    # 2号彩色相机/左彩色相机的校正后投影矩阵, 用于 cam0_rect -> image_2 像素坐标
    P2 = np.array([item for item in lines[2].split(' ')[1:]], dtype=np.float32).reshape(3, 4)
    # 3号彩色相机/右彩色相机的校正后投影矩阵, 用于 cam0_rect -> image_3 像素坐标
    P3 = np.array([item for item in lines[3].split(' ')[1:]], dtype=np.float32).reshape(3, 4)

    # 原始 cam0 坐标系 -> 校正后的 cam0 坐标系的旋转校正矩阵, 用于 cam0_raw -> cam0_rect
    R0_rect = np.array([item for item in lines[4].split(' ')[1:]], dtype=np.float32).reshape(3, 3)
    # Velodyne LiDAR 坐标系 -> 原始 cam0 坐标系的刚体变换矩阵, 用于 velodyne -> cam0_raw
    Tr_velo_to_cam = np.array([item for item in lines[5].split(' ')[1:]], dtype=np.float32).reshape(3, 4)
    # IMU 坐标系 -> Velodyne LiDAR 坐标系的刚体变换矩阵, 用于 imu -> velodyne
    Tr_imu_to_velo = np.array([item for item in lines[6].split(' ')[1:]], dtype=np.float32).reshape(3, 4)

    if extend_matrix:
        P0 = np.concatenate([P0, np.array([[0, 0, 0, 1]])], axis=0)
        P1 = np.concatenate([P1, np.array([[0, 0, 0, 1]])], axis=0)
        P2 = np.concatenate([P2, np.array([[0, 0, 0, 1]])], axis=0)
        P3 = np.concatenate([P3, np.array([[0, 0, 0, 1]])], axis=0)

        R0_rect_extend = np.eye(4, dtype=R0_rect.dtype)
        R0_rect_extend[:3, :3] = R0_rect
        R0_rect = R0_rect_extend

        Tr_velo_to_cam = np.concatenate([Tr_velo_to_cam, np.array([[0, 0, 0, 1]])], axis=0)
        Tr_imu_to_velo = np.concatenate([Tr_imu_to_velo, np.array([[0, 0, 0, 1]])], axis=0)

    calib_dict=dict(
        P0=P0,
        P1=P1,
        P2=P2,
        P3=P3,
        R0_rect=R0_rect,
        Tr_velo_to_cam=Tr_velo_to_cam,
        Tr_imu_to_velo=Tr_imu_to_velo
    )
    return calib_dict


def read_label(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split(' ') for line in lines]
    annotation = {}
    annotation['name'] = np.array([line[0] for line in lines])
    annotation['truncated'] = np.array([line[1] for line in lines], dtype=np.float32)
    annotation['occluded'] = np.array([line[2] for line in lines], dtype=np.int32)
    annotation['alpha'] = np.array([line[3] for line in lines], dtype=np.float32)
    annotation['bbox'] = np.array([line[4:8] for line in lines], dtype=np.float32)
    annotation['dimensions'] = np.array([line[8:11] for line in lines], dtype=np.float32)[:, [2, 0, 1]] # hwl -> camera coordinates (lhw)
    annotation['location'] = np.array([line[11:14] for line in lines], dtype=np.float32)
    annotation['rotation_y'] = np.array([line[14] for line in lines], dtype=np.float32)
    
    return annotation


def write_label(result, file_path, suffix='.txt'):
    '''
    result: dict,
    file_path: str
    '''
    assert os.path.splitext(file_path)[1] == suffix
    name, truncated, occluded, alpha, bbox, dimensions, location, rotation_y, score = \
        result['name'], result['truncated'], result['occluded'], result['alpha'], \
        result['bbox'], result['dimensions'], result['location'], result['rotation_y'], \
        result['score']
    
    with open(file_path, 'w') as f:
        for i in range(len(name)):
            bbox_str = ' '.join(map(str, bbox[i]))
            hwl = ' '.join(map(str, dimensions[i]))
            xyz = ' '.join(map(str, location[i]))
            line = f'{name[i]} {truncated[i]} {occluded[i]} {alpha[i]} {bbox_str} {hwl} {xyz} {rotation_y[i]} {score[i]}\n'
            f.writelines(line)
