import argparse
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import pdb

from utils import setup_seed, read_points, read_calib, read_label, \
    keep_bbox_from_image_range, keep_bbox_from_lidar_range, \
    vis_img_3d, bbox3d2corners_camera, points_camera2image, \
    bbox_camera2lidar, bbox3d2corners
from model import PointPillars


LINES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [2, 6], [7, 3], [1, 5], [4, 0]
]
COLORS = ["red", "green", "blue", "orange"]


def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    return pts 


def save_pointcloud_image(pc, out_path, bboxes=None, labels=None, max_points=50000):
    """
    Save a static 3D point cloud visualization without opening an Open3D/cv2 window.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    points = pc
    if len(points) > max_points:
        sample_indices = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
        points = points[sample_indices]

    xyz = points[:, :3]
    intensity = points[:, 3] if points.shape[1] > 3 else np.ones((len(points),), dtype=np.float32)

    fig = plt.figure(figsize=(12, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        c=intensity, cmap="gray", s=0.15, linewidths=0
    )

    if bboxes is not None and len(bboxes) > 0:
        corners = bbox3d2corners(bboxes) if len(bboxes.shape) == 2 else bboxes
        for i, corner in enumerate(corners):
            label = labels[i] if labels is not None else -1
            color = COLORS[label] if label >= 0 and label < len(COLORS) else COLORS[-1]
            for start, end in LINES:
                xs = [corner[start, 0], corner[end, 0]]
                ys = [corner[start, 1], corner[end, 1]]
                zs = [corner[start, 2], corner[end, 2]]
                ax.plot(xs, ys, zs, color=color, linewidth=0.8)

    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    ax.set_zlabel("z up (m)")
    # 设置 3D 图的观察视角:
    # elev 是上下仰角, 表示从水平面往上抬多少度看; elev=90 接近 BEV 鸟瞰图, elev=0 接近平视图
    # azim 是绕竖直轴的水平旋转角, 表示从哪个水平方向斜着观察点云, azim=-72 表示观察相机在水平面上绕竖直轴旋转了 -72 度
    ax.view_init(elev=18, azim=-72)
    # 设置 x/y/z 三个方向的显示比例:
    # KITTI/PointPillars 常用点云范围大约是 x: 0~70m, y: -40~40m, z: -3~1m
    # x/y 方向跨度很大, z 方向跨度很小; 这里用 70:80:12 让显示效果更接近真实空间比例
    # 这个值只影响 3D 图中坐标轴的显示比例, 不会改变点云和检测框本身的坐标数值
    ax.set_box_aspect((70, 80, 12))
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved point cloud visualization to {out_path}")


def main(args):
    CLASSES = {
        'Pedestrian': 0, 
        'Cyclist': 1, 
        'Car': 2
        }
    LABEL2CLASSES = {v:k for k, v in CLASSES.items()}
    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)
    os.makedirs(args.saved_path, exist_ok=True)

    if not args.no_cuda:
        model = PointPillars(nclasses=len(CLASSES)).cuda()
        model.load_state_dict(torch.load(args.ckpt))
    else:
        model = PointPillars(nclasses=len(CLASSES))
        model.load_state_dict(
            torch.load(args.ckpt, map_location=torch.device('cpu')))
    model.score_thr = args.score_thr
    
    if not os.path.exists(args.pc_path):
        raise FileNotFoundError 
    pc = read_points(args.pc_path)
    pc = point_range_filter(pc)
    pc_torch = torch.from_numpy(pc)
    if os.path.exists(args.calib_path):
        calib_info = read_calib(args.calib_path)
    else:
        calib_info = None
    
    if os.path.exists(args.gt_path):
        gt_label = read_label(args.gt_path)
    else:
        gt_label = None

    if os.path.exists(args.img_path):
        img = cv2.imread(args.img_path, 1)
    else:
        img = None

    model.eval()
    with torch.no_grad():
        if not args.no_cuda:
            pc_torch = pc_torch.cuda()
        
        result_filter = model(batched_pts=[pc_torch], 
                              mode='test')[0]
    if calib_info is not None and img is not None:
        tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
        r0_rect = calib_info['R0_rect'].astype(np.float32)
        P2 = calib_info['P2'].astype(np.float32)

        image_shape = img.shape[:2]
        result_filter = keep_bbox_from_image_range(result_filter, tr_velo_to_cam, r0_rect, P2, image_shape)

    result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
    lidar_bboxes = result_filter['lidar_bboxes']
    labels, scores = result_filter['labels'], result_filter['scores']
    print(f"Kept {len(lidar_bboxes)} predicted boxes after filtering.")

    save_pointcloud_image(
        pc,
        os.path.join(args.saved_path, "pred_pointcloud.png"),
        bboxes=lidar_bboxes,
        labels=labels
    )

    if calib_info is not None and img is not None:
        bboxes2d, camera_bboxes = result_filter['bboxes2d'], result_filter['camera_bboxes'] 
        bboxes_corners = bbox3d2corners_camera(camera_bboxes)
        image_points = points_camera2image(bboxes_corners, P2)
        img = vis_img_3d(img, image_points, labels, rt=True)

    if calib_info is not None and gt_label is not None:
        tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
        r0_rect = calib_info['R0_rect'].astype(np.float32)

        dimensions = gt_label['dimensions']
        location = gt_label['location']
        rotation_y = gt_label['rotation_y']
        gt_labels = np.array([CLASSES.get(item, -1) for item in gt_label['name']])
        sel = gt_labels != -1
        gt_labels = gt_labels[sel]
        bboxes_camera = np.concatenate([location, dimensions, rotation_y[:, None]], axis=-1)
        gt_lidar_bboxes = bbox_camera2lidar(bboxes_camera, tr_velo_to_cam, r0_rect)
        bboxes_camera = bboxes_camera[sel]
        gt_lidar_bboxes = gt_lidar_bboxes[sel]

        gt_labels = [-1] * len(gt_lidar_bboxes) # to distinguish between the ground truth and the predictions
        print(f"Loaded {len(gt_lidar_bboxes)} valid GT boxes.")
        save_pointcloud_image(
            pc,
            os.path.join(args.saved_path, "gt_pointcloud.png"),
            bboxes=gt_lidar_bboxes,
            labels=gt_labels
        )
        
        pred_gt_lidar_bboxes = np.concatenate([lidar_bboxes, gt_lidar_bboxes], axis=0)
        pred_gt_labels = np.concatenate([labels, gt_labels])
        save_pointcloud_image(
            pc,
            os.path.join(args.saved_path, "pred_gt_pointcloud.png"),
            bboxes=pred_gt_lidar_bboxes,
            labels=pred_gt_labels
        )

        if img is not None:
            bboxes_corners = bbox3d2corners_camera(bboxes_camera)
            image_points = points_camera2image(bboxes_corners, P2)
            gt_labels = [-1] * len(bboxes_camera)
            img = vis_img_3d(img, image_points, gt_labels, rt=True)
    
    if calib_info is not None and img is not None:
        img_out_path = os.path.join(args.saved_path, "image_3d_bbox.png")
        cv2.imwrite(img_out_path, img)
        print(f"Saved image bbox visualization to {img_out_path}")
            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--ckpt', default='pretrained/epoch_160.pth', help='your checkpoint for kitti')
    parser.add_argument('--pc_path', default='/mnt/nfs_docker_volume/training-container-space/mnt/datasets/object-detection-datasets/open-source/KITTI/testing/velodyne/004289.bin',help='your point cloud path')
    parser.add_argument('--calib_path', default='/mnt/nfs_docker_volume/training-container-space/mnt/datasets/object-detection-datasets/open-source/KITTI/testing/calib/004289.txt', help='your calib file path')
    parser.add_argument('--gt_path', default='/mnt/nfs_docker_volume/training-container-space/mnt/datasets/object-detection-datasets/open-source/KITTI/testing/label_2/004289.txt', help='your ground truth path')
    parser.add_argument('--img_path', default='/mnt/nfs_docker_volume/training-container-space/mnt/datasets/object-detection-datasets/open-source/KITTI/testing/image_2/004289.png', help='your image path')
    parser.add_argument('--saved_path', default='results/test_vis', help='path to save visualization results')
    parser.add_argument('--score_thr', type=float, default=0.5, help='score threshold for predicted boxes')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)
