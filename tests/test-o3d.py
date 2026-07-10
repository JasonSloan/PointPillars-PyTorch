import numpy as np
import open3d as o3d

bin_path = "/mnt/nfs_docker_volume/training-container-space/mnt/datasets/object-detection-datasets/open-source/KITTI/training/velodyne/000000.bin"

points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
xyz = points[:, :3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

o3d.visualization.draw_geometries([pcd])