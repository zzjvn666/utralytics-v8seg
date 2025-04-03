import cv2
import numpy as np
import os
from glob import glob
from skimage.morphology import skeletonize
from scipy.spatial import KDTree
from ultralytics import YOLO

# 设定像素范围
RANGE_THRESHOLD = 30

# 加载模型
model = YOLO('best.pt')

# 读取所有图片路径
input_folder = r'seg'
output_folder = r'processed'
os.makedirs(output_folder, exist_ok=True)

image_paths = sorted(glob(os.path.join(input_folder, '*.jpg')), key=lambda x: int(os.path.basename(x).split('.')[0]))

for img_path in image_paths:
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"无法读取 {img_path}，跳过。")
        continue

    # 复制一份用于绘制
    frame_with_clusters = frame.copy()

    # 转换为RGB格式
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 进行推理
    results = model(frame_rgb, conf=0.3, iou=0.4)

    # 生成掩码
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for r in results:
        masks = r.masks
        if masks is not None:
            for segments in masks.xy:
                segments = segments.astype(np.int32)
                segments = segments.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [segments], color=255)

    # 提取骨架
    skeleton = skeletonize(mask // 255).astype(np.uint8) * 255

    # 获取骨架上的点
    points = np.argwhere(skeleton == 255)
    if len(points) == 0:
        print(f"{img_path}: 未找到骨架点，跳过。")
        continue

    # 构建KDTree用于快速查找最近点
    tree = KDTree(points)

    # 记录已分配的点
    taken = set()
    clusters = []
    cluster_points_list = []  # 用于存储不同组的坐标

    # 选择起始点（最接近图像边界的点）
    def find_nearest_boundary_point(points, shape):
        distances = [min(x, y, shape[1] - x, shape[0] - y) for y, x in points]
        return points[np.argmin(distances)]

    while len(taken) < len(points):
        # 找到新的起始点
        remaining_points = [p for p in points if tuple(p) not in taken]
        start_point = find_nearest_boundary_point(remaining_points, frame.shape)
        cluster = [start_point]
        taken.add(tuple(start_point))

        while True:
            # 找到范围内的最近点
            distances, indices = tree.query(cluster[-1], k=len(points), distance_upper_bound=RANGE_THRESHOLD)
            found = False
            for d, idx in zip(distances, indices):
                if d < RANGE_THRESHOLD and tuple(points[idx]) not in taken:
                    cluster.append(points[idx])
                    taken.add(tuple(points[idx]))
                    found = True
                    break
            if not found:
                break

        clusters.append(cluster)
        cluster_points_list.append([tuple(p[::-1]) for p in cluster])  # 存储该组的坐标，并转换为(x, y)格式

    # 颜色列表
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    # 画出分组点
    for i, cluster in enumerate(clusters):
        color = colors[i % len(colors)]  # 轮流使用颜色
        for point in cluster:
            y, x = point
            cv2.circle(frame_with_clusters, (x, y), 2, color, -1)

    # 打印不同组的点坐标
    print(f"处理完成: {img_path}")
    for i, cluster_points in enumerate(cluster_points_list):
        print(f"Cluster {i+1}: {cluster_points}")

    # 保存处理后的图片
    output_path = os.path.join(output_folder, os.path.basename(img_path))
    cv2.imwrite(output_path, frame_with_clusters)

print("所有图片处理完成，结果已保存到:", output_folder)
