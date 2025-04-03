from ultralytics import YOLO


def main():
    # 加载预训练模型
    model = YOLO('yolov8n-seg.pt')
    # 训练模型
    results = model.train(
        data='coco8-seg.yaml',  # 替换为你的数据集配置文件路径
        epochs=100,  # 训练的轮数
        imgsz=640,  # 输入图像的大小
        batch=16,  # 批量大小
        device=0,  # 使用的 GPU 设备编号，如果使用 CPU 则设置为 'cpu'
        workers=16,  # 数据加载的工作线程数
        project='runs/segment',  # 训练结果保存的项目目录
        name='exp'  # 训练结果保存的实验名称
    )

    # 评估模型
    metrics = model.val()
    print(metrics)
if __name__ == '__main__':
    main()

