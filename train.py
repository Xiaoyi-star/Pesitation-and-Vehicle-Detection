from ultralytics import YOLO

if __name__ == '__main__':
    # 加载 YOLOv11 nano 预训练权重（首次运行自动下载，约 5MB）
    model = YOLO('yolo11n.pt')

    # 注意：data 填写 data.yaml 的完整路径
    # 如果 train.py 和数据集文件夹放在同一目录，可以用下面的相对路径：
    results = model.train(
        data="Person and Vehicle.v13i.yolov11\data.yaml",
        epochs=10,  # CPU 训练建议 10 轮，约 30-60 分钟
        imgsz=416,  # 降低分辨率以加快 CPU 训练速度
        batch=4,  # CPU 训练 batch 设小一点
        device='0',  # 无 GPU 则使用 CPU
        workers=2,
        project="D:\工程\人工智能课程\Detect_PersonVehicle",
        name='yolo11 _vp',
        val=True,
    )

    print("训练完成！权重保存于:", results.save_dir)