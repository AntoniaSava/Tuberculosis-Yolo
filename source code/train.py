from ultralytics import YOLO
import os

def main():
    print(" Convertim măștile în segmentări YOLO...")
    os.system("python prepare_data.py")

    print(" Începem antrenarea YOLOv8...")

    model = YOLO("yolov8n-seg.pt")  #sa incerc si cu alte variante de YOLOv8
    model.train(
        data="yolo_config/tb.yaml",
        epochs=50,
        imgsz=512,
        project="tb_yolo_seg",
        name="exp2",
        exist_ok=True
    )

    print(" Antrenarea s-a încheiat.")


if __name__ == '__main__':
    main()
