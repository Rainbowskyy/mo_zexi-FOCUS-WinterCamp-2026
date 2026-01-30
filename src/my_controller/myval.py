from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("F:/deeplearning/ultralytics-main/runs/segment/train28/weights/best.pt")
    model.val(split="test",conf=0.1)