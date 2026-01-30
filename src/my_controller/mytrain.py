from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    model = YOLO(model='F:/deeplearning/ultralytics-main/ultralytics-main/yolo26s-seg.pt')
 
    #model = model.load('F:/deeplearning/ultralytics-main/ultralytics-main/yolo26s-seg.pt')


    
    train_results = model.train(
        data="F:/deeplearning/ultralytics-main/ultralytics-main/roboflow/My First Project.v5i.yolo26/data.yaml",        
        epochs=200,                
        imgsz=640,                
        device="0",                
        workers=4,                 
        batch=16,
        augment=True,
        patience=40,
        #rect=True,
        amp=True,
        degrees=10.0,
        shear=5.0,
        perspective=0.001,
        cls=2.5,
        box=10,
        lr0=0.0005,
        cos_lr=True
        )                                                                       
         

    val_metrics = model.val()
# Windows多进程必须的入口保护
if __name__ == "__main__":
    freeze_support()
    main()
