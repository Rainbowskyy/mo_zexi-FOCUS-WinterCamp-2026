from ultralytics import YOLO
import cv2
import numpy as np

# 配置参数
seg_model_path = "F:/deeplearning/ultralytics-main/runs/segment/train20/weights/best.pt"
det_model_path = "F:/deeplearning/ultralytics-main/yolo26n.pt"
BASE_THRESHOLD = 250  
BLIND_EXPAND = 80  
CONF_THRESHOLD = 0.5
INPUT_PATH = "F:/机器学习/TP-Dataset/JPEGImages/Part02/0109.jpg"
OUTPUT_PATH = "F:/deeplearning/ultralytics-main/output/result.jpg"

# 筛选函数
def is_obstacle_near_blind_sidewalk(obstacle_box, blind_masks, base_threshold, blind_expand, img_h, img_w):
    obs_x1, obs_y1, obs_x2, obs_y2 = obstacle_box
    obs_cx = (obs_x1 + obs_x2) / 2
    obs_cy = (obs_y1 + obs_y2) / 2

    for mask in blind_masks:
        mask = mask.astype(np.uint8)
        # 缩放掩码到原图尺寸
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        kernel = np.ones((blind_expand//2, blind_expand//2), np.uint8)
        mask_expanded = cv2.dilate(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(mask_expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            blind_x1, blind_y1, blind_w, blind_h = cv2.boundingRect(cnt)
            blind_x1 = max(0, blind_x1 - blind_expand)
            blind_y1 = max(0, blind_y1 - blind_expand)
            blind_x2 = min(img_w, blind_x1 + blind_w + blind_expand)
            blind_y2 = min(img_h, blind_y1 + blind_h + blind_expand)
            blind_cx = (blind_x1 + blind_x2) / 2
            blind_cy = (blind_y1 + blind_y2) / 2

            inter_x1 = max(obs_x1, blind_x1)
            inter_y1 = max(obs_y1, blind_y1)
            inter_x2 = min(obs_x2, blind_x2)
            inter_y2 = min(obs_y2, blind_y2)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                return True
            distance = np.sqrt((obs_cx - blind_cx)**2 + (obs_cy - blind_cy)**2)
            if distance < base_threshold:
                return True
    return False

# 盲道回归默认显示
def draw_blind_sidewalk(img, seg_results):
    """用模型默认的plot()方法绘制盲道，还原原始显示效果"""
    img_with_blind = seg_results[0].plot(img=img, conf= True )
    return img_with_blind


seg_model = YOLO(seg_model_path)
det_model = YOLO(det_model_path)


img = cv2.imread(INPUT_PATH)
img_h, img_w = img.shape[:2]
print(f"图片分辨率：{img_w}*{img_h} → 建议阈值：{img_w//8}（当前：{BASE_THRESHOLD})")


seg_results = seg_model(img, imgsz=640, conf=CONF_THRESHOLD)
blind_masks = np.array([])
if seg_results[0].masks is not None:
    blind_masks = seg_results[0].masks.data.cpu().numpy()
    print(f"检测到{len(blind_masks)}个盲道区域")


img_vis = draw_blind_sidewalk(img, seg_results)


det_results = det_model(img, imgsz=640, conf=CONF_THRESHOLD)
det_boxes = det_results[0].boxes
print(f"检测到{len(det_boxes)}个障碍物")


for box in det_boxes:
    cls_id = int(box.cls[0])
    conf = box.conf[0].item()
    obstacle_classes = [ 
    0,  # person（行人）
    1,  # bicycle（自行车）
    2,  # car（汽车）
    3,  # motorcycle（摩托车）
    5,  # bus（公交车）
    6,  # train（火车）
    7,  # truck（卡车）
    9,  # traffic light（红绿灯）
    10, # fire hydrant（消防栓）
    11, # stop sign（停车标志）
    13, # bench（长椅）
    14, # bird（鸟类）
    15, # cat（猫）
    16, # dog（狗）
    ]
    if cls_id not in obstacle_classes or conf < CONF_THRESHOLD:
        continue
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls_name = det_model.names[cls_id]

    if blind_masks.size == 0 or is_obstacle_near_blind_sidewalk(
        (x1,y1,x2,y2), blind_masks, BASE_THRESHOLD, BLIND_EXPAND, img_h, img_w):
        # 绘制代码（红色框+标签）
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"{cls_name} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1, 2)
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
        cv2.rectangle(img_vis, (x1, label_y - label_size[1] - 12), 
                  (x1 + label_size[0] + 12, label_y + 12), (0, 0, 255), -1)
        cv2.putText(img_vis, label, (x1 + 2, label_y), cv2.FONT_HERSHEY_DUPLEX, 
                1, (0,0,0), 2)

# 保存+显示
cv2.imwrite(OUTPUT_PATH, img_vis)
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.imshow("Result", img_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(f"结果已保存至：{OUTPUT_PATH}")