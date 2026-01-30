import os
import shutil
from pathlib import Path

# 两个原数据集的路径
dataset1_path = Path("F:/deeplearning/ultralytics-main/ultralytics-main/备份/My First Project.v3i.yolo26")  
dataset2_path = Path("F:/deeplearning/ultralytics-main/ultralytics-main/roboflow/My First Project.v4i.yolo26")  
# 合并后的数据集路径
merged_path = Path("F:/deeplearning/ultralytics-main/ultralytics-main/roboflow/merge2")
# 类别配置
nc = 2  
names = ['broken', 'tactile paving'] 

def copy_files(src_dir, dst_dir, subset):
    
    # 创建目标目录
    img_dst = dst_dir / subset / "images"
    lbl_dst = dst_dir / subset / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    # 遍历第一个数据集的文件
    for img_file in (src_dir / subset / "images").glob("*.*"):
        dst_img = img_dst / img_file.name
        if dst_img.exists():
            dst_img = img_dst / f"{img_file.stem}_1{img_file.suffix}"
        shutil.copy(img_file, dst_img)
        lbl_file = src_dir / subset / "labels" / f"{img_file.stem}.txt"
        if lbl_file.exists():
            dst_lbl = lbl_dst / f"{dst_img.stem}.txt"
            shutil.copy(lbl_file, dst_lbl)


copy_files(dataset1_path, merged_path, "train")
copy_files(dataset1_path, merged_path, "valid")
copy_files(dataset1_path, merged_path, "test")

def copy_files_dataset2(src_dir, dst_dir, subset):
    img_dst = dst_dir / subset / "images"
    lbl_dst = dst_dir / subset / "labels"
    for img_file in (src_dir / subset / "images").glob("*.*"):
        dst_img = img_dst / img_file.name
        if dst_img.exists():
            dst_img = img_dst / f"{img_file.stem}_2{img_file.suffix}"
        shutil.copy(img_file, dst_img)
        lbl_file = src_dir / subset / "labels" / f"{img_file.stem}.txt"
        if lbl_file.exists():
            dst_lbl = lbl_dst / f"{dst_img.stem}.txt"
            shutil.copy(lbl_file, dst_lbl)

copy_files_dataset2(dataset2_path, merged_path, "train")
copy_files_dataset2(dataset2_path, merged_path, "valid")
copy_files_dataset2(dataset2_path, merged_path, "test")


yaml_content = f"""path: {merged_path.absolute()}
train: train/images
val: valid/images
test: test/images
nc: {nc}
names: {names}
"""
with open(merged_path / "data.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"数据集合并完成！合并后路径：{merged_path.absolute()}")
print(f"检查：合并后的data.yaml已生成，类别数={nc}，类别名={names}")