```python
import os
import random
import shutil

# 原始路径
data_dir = r"D:\Personal\Documents\GitHub\YoloV8_MAH\data"
images_dir = os.path.join(data_dir, "training_images")
labels_dir = os.path.join(data_dir, "labels")

# 新的 YOLO 目录结构
yolo_images = os.path.join(data_dir, "images")
yolo_labels = os.path.join(data_dir, "labels_split")
for split in ["train", "val"]:
    os.makedirs(os.path.join(yolo_images, split), exist_ok=True)
    os.makedirs(os.path.join(yolo_labels, split), exist_ok=True)

# 所有图片
all_images = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
random.shuffle(all_images)

# 8:2 划分
train_size = int(0.8 * len(all_images))
train_images = all_images[:train_size]
val_images = all_images[train_size:]

def move_files(image_list, split):
    for img_file in image_list:
        # 图片路径
        src_img = os.path.join(images_dir, img_file)
        dst_img = os.path.join(yolo_images, split, img_file)
        shutil.copy(src_img, dst_img)

        # 标签路径
        label_file = img_file.replace(".jpg", ".txt")
        src_label = os.path.join(labels_dir, label_file)
        dst_label = os.path.join(yolo_labels, split, label_file)
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)
        else:
            # 没标签 → 自动补一个空文件
            with open(dst_label, "w") as f:
                pass

# 移动文件
move_files(train_images, "train")
move_files(val_images, "val")

print(f"✅ 数据整理完成！训练集: {len(train_images)} 张, 验证集: {len(val_images)} 张")

```

    ✅ 数据整理完成！训练集: 800 张, 验证集: 201 张
    


```python
import os

data_dir = r"D:\Personal\Documents\GitHub\YoloV8_MAH\data"

# 路径
train_images = os.path.join(data_dir, "images", "train")
val_images = os.path.join(data_dir, "images", "val")
train_labels = os.path.join(data_dir, "labels_split", "train")
val_labels = os.path.join(data_dir, "labels_split", "val")
test_images = os.path.join(data_dir, "testing_images")

# 函数：统计图片和标签
def count_files(img_dir, label_dir=None):
    imgs = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
    if label_dir:
        labels = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
        return len(imgs), len(labels)
    return len(imgs)

# 统计
train_img_count, train_label_count = count_files(train_images, train_labels)
val_img_count, val_label_count = count_files(val_images, val_labels)
test_img_count = count_files(test_images)

# 打印结果
print("📊 数据集统计结果：")
print(f"训练集: {train_img_count} 张图片, {train_label_count} 个标签文件")
print(f"验证集: {val_img_count} 张图片, {val_label_count} 个标签文件")
print(f"测试集: {test_img_count} 张图片 (无标签)")
```

    📊 数据集统计结果：
    训练集: 1001 张图片, 958 个标签文件
    验证集: 684 张图片, 359 个标签文件
    测试集: 175 张图片 (无标签)
    


```python
import os

yaml_content = """train: D:/Personal/Documents/GitHub/YoloV8_MAH/data/images/train
val: D:/Personal/Documents/GitHub/YoloV8_MAH/data/images/val

nc: 1
names: ['car']
"""

yaml_path = r"D:\Personal\Documents\GitHub\YoloV8_MAH\data\car.yaml"

with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"✅ car.yaml 已生成: {yaml_path}")

```

    ✅ car.yaml 已生成: D:\Personal\Documents\GitHub\YoloV8_MAH\data\car.yaml
    


```python
from ultralytics import YOLO
import shutil
import os

# 加载预训练 YOLOv8 模型
model = YOLO("yolov8s.pt")

# 开始训练
results = model.train(
    data="D:/Personal/Documents/GitHub/YoloV8_MAH/data/car.yaml",
    epochs=50,     
    imgsz=640,     
    batch=16,      
    name="car_exp1"  
)

# 训练完成后，保存 best.pt 到指定路径
best_model_path = "runs/train/car_exp1/weights/best.pt"
save_path = "D:/Personal/Documents/GitHub/YoloV8_MAH/data/best_car_model.pt"

if os.path.exists(best_model_path):
    shutil.copy(best_model_path, save_path)
    print(f"✅ 最优模型已额外保存到: {save_path}")
else:
    print("⚠️ 没找到 best.pt，请确认训练是否完成。")

```

    Ultralytics 8.3.204  Python-3.13.5 torch-2.8.0+cpu CPU (AMD Ryzen 9 7845HX with Radeon Graphics)
    [34m[1mengine\trainer: [0magnostic_nms=False, amp=True, augment=False, auto_augment=randaugment, batch=16, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=D:\Personal\Documents\GitHub\YoloV8_MAH\data\car.yaml, degrees=0.0, deterministic=True, device=cpu, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, epochs=50, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolov8s.pt, momentum=0.937, mosaic=1.0, multi_scale=False, name=car_exp16, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=100, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=None, rect=False, resume=False, retina_masks=False, save=True, save_conf=False, save_crop=False, save_dir=D:\Personal\Documents\GitHub\YoloV8_MAH\runs\detect\car_exp16, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
    Overriding model.yaml nc=80 with nc=1
    
                       from  n    params  module                                       arguments                     
      0                  -1  1       928  ultralytics.nn.modules.conv.Conv             [3, 32, 3, 2]                 
      1                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
      2                  -1  1     29056  ultralytics.nn.modules.block.C2f             [64, 64, 1, True]             
      3                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
      4                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           
      5                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
      6                  -1  2    788480  ultralytics.nn.modules.block.C2f             [256, 256, 2, True]           
      7                  -1  1   1180672  ultralytics.nn.modules.conv.Conv             [256, 512, 3, 2]              
      8                  -1  1   1838080  ultralytics.nn.modules.block.C2f             [512, 512, 1, True]           
      9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
     10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
     11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     12                  -1  1    591360  ultralytics.nn.modules.block.C2f             [768, 256, 1]                 
     13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
     14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     15                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
     16                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
     17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     18                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
     19                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
     20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     21                  -1  1   1969152  ultralytics.nn.modules.block.C2f             [768, 512, 1]                 
     22        [15, 18, 21]  1   2116435  ultralytics.nn.modules.head.Detect           [1, [128, 256, 512]]          
    Model summary: 129 layers, 11,135,987 parameters, 11,135,971 gradients, 28.6 GFLOPs
    
    Transferred 349/355 items from pretrained weights
    Freezing layer 'model.22.dfl.conv.weight'
    [34m[1mtrain: [0mFast image access  (ping: 0.00.0 ms, read: 22.37.9 MB/s, size: 104.6 KB)
    [K[34m[1mtrain: [0mScanning D:\Personal\Documents\GitHub\YoloV8_MAH\data\labels\train... 343 images, 658 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 1001/1001 1.4Kit/s 0.7s.1ss
    [34m[1mtrain: [0mNew cache created: D:\Personal\Documents\GitHub\YoloV8_MAH\data\labels\train.cache
    [34m[1mval: [0mFast image access  (ping: 0.00.0 ms, read: 19.32.6 MB/s, size: 92.3 KB)
    [K[34m[1mval: [0mScanning D:\Personal\Documents\GitHub\YoloV8_MAH\data\labels\val... 130 images, 554 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 684/684 1.5Kit/s 0.4s0.0s
    [34m[1mval: [0mNew cache created: D:\Personal\Documents\GitHub\YoloV8_MAH\data\labels\val.cache
    Plotting labels to D:\Personal\Documents\GitHub\YoloV8_MAH\runs\detect\car_exp16\labels.jpg... 
    [34m[1moptimizer:[0m 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
    [34m[1moptimizer:[0m AdamW(lr=0.002, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)
    Image sizes 640 train, 640 val
    Using 0 dataloader workers
    Logging results to [1mD:\Personal\Documents\GitHub\YoloV8_MAH\runs\detect\car_exp16[0m
    Starting training for 50 epochs...
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K       1/50         0G       1.81      8.782      1.225         15        640: 3% ──────────── 2/63 0.1it/s 31.7s<18:44
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[4], line 9
          6 model = YOLO("yolov8s.pt")
          8 # 开始训练
    ----> 9 results = model.train(
         10     data="D:/Personal/Documents/GitHub/YoloV8_MAH/data/car.yaml",
         11     epochs=50,     
         12     imgsz=640,     
         13     batch=16,      
         14     name="car_exp1"  
         15 )
         17 # 训练完成后，保存 best.pt 到指定路径
         18 best_model_path = "runs/train/car_exp1/weights/best.pt"
    

    File D:\Python\Anaconda\Lib\site-packages\ultralytics\engine\model.py:800, in Model.train(self, trainer, **kwargs)
        797     self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
        798     self.model = self.trainer.model
    --> 800 self.trainer.train()
        801 # Update model and cfg after training
        802 if RANK in {-1, 0}:
    

    File D:\Python\Anaconda\Lib\site-packages\ultralytics\engine\trainer.py:235, in BaseTrainer.train(self)
        232         ddp_cleanup(self, str(file))
        234 else:
    --> 235     self._do_train()
    

    File D:\Python\Anaconda\Lib\site-packages\ultralytics\engine\trainer.py:428, in BaseTrainer._do_train(self)
        423     self.tloss = (
        424         (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
        425     )
        427 # Backward
    --> 428 self.scaler.scale(self.loss).backward()
        430 # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
        431 if ni - last_opt_step >= self.accumulate:
    

    File D:\Python\Anaconda\Lib\site-packages\torch\_tensor.py:647, in Tensor.backward(self, gradient, retain_graph, create_graph, inputs)
        637 if has_torch_function_unary(self):
        638     return handle_torch_function(
        639         Tensor.backward,
        640         (self,),
       (...)
        645         inputs=inputs,
        646     )
    --> 647 torch.autograd.backward(
        648     self, gradient, retain_graph, create_graph, inputs=inputs
        649 )
    

    File D:\Python\Anaconda\Lib\site-packages\torch\autograd\__init__.py:354, in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
        349     retain_graph = create_graph
        351 # The reason we repeat the same comment below is that
        352 # some Python versions print out the first line of a multi-line function
        353 # calls in the traceback and some print out the last line
    --> 354 _engine_run_backward(
        355     tensors,
        356     grad_tensors_,
        357     retain_graph,
        358     create_graph,
        359     inputs_tuple,
        360     allow_unreachable=True,
        361     accumulate_grad=True,
        362 )
    

    File D:\Python\Anaconda\Lib\site-packages\torch\autograd\graph.py:829, in _engine_run_backward(t_outputs, *args, **kwargs)
        827     unregister_hooks = _register_logging_hooks_on_whole_graph(t_outputs)
        828 try:
    --> 829     return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
        830         t_outputs, *args, **kwargs
        831     )  # Calls into the C++ engine to run the backward pass
        832 finally:
        833     if attach_logging_hooks:
    

    KeyboardInterrupt: 



```python

```
