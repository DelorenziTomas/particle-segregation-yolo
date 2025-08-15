from ultralytics import YOLO

model = YOLO('yolo11x.pt')

# Entrenamiento con carpeta personalizada
results = model.train(
    data='/home/tdelorenzi/1_yolo/1_segregacion/3_datasets/dataset_discospmma/data.yaml',
    epochs=300,
    imgsz=640,
    batch=4,
    project='/home/tdelorenzi/1_yolo/1_segregacion/4_modelos/1_discos-pmma',  # Ruta ABSOLUTA
    name='train',  # Nombre de la subcarpeta (obligatorio)
    exist_ok=True  # Evita errores si la carpeta existe
)


"""
tdelorenzi@rog:~/1_yolo$ source /home/tdelorenzi/1_yolo/env1/bin/activate
(env1) tdelorenzi@rog:~/1_yolo$ /home/tdelorenzi/1_yolo/env1/bin/python /home/tdelorenzi/1_yolo/1_segregacion/2_entrenamientos/train-discos-pmma.py
Ultralytics 8.3.170 🚀 Python-3.12.3 torch-2.7.1+cu128 CUDA:0 (NVIDIA GeForce RTX 4070 Laptop GPU, 7806MiB)
engine/trainer: agnostic_nms=False, amp=True, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=/home/tdelorenzi/1_yolo/1_segregacion/3_datasets/dataset_discospmma/data.yaml, degrees=0.0, deterministic=True, device=None, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, epochs=300, erasing=0.4, exist_ok=True, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolo11x.pt, momentum=0.937, mosaic=1.0, multi_scale=False, name=train, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=100, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=/home/tdelorenzi/1_yolo/1_segregacion/4_modelos/1_discos-pmma, rect=False, resume=False, retina_masks=False, save=True, save_conf=False, save_crop=False, save_dir=/home/tdelorenzi/1_yolo/1_segregacion/4_modelos/1_discos-pmma/train, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=1

                   from  n    params  module                                       arguments                     
  0                  -1  1      2784  ultralytics.nn.modules.conv.Conv             [3, 96, 3, 2]                 
  1                  -1  1    166272  ultralytics.nn.modules.conv.Conv             [96, 192, 3, 2]               
  2                  -1  2    389760  ultralytics.nn.modules.block.C3k2            [192, 384, 2, True, 0.25]     
  3                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
  4                  -1  2   1553664  ultralytics.nn.modules.block.C3k2            [384, 768, 2, True, 0.25]     
  5                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
  6                  -1  2   5022720  ultralytics.nn.modules.block.C3k2            [768, 768, 2, True]           
  7                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
  8                  -1  2   5022720  ultralytics.nn.modules.block.C3k2            [768, 768, 2, True]           
  9                  -1  1   1476864  ultralytics.nn.modules.block.SPPF            [768, 768, 5]                 
 10                  -1  2   3264768  ultralytics.nn.modules.block.C2PSA           [768, 768, 2]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  2   5612544  ultralytics.nn.modules.block.C3k2            [1536, 768, 2, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  2   1700352  ultralytics.nn.modules.block.C3k2            [1536, 384, 2, True]          
 17                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  2   5317632  ultralytics.nn.modules.block.C3k2            [1152, 768, 2, True]          
 20                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  2   5612544  ultralytics.nn.modules.block.C3k2            [1536, 768, 2, True]          
 23        [16, 19, 22]  1   3146707  ultralytics.nn.modules.head.Detect           [1, [384, 768, 768]]          
YOLO11x summary: 357 layers, 56,874,931 parameters, 56,874,915 gradients, 195.4 GFLOPs

Transferred 1009/1015 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 5650.4±1679.1 MB/s, size: 265.1 KB)
train: Scanning /home/tdelorenzi/1_yolo/1_segregacion/3_datasets/dataset_discospmma/train/labels... 30 images, 0 backgrounds, 0 corrupt: 100%|█
train: New cache created: /home/tdelorenzi/1_yolo/1_segregacion/3_datasets/dataset_discospmma/train/labels.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 4014.7±2789.5 MB/s, size: 263.9 KB)
val: Scanning /home/tdelorenzi/1_yolo/1_segregacion/3_datasets/dataset_discospmma/valid/labels... 5 images, 0 backgrounds, 0 corrupt: 100%|████
val: New cache created: /home/tdelorenzi/1_yolo/1_segregacion/3_datasets/dataset_discospmma/valid/labels.cache
Plotting labels to /home/tdelorenzi/1_yolo/1_segregacion/4_modelos/1_discos-pmma/train/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.002, momentum=0.9) with parameter groups 167 weight(decay=0.0), 174 weight(decay=0.0005), 173 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 8 dataloader workers
Logging results to /home/tdelorenzi/1_yolo/1_segregacion/4_modelos/1_discos-pmma/train
Starting training for 300 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/300      5.27G      1.954      2.548      1.345        242        640: 100%|██████████| 8/8 [00:02<00:00,  3.41it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  6.99it/s]
                   all          5        971      0.661      0.995      0.691      0.407

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/300      5.65G      1.317     0.8227      1.036        296        640: 100%|██████████| 8/8 [00:01<00:00,  4.77it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  9.19it/s]
                   all          5        971      0.257      0.396      0.209      0.122




.
.
.


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    300/300      5.15G     0.9991     0.3637     0.9042        377        640: 100%|██████████| 8/8 [00:01<00:00,  5.32it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  9.87it/s]
                   all          5        971      0.999      0.998      0.994      0.764

300 epochs completed in 0.221 hours.
Optimizer stripped from /home/tdelorenzi/1_yolo/1_segregacion/4_modelos/1_discos-pmma/train/weights/last.pt, 114.4MB
Optimizer stripped from /home/tdelorenzi/1_yolo/1_segregacion/4_modelos/1_discos-pmma/train/weights/best.pt, 114.4MB

Validating /home/tdelorenzi/1_yolo/1_segregacion/4_modelos/1_discos-pmma/train/weights/best.pt...
Ultralytics 8.3.170 🚀 Python-3.12.3 torch-2.7.1+cu128 CUDA:0 (NVIDIA GeForce RTX 4070 Laptop GPU, 7806MiB)
YOLO11x summary (fused): 190 layers, 56,828,179 parameters, 0 gradients, 194.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  9.31it/s]
                   all          5        971      0.999      0.998      0.994      0.777
Speed: 0.2ms preprocess, 15.1ms inference, 0.0ms loss, 0.7ms postprocess per image
Results saved to /home/tdelorenzi/1_yolo/1_segregacion/4_modelos/1_discos-pmma/train
(env1) tdelorenzi@rog:~/1_yolo$ 


"""