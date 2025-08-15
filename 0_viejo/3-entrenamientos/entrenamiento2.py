from ultralytics import YOLO

# Crear nuevo modelo sin pesos preentrenados
model = YOLO('yolo11x.pt')  # Necesitarás el archivo de configuración YOLOv11

# Configurar entrenamiento
results = model.train(
    data='/home/tdelorenzi/testYolo/dataset2/data.yaml',
    epochs=300,  # Más épocas cuando se entrena desde cero
    imgsz=640,
    batch=4
)