from ultralytics import YOLO

model = YOLO('yolo11x.pt')

# Entrenamiento con carpeta personalizada
results = model.train(
    data='/home/tdelorenzi/1_yolo/1_segregacion/3_datasets/dataset_pmmaazul/data.yaml',
    epochs=300,
    imgsz=640,
    batch=4,
    project='/home/tdelorenzi/1_yolo/1_segregacion/4_modelos/3_discospmmaazul',  # Ruta ABSOLUTA
    name='train',  # Nombre de la subcarpeta (obligatorio)
    exist_ok=True  # Evita errores si la carpeta existe
)


"""



"""