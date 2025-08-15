from ultralytics import YOLO
import cv2
import numpy as np
import os
import math
import csv

# ------------------- CONFIGURACIÓN ----------------------
model_path = 'runs/detect/train8/weights/best.pt'
video_path = '/home/tdelorenzi/testYolo/1-imagenesvideos/tambor_recortado_36s_rotado2.mp4'
output_dir = '/home/tdelorenzi/testYolo/2-resultados'
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, 'detecciones_polares.csv')

class_names = ['disco blanco', 'estrella blanca']
heatmap_colors = [(0, 255, 0), (0, 0, 255)]  

# Parámetro de desvanecimiento: cuanto menor, más rápido se apaga el rastro
fade_factor = 0.96  # Prueba con 0.95 para rastros más cortos

# Carga modelo YOLO
model = YOLO(model_path)

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
xc, yc = width // 2, height // 2

detecciones_polares = []

# Heatmap por especie (float para acumulación y fade)
heatmaps = [np.zeros((height, width), dtype=np.float32) for _ in class_names]

cv2.namedWindow("Video + Heatmap partículas", cv2.WINDOW_NORMAL)

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # Desvanecer (fade-out) el heatmap de cada especie
    for hm in heatmaps:
        hm *= fade_factor

    results = model.predict(frame, conf=0.23)
    frame_data = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            r = np.sqrt((cx - xc)**2 + (cy - yc)**2)
            theta = np.arctan2(cy - yc, cx - xc)
            frame_data.append({
                "frame": frame_idx,
                "class_id": class_id,
                "especie": class_names[class_id],
                "confianza": conf,
                "x": cx,
                "y": cy,
                "r": float(r),
                "theta": float(theta)
            })
            # Visualización: círculo y texto
            cv2.circle(frame, (cx, cy), 10, heatmap_colors[class_id], 2)
            cv2.putText(frame, class_names[class_id], (cx+8, cy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, heatmap_colors[class_id], 1, cv2.LINE_AA)
            # Sumar al heatmap de la especie (con kernel para rastro ancho)
            if 0 <= cy < height and 0 <= cx < width:
                cv2.circle(heatmaps[class_id], (cx, cy), 16, 1, -1)  # Relleno circular, radio 12 px

    detecciones_polares.extend(frame_data)

    # Normalización para visualización (independiente por canal)
    vis_heatmap = np.zeros((height, width, 3), dtype=np.uint8)
    for i, (hm, color) in enumerate(zip(heatmaps, heatmap_colors)):
        if np.max(hm) > 0:
            norm_hm = (hm / np.max(hm))
            for c in range(3):
                vis_heatmap[:, :, c] += (norm_hm * color[c]).astype(np.uint8)

    # Suavizado para efecto difuso (opcional)
    vis_heatmap = cv2.GaussianBlur(vis_heatmap, (0,0), sigmaX=12, sigmaY=12)

    # Panel derecho: cuadrado negro + heatmap
    heatmap_panel = np.zeros_like(frame)
    mask = (np.sum(vis_heatmap, axis=2) > 0)
    heatmap_panel[mask] = vis_heatmap[mask]

    # Unir video (izquierda) + heatmap (derecha)
    combined = np.hstack((frame, heatmap_panel))
    cv2.imshow("Video + Heatmap partículas", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Guardar CSV
with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = ['frame', 'class_id', 'especie', 'confianza', 'x', 'y', 'r', 'theta']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in detecciones_polares:
        writer.writerow(row)
print(f"Exportado a {csv_path}")