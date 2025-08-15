from ultralytics import YOLO
import cv2
import numpy as np
import os
import csv

# ------------------- CONFIGURACIÓN ----------------------
model_path = 'runs/detect/train9/weights/best.pt'
video_path = '/home/tdelorenzi/testYolo/1-imagenesvideos/tambor_recortado_36s_rotado2.mp4'
output_dir = '/home/tdelorenzi/testYolo/2-resultados'
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, 'detecciones_polares2.csv')
output_video_path = os.path.join(output_dir, 'heatmap_output2.mp4')

class_names = ['disco blanco', 'estrella blanca']

# ---- PEDIR AL USUARIO QUE SELECCIONE UNA ESPECIE ----
print("\nEspecies disponibles:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

selected_class = None
while selected_class is None:
    try:
        user_input = int(input("Ingrese el número de la especie a visualizar: "))
        if 0 <= user_input < len(class_names):
            selected_class = user_input
        else:
            print(f"Error: Ingrese un número entre 0 y {len(class_names)-1}")
    except ValueError:
        print("Error: Debe ingresar un número entero válido")

print(f"\nVisualizando mapa de calor para: {class_names[selected_class]}\n")

# Cargar modelo YOLO
model = YOLO(model_path)

# Abrir video
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

# Propiedades del video
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Inicializar video writer
video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# Configurar ventana redimensionable
cv2.namedWindow("Mapa de Calor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mapa de Calor", 1200, 800)  # Tamaño inicial grande

# Calcular tamaño del buffer (2 segundos)
buffer_size = int(2 * fps)
heatmap_buffer = [np.zeros((h, w), dtype=np.float32) for _ in range(buffer_size)]
buffer_idx = 0

# Variables para coordenadas polares
xc, yc = w // 2, h // 2
detecciones_polares = []

frame_idx = 0
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Procesamiento de video completado")
        break

    frame_idx += 1
    frame_heat = np.zeros((h, w), dtype=np.float32)  # Heatmap para este frame

    # Ejecutar detección
    results = model.predict(im0, verbose=False, conf=0.34)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Registrar datos para CSV
            r = np.sqrt((cx - xc)**2 + (cy - yc)**2)
            theta = np.arctan2(cy - yc, cx - xc)
            detecciones_polares.append({
                "frame": frame_idx,
                "class_id": class_id,
                "especie": class_names[class_id],
                "confianza": conf,
                "x": cx,
                "y": cy,
                "r": float(r),
                "theta": float(theta)
            })

            # Solo agregar al heatmap si es la especie seleccionada
            if class_id == selected_class:
                cv2.circle(frame_heat, (cx, cy), 20, 1, -1)  # Radio 20 píxeles

    # Actualizar buffer circular
    heatmap_buffer[buffer_idx] = frame_heat
    buffer_idx = (buffer_idx + 1) % buffer_size

    # Sumar heatmaps del buffer
    heatmap_sum = np.sum(heatmap_buffer, axis=0)
    heatmap_norm = cv2.normalize(heatmap_sum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    # Mezclar con el frame original (70% heatmap, 30% fondo)
    blended = cv2.addWeighted(heatmap_color, 0.7, im0, 0.3, 0)

    video_writer.write(blended)
    
    # Mostrar ventana maximizable
    cv2.imshow("Mapa de Calor", blended)
    
    # Tecla para salir (ESC o Q)
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

# Guardar CSV
with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = ['frame', 'class_id', 'especie', 'confianza', 'x', 'y', 'r', 'theta']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(detecciones_polares)

print(f"Video guardado en: {output_video_path}")
print(f"Datos guardados en: {csv_path}")