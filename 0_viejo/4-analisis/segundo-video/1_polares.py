from ultralytics import YOLO
import cv2
import os
import math
import pandas as pd
from collections import defaultdict

# 1. Cargar el modelo entrenado
model_path = 'runs/detect/train9/weights/best.pt'
model = YOLO(model_path)

# 2. Configuración de clases
class_names = ['disco blanco', 'estrella blanca']
class_colors = [(0, 255, 0), (0, 0, 255)]  # Verde para discos, Rojo para estrellas

# 3. Preguntar al usuario sobre el tipo de visualización
print("\nOpciones de visualización:")
print("1. Mostrar bounding boxes con etiquetas y confianza")
print("2. Mostrar solo bounding boxes (sin texto)")
visual_mode = input("Seleccione el modo de visualización (1/2): ").strip()
show_labels = visual_mode == '1'

# 4. Configurar rutas
video_path = '/home/tdelorenzi/testYolo/1-imagenesvideos/tambor_recortado_36s_rotado2.mp4'
output_dir = '/home/tdelorenzi/testYolo/2-resultados'
os.makedirs(output_dir, exist_ok=True)

# 5. Configurar video de entrada/salida
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error al abrir el video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = os.path.join(output_dir, 'resultado_video3.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 6. Configurar origen para coordenadas polares
origin_x, origin_y = 509, 513  # Coordenadas del origen en píxeles

# Estructura para almacenar los datos
tracking_data = defaultdict(list)
frame_count = 0

# Función para convertir a coordenadas polares
def to_polar(x, y, origin_x, origin_y):
    # Convertir coordenadas cartesianas relativas al origen
    dx = x - origin_x
    dy = origin_y - y  # Invertir eje Y para que crezca hacia arriba
    
    # Calcular radio (distancia desde el origen)
    radius = math.sqrt(dx**2 + dy**2)
    
    # Calcular ángulo en radianes y luego convertirlo a grados
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    # Ajustar ángulo para que esté en el rango [0, 360)
    angle_deg = angle_deg % 360
    
    return radius, angle_deg

# Función para dibujar anotaciones personalizadas y calcular coordenadas polares
def draw_custom_annotations(frame, results, show_text=True):
    annotated_frame = frame.copy()
    
    # Dibujar el origen como un punto de referencia
    cv2.circle(annotated_frame, (origin_x, origin_y), 5, (255, 0, 0), -1)
    
    for result in results:
        for box in result.boxes:
            # Obtener datos de la detección
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Calcular centro de la bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Calcular coordenadas polares
            radius, angle = to_polar(center_x, center_y, origin_x, origin_y)
            
            # Almacenar datos
            tracking_data[frame_count].append({
                'class': class_names[class_id],
                'confidence': confidence,
                'center_x': center_x,
                'center_y': center_y,
                'radius': radius,
                'angle': angle,
                'bbox': [x1, y1, x2, y2]
            })
            
            # Seleccionar color según la clase
            color = class_colors[class_id]
            
            # Dibujar bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Dibujar línea desde el origen al centro
            cv2.line(annotated_frame, (origin_x, origin_y), (center_x, center_y), (255, 255, 255), 1)
            
            if show_text:
                # Preparar texto de la etiqueta
                label = f"{class_names[class_id]} {confidence:.2f}"
                polar_info = f"r:{radius:.1f} θ:{angle:.1f}°"
                
                # Calcular tamaño del texto
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                
                # Dibujar fondo para el texto
                cv2.rectangle(
                    annotated_frame, 
                    (x1, y1 - text_height - 25), 
                    (x1 + text_width, y1), 
                    color, -1)
                
                # Dibujar texto
                cv2.putText(
                    annotated_frame, 
                    label, 
                    (x1, y1 - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    1, 
                    cv2.LINE_AA)
                
                cv2.putText(
                    annotated_frame, 
                    polar_info, 
                    (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    1, 
                    cv2.LINE_AA)
    
    return annotated_frame

# Crear ventana
cv2.namedWindow("Detección de Partículas", cv2.WINDOW_NORMAL)

# Procesar video
print("\nProcesando video... (Presione 'q' para detener)")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Realizar predicción
    results = model.predict(frame, conf=0.3740)
    
    # Dibujar anotaciones según el modo seleccionado
    annotated_frame = draw_custom_annotations(frame, results, show_labels)
    
    # Mostrar y guardar resultados
    cv2.imshow("Detección de Partículas", annotated_frame)
    out.write(annotated_frame)
    
    frame_count += 1
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\nProcesamiento completado. Video guardado en {output_path}")

# Guardar datos de coordenadas polares en un archivo CSV
polar_data = []
for frame_num, detections in tracking_data.items():
    for det in detections:
        polar_data.append({
            'frame': frame_num,
            'time_sec': frame_num / fps,
            'class': det['class'],
            'confidence': det['confidence'],
            'center_x': det['center_x'],
            'center_y': det['center_y'],
            'radius_px': det['radius'],
            'angle_deg': det['angle'],
            'bbox_x1': det['bbox'][0],
            'bbox_y1': det['bbox'][1],
            'bbox_x2': det['bbox'][2],
            'bbox_y2': det['bbox'][3]
        })

df = pd.DataFrame(polar_data)
csv_path = os.path.join(output_dir, 'polar_coordinates2.csv')
df.to_csv(csv_path, index=False)
print(f"Datos de coordenadas polares guardados en {csv_path}")