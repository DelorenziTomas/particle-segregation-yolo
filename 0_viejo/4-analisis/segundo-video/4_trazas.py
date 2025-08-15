from ultralytics import YOLO
import cv2
import os
import numpy as np
from collections import defaultdict, deque
import math
from scipy.optimize import linear_sum_assignment

# 1. Cargar el modelo entrenado
model_path = 'runs/detect/train8/weights/best.pt'
model = YOLO(model_path)

# 2. Configuración de clases
class_names = ['disco blanco', 'estrella blanca']
class_colors = [(0, 255, 0), (0, 0, 255)]  # Verde para discos, Rojo para estrellas

# 3. Configuración del sistema de tracking mejorado
MAX_IDS = 30
MAX_DISTANCE = 10
MIN_CONFIDENCE = 0.33
MAX_MISSED_FRAMES = 90
TRAIL_LENGTH = 30  # Longitud máxima de la traza en píxeles

active_tracks = {}  # {track_id: {'class_id': int, 'positions': deque, 'last_seen': int}}
available_ids = deque(range(1, MAX_IDS + 1))
frame_counter = 0
next_id = MAX_IDS + 1

# Usamos maxlen=100 para el historial global, pero limitamos longitud real de dibujo a 30px más abajo
track_history = defaultdict(lambda: deque(maxlen=100))

# 4. Preguntar al usuario sobre el tipo de visualización
print("\nOpciones de visualización:")
print("1. Mostrar bounding boxes con etiquetas y confianza")
print("2. Mostrar solo bounding boxes (sin texto)")
visual_mode = input("Seleccione el modo de visualización (1/2): ").strip()
show_labels = visual_mode == '1'

# 5. Configurar rutas
video_path = '/home/tdelorenzi/testYolo/1-imagenesvideos/tambor_recortado_36s_rotado2.mp4'
output_dir = '/home/tdelorenzi/testYolo/2-resultados'
os.makedirs(output_dir, exist_ok=True)

# 6. Configurar video de entrada/salida
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error al abrir el video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = os.path.join(output_dir, 'resultado_video_tracked_mejorado.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def update_tracks(active_tracks, frame_counter):
    global available_ids
    # Limpiar tracks que llevan mucho tiempo sin verse
    to_remove = []
    for track_id, track_data in active_tracks.items():
        if frame_counter - track_data['last_seen'] > MAX_MISSED_FRAMES:
            to_remove.append(track_id)
    for track_id in to_remove:
        if track_id <= MAX_IDS:
            available_ids.append(track_id)
        del active_tracks[track_id]
    available_ids = deque(sorted(available_ids))

def draw_detections(frame, detections, show_text=True):
    annotated_frame = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection['xyxy']
        class_id = detection['class_id']
        track_id = detection['track_id']
        confidence = detection['confidence']
        color = class_colors[class_id]
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        if track_id in track_history:
            for i in range(1, len(track_history[track_id])):
                cv2.line(annotated_frame, 
                         track_history[track_id][i-1], 
                         track_history[track_id][i], 
                         color, 1)
        if show_text:
            label = f"ID:{track_id} {class_names[class_id]} {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(
                annotated_frame, 
                (x1, y1 - text_height - 10), 
                (x1 + text_width, y1), 
                color, -1)
            cv2.putText(
                annotated_frame, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                1, 
                cv2.LINE_AA)
        else:
            cv2.putText(
                annotated_frame, 
                f"ID:{track_id}", 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                1, 
                cv2.LINE_AA)
    return annotated_frame

def draw_traces_only(track_history, class_colors, class_names, width, height):
    """Dibuja las trazas de todas las partículas sobre fondo negro, longitud máxima de 30px."""
    trace_img = np.zeros((height, width, 3), dtype=np.uint8)
    for track_id, points in track_history.items():
        # Selecciona solo la porción final de la traza con longitud máxima de 30 píxeles
        if len(points) < 2:
            continue

        # Recorta la traza para que no supere 30px de longitud total (sumando distancias)
        trimmed_points = [points[-1]]
        acc_dist = 0
        for i in range(len(points)-2, -1, -1):
            d = euclidean_distance(points[i+1], points[i])
            if acc_dist + d > TRAIL_LENGTH:
                # Agrega el punto de corte proporcional para llegar justo a TRAIL_LENGTH
                if d != 0:
                    ratio = (TRAIL_LENGTH - acc_dist) / d
                    x = int(points[i+1][0] + ratio * (points[i][0] - points[i+1][0]))
                    y = int(points[i+1][1] + ratio * (points[i][1] - points[i+1][1]))
                    trimmed_points.append((x, y))
                break
            acc_dist += d
            trimmed_points.append(points[i])
        trimmed_points = trimmed_points[::-1]  # Devolver al orden original

        # Obtener color según clase
        class_id = None
        for tid, data in active_tracks.items():
            if tid == track_id:
                class_id = data['class_id']
                break
        color = class_colors[class_id] if class_id is not None else (255, 255, 255)

        for i in range(1, len(trimmed_points)):
            cv2.line(trace_img, trimmed_points[i-1], trimmed_points[i], color, 2)
        # Marca el último punto con un círculo e ID muy pequeño
        x, y = trimmed_points[-1]
        cv2.circle(trace_img, (x, y), 4, color, -1)
        cv2.putText(trace_img, f"{track_id}", (x+4, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.32, color, 1, cv2.LINE_AA)  # TAMANO DE TEXTO MUY PEQUEÑO
    return trace_img

# --- Configura aquí el centro del tambor para coordenadas polares ---
xc, yc = width // 2, height // 2  # Ajusta estos valores si es necesario

# Crear ventana
cv2.namedWindow("Seguimiento de Partículas + Trazas", cv2.WINDOW_NORMAL)

print("\nProcesando video con seguimiento mejorado... (Presione 'q' para detener)")
frame_counter = 0

# Para guardar posiciones y ángulos en cada frame (opcional)
polar_tracking_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_counter += 1

    # Realizar predicción SIN tracking
    results = model.predict(frame, conf=MIN_CONFIDENCE)
    # Procesar detecciones
    current_detections = []
    for result in results:
        for box in result.boxes:
            if box.conf[0] < MIN_CONFIDENCE:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            detection_data = {
                'xyxy': (x1, y1, x2, y2),
                'class_id': class_id,
                'confidence': confidence,
                'center': center
            }
            current_detections.append(detection_data)

    # --- Tracking robusto por clase usando algoritmo húngaro ---
    used_detection_indices = set()
    for class_id in range(len(class_names)):
        # Tracks activos de esta clase
        tracks_of_class = {tid: t for tid, t in active_tracks.items() if t['class_id'] == class_id}
        detections_of_class = [i for i, d in enumerate(current_detections) if d['class_id'] == class_id]
        if not tracks_of_class or not detections_of_class:
            continue
        tracks_ids = list(tracks_of_class.keys())
        track_centers = [tracks_of_class[tid]['positions'][-1] for tid in tracks_ids]
        det_centers = [current_detections[i]['center'] for i in detections_of_class]
        cost_matrix = np.zeros((len(track_centers), len(det_centers)), dtype=np.float32)
        for i, tc in enumerate(track_centers):
            for j, dc in enumerate(det_centers):
                cost_matrix[i, j] = euclidean_distance(tc, dc)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_detections = set()
        assigned_tracks = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < MAX_DISTANCE:
                idx_det = detections_of_class[c]
                tid = tracks_ids[r]
                current_detections[idx_det]['track_id'] = tid
                active_tracks[tid]['positions'].append(current_detections[idx_det]['center'])
                active_tracks[tid]['last_seen'] = frame_counter
                track_history[tid].append(current_detections[idx_det]['center'])
                assigned_detections.add(idx_det)
                assigned_tracks.add(tid)
                used_detection_indices.add(idx_det)
        # Para detecciones no asignadas, crear nuevo track
        for i in detections_of_class:
            if i not in assigned_detections:
                if available_ids:
                    new_id = available_ids.popleft()
                else:
                    new_id = next_id
                    next_id += 1
                current_detections[i]['track_id'] = new_id
                active_tracks[new_id] = {
                    'class_id': current_detections[i]['class_id'],
                    'positions': deque([current_detections[i]['center']], maxlen=50),
                    'last_seen': frame_counter
                }
                track_history[new_id].append(current_detections[i]['center'])
                used_detection_indices.add(i)
    # Si quedara alguna detección sin asignar (de clases nuevas)
    for i, detection in enumerate(current_detections):
        if 'track_id' not in detection:
            if available_ids:
                new_id = available_ids.popleft()
            else:
                new_id = next_id
                next_id += 1
            detection['track_id'] = new_id
            active_tracks[new_id] = {
                'class_id': detection['class_id'],
                'positions': deque([detection['center']], maxlen=50),
                'last_seen': frame_counter
            }
            track_history[new_id].append(detection['center'])

    # Actualizar tracks y liberar IDs inactivos
    update_tracks(active_tracks, frame_counter)

    # Dibuja video original con bounding boxes e info
    annotated_frame = draw_detections(frame, current_detections, show_labels)
    cv2.putText(annotated_frame, f"Frame: {frame_counter}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"Tracks activos: {len(active_tracks)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Dibuja las trazas sobre fondo negro
    trace_img = draw_traces_only(track_history, class_colors, class_names, width, height)

    # Unir ambas imágenes horizontalmente
    combined = np.hstack((annotated_frame, trace_img))

    # Muestra la ventana combinada
    cv2.imshow("Seguimiento de Partículas + Trazas", combined)
    out.write(annotated_frame)  # Si solo quieres guardar el video original con bounding boxes

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # --- Post-proceso: guarda posiciones polares por frame ---
    frame_data = []
    for d in current_detections:
        x, y = d['center']
        r = np.sqrt((x - xc)**2 + (y - yc)**2)
        theta = np.arctan2(y - yc, x - xc)
        frame_data.append({
            "frame": frame_counter,
            "track_id": d['track_id'],
            "class_id": d['class_id'],
            "x": x,
            "y": y,
            "r": r,
            "theta": theta
        })
    polar_tracking_data.append(frame_data)

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\nProcesamiento con seguimiento mejorado completado. Video guardado en {output_path}")

# --- Guardar datos de tracking en archivo CSV (opcional) ---
import csv
csv_path = os.path.join(output_dir, 'tracking_polares.csv')
with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = ['frame', 'track_id', 'class_id', 'x', 'y', 'r', 'theta']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for frame_data in polar_tracking_data:
        for row in frame_data:
            writer.writerow(row)
print(f"Tracking exportado a {csv_path}")