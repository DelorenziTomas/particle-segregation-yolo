from ultralytics import YOLO
import cv2
import os
import math
import numpy as np

# 1. Cargar el modelo entrenado
model_path = 'runs/detect/train9/weights/best.pt'
model = YOLO(model_path)

# 2. Configuración de clases y conteo total conocido
class_names = ['disco blanco', 'estrella blanca']
class_colors = [(0, 255, 0), (0, 0, 255)]  # Verde para discos, Rojo para estrellas

# Valores conocidos del total de partículas
total_discos = 124
total_estrellas = 138
x_o = total_estrellas / (total_discos + total_estrellas)  # Fracción inicial de estrellas

# 3. Configuración de visualización
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
output_path = os.path.join(output_dir, 'resultado_con_anillos_visibles.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Configuración de regiones/anillos para cálculo de segregación
num_regions = 9  # 9 anillos concéntricos como en el paper
center_x, center_y = width // 2, height // 2
max_radius = min(width, height) // 2

# Colores para los anillos (alternando para mejor visibilidad)
ring_colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
    (192, 192, 192), (128, 128, 128), (64, 64, 64)
]

# Variables para acumular datos de segregación
total_frames = 0
x_i_accum = np.zeros(num_regions)  # Para acumular x_i por región
region_counts = np.zeros((num_regions, 2))  # Contador de [discos, estrellas] por región

# Función para calcular el índice de segregación (SI)
def calculate_segregation_index(x_i, x_o):
    N = len(x_i)
    mean_xi_xo = np.mean(x_i / x_o)
    SI = np.sqrt(np.sum(((x_i / x_o - mean_xi_xo) ** 2) / N))
    return SI

# Función para dibujar anillos concéntricos
def draw_concentric_rings(frame):
    annotated_frame = frame.copy()
    for i in range(num_regions, 0, -1):  # Dibujar de afuera hacia adentro
        radius = int(i * (max_radius / num_regions))
        color = ring_colors[i-1]
        thickness = 2 if i % 2 == 0 else 1  # Grosor alternado para mejor visibilidad
        cv2.circle(annotated_frame, (center_x, center_y), radius, color, thickness)
        
        # Etiquetar las regiones
        if i == num_regions:
            label_pos = (center_x + radius - 30, center_y - 10)
            cv2.putText(annotated_frame, f"Region {i}", label_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return annotated_frame

# Función principal de procesamiento
def process_frame(frame, results, show_text=True):
    annotated_frame = draw_concentric_rings(frame)
    current_region_counts = np.zeros((num_regions, 2))
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            center_x_p = (x1 + x2) // 2
            center_y_p = (y1 + y2) // 2
            distance = math.sqrt((center_x_p - center_x)**2 + (center_y_p - center_y)**2)
            
            region = min(int(distance // (max_radius / num_regions)), num_regions - 1)
            current_region_counts[region, class_id] += 1
            
            # Dibujar bounding box
            color = class_colors[class_id]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            if show_text:
                label = f"{class_names[class_id]} {confidence:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - text_height - 10), 
                             (x1 + text_width, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    
    # Calcular métricas
    total_particles = current_region_counts.sum(axis=1)
    x_i = np.where(total_particles > 0, 
                  current_region_counts[:,1] / total_particles, 
                  0)
    
    global region_counts, total_frames, x_i_accum
    region_counts += current_region_counts
    x_i_accum += x_i
    total_frames += 1
    
    # Calcular SI para este frame
    valid_regions = total_particles > 0
    if np.any(valid_regions):
        current_SI = calculate_segregation_index(x_i[valid_regions], x_o)
        
        # Mostrar información
        info_text = [
            f"Estrellas: {int(current_region_counts[:,1].sum())}/{total_estrellas}",
            f"Discos: {int(current_region_counts[:,0].sum())}/{total_discos}",
            f"SI: {current_SI:.3f} (x_o={x_o:.3f})",
            f"Anillos: {num_regions} (R={max_radius}px)"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(annotated_frame, text, (20, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    return annotated_frame

# Procesamiento del video
cv2.namedWindow("Detección con Anillos Concéntricos", cv2.WINDOW_NORMAL)
print("\nProcesando video... (Presione 'q' para detener)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.predict(frame, conf=0.346730)
    annotated_frame = process_frame(frame, results, show_labels)
    
    cv2.imshow("Detección con Anillos Concéntricos", annotated_frame)
    out.write(annotated_frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Resultados finales
total_particles_per_region = region_counts.sum(axis=1)
x_i_final = np.where(total_particles_per_region > 0,
                    region_counts[:,1] / total_particles_per_region,
                    0)

valid_regions = total_particles_per_region > 0
final_SI = calculate_segregation_index(x_i_final[valid_regions], x_o)

print("\n=== RESULTADOS FINALES ===")
print(f"Total frames: {total_frames}")
print(f"Estrellas detectadas: {int(region_counts[:,1].sum())} (de {total_estrellas})")
print(f"Discos detectados: {int(region_counts[:,0].sum())} (de {total_discos})")
print(f"Fracción inicial estrellas (x_o): {x_o:.3f}")
print(f"Índice de Segregación (SI): {final_SI:.3f}")

print("\nDISTRIBUCIÓN POR REGIONES:")
print("Región | Radio (px) | Estrellas | Discos | Fracción estrellas")
for i in range(num_regions):
    radius_inner = int((i) * (max_radius / num_regions))
    radius_outer = int((i+1) * (max_radius / num_regions))
    if total_particles_per_region[i] > 0:
        frac = region_counts[i,1] / total_particles_per_region[i]
        print(f"{i+1:6} | {radius_inner:3}-{radius_outer:3} | {int(region_counts[i,1]):8} | {int(region_counts[i,0]):6} | {frac:.3f}")
    else:
        print(f"{i+1:6} | {radius_inner:3}-{radius_outer:3} | {'-':8} | {'-':6} | {'-':5}")

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\nVideo guardado en: {output_path}")