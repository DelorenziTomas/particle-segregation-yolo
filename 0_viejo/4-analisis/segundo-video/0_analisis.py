from ultralytics import YOLO
import cv2
import os

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
output_path = os.path.join(output_dir, 'resultado_30-05-2024.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Función para dibujar anotaciones personalizadas
def draw_custom_annotations(frame, results, show_text=True):
    annotated_frame = frame.copy()
    
    for result in results:
        for box in result.boxes:
            # Obtener datos de la detección
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Seleccionar color según la clase
            color = class_colors[class_id]
            
            # Dibujar bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            if show_text:
                # Preparar texto de la etiqueta
                label = f"{class_names[class_id]} {confidence:.2f}"
                
                # Calcular tamaño del texto
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                
                # Dibujar fondo para el texto
                cv2.rectangle(
                    annotated_frame, 
                    (x1, y1 - text_height - 10), 
                    (x1 + text_width, y1), 
                    color, -1)
                
                # Dibujar texto
                cv2.putText(
                    annotated_frame, 
                    label, 
                    (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
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
    results = model.predict(frame, conf=0.346730)
    
    # Dibujar anotaciones según el modo seleccionado
    annotated_frame = draw_custom_annotations(frame, results, show_labels)
    
    # Mostrar y guardar resultados
    cv2.imshow("Detección de Partículas", annotated_frame)
    out.write(annotated_frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\nProcesamiento completado. Video guardado en {output_path}")