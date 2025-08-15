import cv2
import numpy as np

# Parámetros del tambor
center_x, center_y = 562, 542
radius = 525

# Tiempo de inicio deseado (36 segundos)
start_time = 36  # segundos
rotation_angle = -90  # Negativo para rotación horaria

# Ruta del video de entrada y salida
input_video_path = "/home/tdelorenzi/testYolo/1-imagenesvideos/20250527_160929.mp4"
output_video_path = "/home/tdelorenzi/testYolo/1-imagenesvideos/tambor_recortado_36s_rotado2.mp4"

# Abrir el video de entrada
cap = cv2.VideoCapture(input_video_path)

# Obtener propiedades del video
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calcular el frame de inicio
start_frame = int(start_time * fps)

# Crear objeto para escribir el video de salida
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (radius*2, radius*2))

# Crear una máscara circular
mask = np.zeros((radius*2, radius*2, 3), dtype=np.uint8)
cv2.circle(mask, (radius, radius), radius, (255, 255, 255), -1)

# Contador de frames
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Solo procesar frames a partir del segundo 36
    if frame_count >= start_frame:
        # Recortar la región rectangular que contiene el círculo
        x1 = max(center_x - radius, 0)
        y1 = max(center_y - radius, 0)
        x2 = min(center_x + radius, width)
        y2 = min(center_y + radius, height)
        
        # Ajustar si el círculo está cerca del borde
        cropped = frame[y1:y2, x1:x2]
        
        # Si el recorte no es cuadrado (debido a bordes), rellenar con negro
        if cropped.shape[0] != radius*2 or cropped.shape[1] != radius*2:
            padded = np.zeros((radius*2, radius*2, 3), dtype=np.uint8)
            start_y = radius - (center_y - y1)
            start_x = radius - (center_x - x1)
            padded[start_y:start_y+cropped.shape[0], start_x:start_x+cropped.shape[1]] = cropped
            cropped = padded
        
        # Aplicar la máscara circular
        masked = cv2.bitwise_and(cropped, mask)
        
        # Rotar la imagen 45 grados en sentido horario
        rotation_matrix = cv2.getRotationMatrix2D((radius, radius), rotation_angle, 1)
        rotated = cv2.warpAffine(masked, rotation_matrix, (radius*2, radius*2))
        
        # Escribir el frame procesado
        out.write(rotated)
    
    frame_count += 1

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video procesado guardado en: {output_video_path}")
print(f"El video comienza desde el segundo {start_time} (frame {start_frame})")
print(f"El video ha sido rotado {abs(rotation_angle)} grados en sentido horario")