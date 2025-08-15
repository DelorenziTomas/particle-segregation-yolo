import cv2
import numpy as np
import multiprocessing as mp
from functools import partial

# Variable global para compartir las coordenadas del mouse
mouse_coords = None

def mouse_callback(event, x, y, flags, param):
    """Callback para capturar eventos del mouse"""
    global mouse_coords
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_coords = (x, y)
        print(f"Posición del mouse: x={x}, y={y}")

def process_frame(frame, scale_factor=1.0):
    """Función para procesamiento paralelo de frames"""
    if scale_factor != 1.0:
        return cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
    return frame

def video_worker(video_path, output_queue, start_second=43):
    """Worker para lectura de video"""
    cap = cv2.VideoCapture(video_path)
    
    # Establecer el punto de inicio en milisegundos (43 segundos * 1000)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_second * 1000)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Si llega al final, reiniciar desde el segundo 43
            cap.set(cv2.CAP_PROP_POS_MSEC, start_second * 1000)
            continue
        output_queue.put(frame)
    cap.release()

def display_worker(input_queue):
    """Worker para visualización"""
    global mouse_coords
    
    cv2.namedWindow('Video', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Configurar el callback del mouse
    cv2.setMouseCallback('Video', mouse_callback)
    
    while True:
        if not input_queue.empty():
            frame = input_queue.get()
            
            # Opcional: dibujar las coordenadas en el frame
            if mouse_coords:
                # Dibujar un círculo en la posición del mouse
                cv2.circle(frame, mouse_coords, 10, (0, 255, 0), 2)
                # Dibujar el texto con las coordenadas
                text = f"({mouse_coords[0]}, {mouse_coords[1]})"
                cv2.putText(frame, text, (mouse_coords[0] + 15, mouse_coords[1] - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Video', frame)
        
        # Salir con 'q' o ESC
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
    
    cv2.destroyAllWindows()

def main():
    video_path = '1_segregacion/1_imagenesvideos/0_crudos/20250806_154957.mp4'
    start_second = 43  # Segundo desde el que comenzar
    
    # Verificar video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: No se pudo abrir el video")
        print(f"Ruta probada: {video_path}")
        exit()
    
    # Obtener propiedades del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"Video info: {width}x{height}, {fps} FPS")
    print(f"Iniciando reproducción desde el segundo: {start_second}")
    print("Mueve el mouse sobre el video para ver las coordenadas en la terminal")
    print("Presiona 'q' o ESC para salir")
    
    # Colas para comunicación entre procesos
    frame_queue = mp.Queue(maxsize=10)
    
    # Crear procesos
    processes = [
        mp.Process(target=video_worker, args=(video_path, frame_queue, start_second)),
        mp.Process(target=display_worker, args=(frame_queue,))
    ]
    
    # Iniciar procesos
    for p in processes:
        p.start()
    
    # Esperar a que termine el proceso de visualización
    processes[1].join()
    
    # Terminar otros procesos
    for p in processes:
        p.terminate()

if __name__ == '__main__':
    mp.freeze_support()
    main()