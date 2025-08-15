import cv2
import numpy as np
import time
import os
import math
import multiprocessing
import subprocess
import shutil
from queue import Empty

# ==============================================================================
# PARÁMETROS DE CONFIGURACIÓN
# ==============================================================================

# Parámetros del tambor
center_x, center_y = 940, 560
radius = 1051 - 556

# Tiempo de inicio deseado
start_time_sec = 38.5  # segundos
rotation_angle = 0  # Negativo para rotación horaria

# Ruta del video de entrada y salida
input_video_path = "1_segregacion/1_imagenesvideos/0_crudos/20250806_154957.mp4"
output_video_path = "/home/tdelorenzi/1_yolo/1_segregacion/1_imagenesvideos/1_procesados/20250806_154957.mp4"

# Directorio para los videos temporales
TEMP_DIR = "temp_video_segments"

# --- NUEVO PARÁMETRO DE CALIDAD ---
# Constant Rate Factor (CRF). Usado por el códec x264.
# Rango: 0-51. Menor es mejor calidad. 0 es sin pérdidas.
# 18 es visualmente sin pérdidas o casi. 23 es el valor por defecto.
CRF_VALUE = "18"

# Frecuencia con la que los trabajadores reportan el progreso (en fotogramas)
PROGRESS_REPORT_STEP = 100

# ==============================================================================
# FUNCIÓN DEL PROCESO TRABAJADOR (MODIFICADA PARA ALTA CALIDAD)
# ==============================================================================

def process_video_segment(args):
    """
    Procesa un segmento del video.
    MODIFICADO: Usa un subproceso de FFmpeg para escribir los fotogramas en alta calidad.
    """
    (task_id, start_frame, num_frames_to_process, input_path, 
     temp_dir, params, progress_queue) = args

    # Extraer parámetros
    center_x, center_y = params['center']
    radius = params['radius']
    rotation_angle = params['rotation_angle']
    crf = params['crf']
    
    temp_output_path = os.path.join(temp_dir, f"segment_{task_id:04d}.mp4")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video en el trabajador {task_id}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # --- INICIO DE LA MODIFICACIÓN: REEMPLAZO DE VideoWriter POR FFmpeg ---
    
    output_width = radius * 2
    output_height = radius * 2

    # Comando de FFmpeg para recibir fotogramas crudos y codificarlos en H.264
    ffmpeg_command = [
        'ffmpeg',
        '-y',  # Sobrescribir archivo de salida si existe
        '-f', 'rawvideo',  # Formato de entrada: video crudo
        '-vcodec', 'rawvideo',  # Códec de entrada
        '-s', f'{output_width}x{output_height}',  # Tamaño del fotograma WxH
        '-pix_fmt', 'bgr24',  # Formato de píxeles de OpenCV (BGR, 8-bit por canal)
        '-r', str(fps),  # Tasa de fotogramas
        '-i', '-',  # Leer la entrada desde la tubería (stdin)
        '-c:v', 'libx264',  # Códec de video de salida: H.264
        '-pix_fmt', 'yuv420p', # Formato de píxeles para máxima compatibilidad
        '-crf', crf,  # Factor de calidad constante (CRF)
        '-preset', 'medium', # Balance entre velocidad de compresión y tamaño
        temp_output_path,
    ]

    # Iniciar el subproceso de FFmpeg
    ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # --- FIN DE LA MODIFICACIÓN ---

    mask = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    cv2.circle(mask, (radius, radius), radius, (255, 255, 255), -1)
    rotation_matrix = cv2.getRotationMatrix2D((radius, radius), rotation_angle, 1)

    x1 = max(center_x - radius, 0)
    y1 = max(center_y - radius, 0)
    x2 = min(center_x + radius, width)
    y2 = min(center_y + radius, height)
    crop_width = x2 - x1
    crop_height = y2 - y1

    padded = np.zeros((output_height, output_width, 3), dtype=np.uint8) if crop_width != output_width or crop_height != output_height else None
    start_y_pad = radius - (center_y - y1)
    start_x_pad = radius - (center_x - x1)

    frames_processed_in_batch = 0
    try:
        for i in range(num_frames_to_process):
            ret, frame = cap.read()
            if not ret:
                break
            
            cropped = frame[y1:y2, x1:x2]
            
            if padded is not None:
                padded.fill(0)
                padded[start_y_pad:start_y_pad+crop_height, start_x_pad:start_x_pad+crop_width] = cropped
                final_crop = padded
            else:
                final_crop = cropped

            masked = cv2.bitwise_and(final_crop, mask)
            rotated = cv2.warpAffine(masked, rotation_matrix, (output_width, output_height))
            
            # --- MODIFICACIÓN: Escribir el fotograma al subproceso de FFmpeg ---
            ffmpeg_process.stdin.write(rotated.tobytes())

            frames_processed_in_batch += 1
            if frames_processed_in_batch >= PROGRESS_REPORT_STEP:
                progress_queue.put(frames_processed_in_batch)
                frames_processed_in_batch = 0

    finally:
        # Asegurarse de que todo se cierre correctamente
        cap.release()
        # Cerrar la tubería de FFmpeg y esperar a que termine la codificación
        if ffmpeg_process.stdin:
            ffmpeg_process.stdin.close()
        ffmpeg_process.wait()

    if frames_processed_in_batch > 0:
        progress_queue.put(frames_processed_in_batch)

    return temp_output_path

# ==============================================================================
# PROCESO PRINCIPAL (Sin cambios, pero ahora orquesta el nuevo flujo)
# ==============================================================================

def main():
    start_time_total = time.perf_counter()

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError("No se pudo abrir el video de entrada para leer metadatos")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    start_frame_global = int(start_time_sec * fps)
    frames_to_process_total = total_frames - start_frame_global
    
    if frames_to_process_total <= 0:
        print("El tiempo de inicio es posterior a la duración del video.")
        return

    num_processes = multiprocessing.cpu_count()
    print(f"Usando {num_processes} procesos para el trabajo.")
    frames_per_process = math.ceil(frames_to_process_total / num_processes)

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()

    tasks = []
    params = {
        'center': (center_x, center_y),
        'radius': radius,
        'rotation_angle': rotation_angle,
        'crf': CRF_VALUE # <-- Pasamos el valor de CRF a los trabajadores
    }
    for i in range(num_processes):
        start_frame_for_task = start_frame_global + (i * frames_per_process)
        num_frames_for_task = min(frames_per_process, total_frames - start_frame_for_task)
        if num_frames_for_task <= 0:
            continue
        tasks.append((
            i, start_frame_for_task, num_frames_for_task, input_video_path,
            TEMP_DIR, params, progress_queue
        ))

    print(f"\nProcesando {frames_to_process_total} fotogramas en {len(tasks)} segmentos con CRF={CRF_VALUE}...")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        async_result = pool.map_async(process_video_segment, tasks)

        processed_frames = 0
        start_time_proc = time.perf_counter()
        
        while not async_result.ready():
            try:
                num_done = progress_queue.get(timeout=0.1) 
                processed_frames += num_done
            except Empty:
                pass
            
            elapsed = time.perf_counter() - start_time_proc
            if elapsed > 0 and processed_frames > 0:
                progress = processed_frames / frames_to_process_total
                fps_estimate = processed_frames / elapsed
                remaining_frames = frames_to_process_total - processed_frames
                if fps_estimate > 0:
                    remaining_time = remaining_frames / fps_estimate
                    mins, secs = divmod(int(remaining_time), 60)
                    hrs, mins = divmod(mins, 60)
                    eta_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"
                else:
                    eta_str = "??:??:??"
                
                print(
                    f"Progreso: {processed_frames}/{frames_to_process_total} "
                    f"({progress:.1%}) | "
                    f"ETA: {eta_str} | "
                    f"Velocidad: {fps_estimate:.1f} FPS",
                    end='\r'
                )
        
        temp_video_files = async_result.get()

    print(f"Progreso: {frames_to_process_total}/{frames_to_process_total} (100.0%) | Completado.                 ")
    temp_video_files = [f for f in temp_video_files if f is not None]
    temp_video_files.sort()

    print("\nProcesamiento de segmentos completado. Concatenando archivos...")
    file_list_path = os.path.join(TEMP_DIR, "filelist.txt")
    with open(file_list_path, 'w') as f:
        for file_path in temp_video_files:
            # Usar rutas relativas simples es más seguro para el concat de ffmpeg
            f.write(f"file '{os.path.basename(file_path)}'\n")

    # Comando de concatenación. Se ejecuta desde dentro del directorio temporal.
    ffmpeg_concat_command = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', os.path.basename(file_list_path),
        '-c', 'copy', os.path.abspath(output_video_path)
    ]
    try:
        # Ejecutar el comando desde el directorio temporal para que encuentre los archivos
        subprocess.run(ffmpeg_concat_command, check=True, capture_output=True, text=True, cwd=TEMP_DIR)
        print("Concatenación completada exitosamente.")
    except subprocess.CalledProcessError as e:
        print("\n--- ERROR DURANTE LA CONCATENACIÓN CON FFMPEG ---")
        print(f"Comando ejecutado en el directorio: {TEMP_DIR}")
        print(f"Stderr: {e.stderr}")
        return

    print("Limpiando archivos temporales...")
    shutil.rmtree(TEMP_DIR)
    elapsed_total = time.perf_counter() - start_time_total
    print(f"\n¡Proceso completado en {elapsed_total:.2f} segundos!")
    print(f"Video final de alta calidad guardado en: {output_video_path}")


if __name__ == '__main__':
    main()