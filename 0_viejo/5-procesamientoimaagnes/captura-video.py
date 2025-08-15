import subprocess
from datetime import datetime
import os
import signal
import math
import shutil
import psutil
from pynput import keyboard

# Configuración
OUTPUT_DIR = "/home/tdelorenzi/testYolo/6-carpeta"
RESOLUTION = "1920:1080"
FPS = 60
BITRATE = "20M"
URL = "http://localhost:8080/videofeed?type=some.mjpeg"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def format_size(bytes_size):
    """Formatea el tamaño en bytes a una representación legible"""
    if bytes_size == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(bytes_size, 1024)))
    p = math.pow(1024, i)
    s = round(bytes_size / p, 2)
    return f"{s} {size_name[i]}"

def kill_ffmpeg_processes():
    """Termina cualquier proceso ffmpeg residual"""
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] == 'ffmpeg':
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except:
                proc.kill()

def get_video_info(file_path):
    """Obtiene información técnica del video"""
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return None
            
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name,width,height,bit_rate,r_frame_rate,pix_fmt',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            file_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        info = result.stdout.strip().split('\n')
        
        return {
            'codec': info[0],
            'width': info[1],
            'height': info[2],
            'bitrate': f"{round(int(info[3])/1000000, 2)} Mbps" if info[3] != 'N/A' else BITRATE,
            'fps': eval(info[4]),
            'pix_fmt': info[5]
        }
    except:
        return None

def on_press(key):
    """Manejador de teclas"""
    global process, start_time, output_file
    
    try:
        if key == keyboard.Key.space and not hasattr(on_press, 'recording_started'):
            # Iniciar grabación
            kill_ffmpeg_processes()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(OUTPUT_DIR, f"fullhd_{timestamp}.mkv")
            
            command = [
                'ffmpeg',
                '-i', URL,
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-vf', f'fps={FPS}',
                '-b:v', BITRATE,
                '-pix_fmt', 'yuv420p',
                '-profile:v', 'high',
                '-y',
                output_file
            ]
            
            print("\n⏺️  Grabación INICIADA! Presiona 'Q' para detener...")
            process = subprocess.Popen(command, stderr=subprocess.PIPE, universal_newlines=True)
            start_time = datetime.now()
            on_press.recording_started = True
            
        elif key == keyboard.KeyCode.from_char('q') and hasattr(on_press, 'recording_started'):
            # Detener grabación y salir
            process.send_signal(signal.SIGINT)
            process.wait(timeout=5)
            
            end_time = datetime.now()
            duration = end_time - start_time
            minutes, seconds = divmod(duration.total_seconds(), 60)
            time_str = f"{int(minutes)} min {int(seconds)} seg"
            
            # Obtener información del video
            file_valid = os.path.exists(output_file) and os.path.getsize(output_file) > 0
            if file_valid:
                file_size = os.path.getsize(output_file)
                formatted_size = format_size(file_size)
                video_info = get_video_info(output_file) or {}
                disk_usage = shutil.disk_usage(OUTPUT_DIR)
            
            # Mostrar reporte completo
            print("\n" + "═"*60)
            print("📹  REPORTE FINAL DE GRABACIÓN")
            print("═"*60)
            print(f"📂 Archivo: {output_file}")
            print(f"📊 Tamaño: {formatted_size if file_valid else '0B (vacío)'}")
            print(f"⏱  Duración: {time_str}")
            print(f"🕒 Inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🕒 Fin: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if file_valid and video_info:
                print("\n🔧 ESPECIFICACIONES TÉCNICAS:")
                print(f"🎞  Codec: {video_info.get('codec', 'H.264')}")
                print(f"🖼  Resolución: {video_info.get('width', '1920')}x{video_info.get('height', '1080')}")
                print(f"📈 FPS: {video_info.get('fps', FPS)}")
                print(f"⚡ Bitrate: {video_info.get('bitrate', BITRATE)}")
                print(f"🎨 Formato de píxeles: {video_info.get('pix_fmt', 'yuv420p')}")
            
            print("\n💾 ALMACENAMIENTO:")
            print(f"📁 Directorio: {OUTPUT_DIR}")
            if file_valid:
                print(f"🆓 Espacio libre: {format_size(disk_usage.free)}")
            print("═"*60)
            print("✅ Programa finalizado")
            
            # Salir del programa
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if process and process.poll() is None:
            process.terminate()
        return False

# Limpiar procesos antes de iniciar
kill_ffmpeg_processes()

print("\n" + "━"*50)
print("🎥  SISTEMA DE GRABACIÓN DE VIDEO (UN SOLO USO)")
print("━"*50)
print("👉 Presiona ESPACIO para comenzar a grabar")
print("👉 Presiona Q para detener y salir")
print("━"*50)

# Configurar listener (se cerrará después de la primera grabación)
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()