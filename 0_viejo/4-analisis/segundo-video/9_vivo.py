import cv2
from ultralytics import YOLO
import subprocess

# Configuración
ADB_PORT = "8080"
MODEL_PATH = 'yolo11x.pt'

def setup_adb():
    """Configura el reenvío de puertos ADB"""
    try:
        subprocess.run(["adb", "kill-server"], check=True)
        subprocess.run(["adb", "start-server"], check=True)
        subprocess.run(["adb", "forward", f"tcp:{ADB_PORT}", f"tcp:{ADB_PORT}"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error ADB: {e}")
        return False

def main():
    if not setup_adb():
        print("\nConfigura ADB y activa Depuración USB")
        return

    # Inicializar ventana UNA sola vez
    cv2.namedWindow('YOLOv11 - Detección USB', cv2.WINDOW_NORMAL)
    
    cap = cv2.VideoCapture(f"http://localhost:{ADB_PORT}/videofeed")
    if not cap.isOpened():
        print("\nError: Verifica la app de cámara en tu celular")
        return

    model = YOLO(MODEL_PATH)  # Cargar modelo una vez

    print("\nDetección activa. Presiona Q para salir.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\nError: Revisa la conexión USB")
                break
            
            # Procesamiento eficiente
            results = model(frame, stream=True, imgsz=640)
            
            # Actualizar la MISMA ventana
            for result in results:
                cv2.imshow('YOLOv11 - Detección USB', result.plot())
            
            # Control de salida
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()  # Cierra todas las ventanas al final
        print("Programa terminado")

if __name__ == "__main__":
    main()