import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, Any, List
# ¡Importante! Necesitarás instalar scipy: pip install scipy
from scipy.spatial import KDTree
from tqdm import tqdm # Librería para barras de progreso, ¡muy útil! Instalar con: pip install tqdm

# -- 1. CONFIGURACIÓN CENTRALIZADA --
CONFIGURACION: Dict[str, Any] = {
    "rutas_modelos": {
        'particula_convexa': Path('/home/tdelorenzi/1_yolo/1_segregacion/4_modelos/1_discos-pmma/train/weights/best.pt'),
        'particula_no_convexa': Path('/home/tdelorenzi/1_yolo/1_segregacion/4_modelos/2_noconvexas-1corte/train/weights/best.pt')
    },
    "umbrales_confianza": {
        'particula_convexa': 0.55,
        'particula_no_convexa': 0.7
    },
    "ruta_video": Path('/home/tdelorenzi/1_yolo/1_segregacion/1_imagenesvideos/1_procesados/20250804_103822.mp4'),
    # Directorio de salida único para todos los resultados
    "directorio_salida": Path('/home/tdelorenzi/1_yolo/1_segregacion/6_resultados/analisis_sensibilidad_k'),
    "informacion_clases": {
        'nombres': ['particula convexa', 'particula no convexa'], # id_clase 0 y 1
        'colores': [(0, 255, 0), (0, 0, 255)]
    },
    "conteos_particulas": {
        'total_convexas': 193,
        'total_no_convexas': 174
    }
    # "parametros_analisis" se define en el bucle principal ahora
}

class AnalizadorSegregacionNDM:
    """
    Versión optimizada para calcular el índice NDM sin generar video.
    """
    def __init__(self, configuracion: Dict[str, Any]):
        self.configuracion = configuracion
        self.modelos = self._cargar_modelos()

        self.captura = cv2.VideoCapture(str(configuracion["ruta_video"]))
        if not self.captura.isOpened():
            raise IOError(f"Error al abrir el video: {configuracion['ruta_video']}")

        self.fps = self.captura.get(cv2.CAP_PROP_FPS)
        self.total_video_frames = int(self.captura.get(cv2.CAP_PROP_FRAME_COUNT))

        total_convexas = configuracion["conteos_particulas"]["total_convexas"]
        total_no_convexas = configuracion["conteos_particulas"]["total_no_convexas"]
        self.x_o = total_convexas / (total_convexas + total_no_convexas) if (total_convexas + total_no_convexas) > 0 else 0

        self.datos_frames: List[Dict[str, Any]] = []

    def _cargar_modelos(self) -> Dict[str, YOLO]:
        return {nombre: YOLO(ruta) for nombre, ruta in self.configuracion["rutas_modelos"].items()}

    def _detectar_particulas(self, frame: np.ndarray, modelo: YOLO, id_clase: int, umbral_conf: float) -> List[Dict]:
        detecciones = []
        resultados = modelo.predict(frame, conf=umbral_conf, verbose=False)
        for caja in resultados[0].boxes:
            x1, y1, x2, y2 = map(int, caja.xyxy[0])
            centro_p = ((x1 + x2) // 2, (y1 + y2) // 2)
            detecciones.append({'centro': centro_p, 'id_clase': id_clase})
        return detecciones

    def _calcular_indice_ndm(self, detecciones: List[Dict], k: int) -> float:
        if len(detecciones) < k + 1: return 0.0

        puntos = np.array([d['centro'] for d in detecciones])
        clases = np.array([d['id_clase'] for d in detecciones])
        indices_referencia = np.where(clases == 0)[0]
        if len(indices_referencia) == 0: return 0.0

        kdtree = KDTree(puntos)
        distancias, indices_vecinos = kdtree.query(puntos[indices_referencia], k=k + 1)

        fracciones_locales = []
        for i in range(len(indices_referencia)):
            vecinos_reales = indices_vecinos[i][1:]
            clases_vecinos = clases[vecinos_reales]
            conteo_mismo_tipo = np.sum(clases_vecinos == 0)
            fracciones_locales.append(conteo_mismo_tipo / k)

        if not fracciones_locales: return 0.0
        p_observado = np.mean(fracciones_locales)
        if (1.0 - self.x_o) == 0: return 1.0
        
        ndm = (p_observado - self.x_o) / (1.0 - self.x_o)
        indice_final = 1.0 - ndm
        return max(0, min(1, indice_final))

    def procesar_video(self, k_actual: int):
        # Usamos tqdm para mostrar una barra de progreso en la consola
        with tqdm(total=self.total_video_frames, desc=f"Procesando k={k_actual}") as pbar:
            frame_actual = 0
            while self.captura.isOpened():
                ret, frame = self.captura.read()
                if not ret: break
                
                detecciones_convexas = self._detectar_particulas(frame, self.modelos['particula_convexa'], 0, self.configuracion['umbrales_confianza']['particula_convexa'])
                detecciones_no_convexas = self._detectar_particulas(frame, self.modelos['particula_no_convexa'], 1, self.configuracion['umbrales_confianza']['particula_no_convexa'])
                todas_detecciones = detecciones_convexas + detecciones_no_convexas

                ndm_actual = self._calcular_indice_ndm(todas_detecciones, k_actual)

                self.datos_frames.append({
                    'frame': frame_actual,
                    'tiempo': frame_actual / self.fps,
                    'indice_ndm': ndm_actual,
                    'k_usado': k_actual
                })
                frame_actual += 1
                pbar.update(1)

        self.captura.release()

    def guardar_resultados(self, k_actual: int):
        if not self.datos_frames:
            print(f"No se procesaron frames para k={k_actual}, no hay resultados que guardar.")
            return

        ruta_csv = self.configuracion["directorio_salida"] / f'datos_segregacion_k_{k_actual}.csv'
        df = pd.DataFrame(self.datos_frames)
        df.to_csv(ruta_csv, index=False)
        print(f"Resultados para k={k_actual} guardados en: {ruta_csv}")

def generar_grafico_comparativo(directorio_salida: Path, valores_k: range):
    print("\nGenerando gráfico comparativo...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 8))

    for k in valores_k:
        ruta_csv = directorio_salida / f'datos_segregacion_k_{k}.csv'
        if ruta_csv.exists():
            df = pd.read_csv(ruta_csv)
            plt.plot(df['tiempo'], df['indice_ndm'], label=f'k = {k}')
        else:
            print(f"Advertencia: No se encontró el archivo {ruta_csv} para graficar.")

    plt.xlabel('Tiempo (s)')
    plt.ylabel('Índice de Mezcla NDM (1=Mezcla)')
    plt.title('Análisis de Sensibilidad: Evolución del Índice NDM para Diferentes `k`')
    plt.ylim(0, 1.05)
    plt.legend(title='Valor de k')
    plt.grid(True)
    
    ruta_grafico = directorio_salida / 'grafico_comparativo_sensibilidad_k.png'
    plt.savefig(ruta_grafico)
    plt.close()
    print(f"Gráfico comparativo guardado en: {ruta_grafico}")

def principal():
    try:
        # Rango de valores de k para el análisis de sensibilidad
        valores_k_a_probar = range(9, 14) # Esto probará k = 2, 3, 4, 5, 6, 7, 8

        # Asegurarse de que el directorio de salida exista
        directorio_salida = CONFIGURACION["directorio_salida"]
        directorio_salida.mkdir(parents=True, exist_ok=True)
        
        print("--- Iniciando Análisis de Sensibilidad para el parámetro k ---")
        for k in valores_k_a_probar:
            analizador = AnalizadorSegregacionNDM(CONFIGURACION)
            analizador.procesar_video(k_actual=k)
            analizador.guardar_resultados(k_actual=k)
            # Liberar memoria
            del analizador

        # Generar el gráfico final que compara todos los resultados
        generar_grafico_comparativo(directorio_salida, valores_k_a_probar)
        
        print("\n--- Análisis de Sensibilidad Finalizado ---")

    except ImportError:
        print("\nERROR: Faltan librerías. Por favor, asegúrate de tener 'scipy' y 'tqdm' instalados.")
        print("Puedes instalarlos ejecutando: pip install scipy tqdm")
    except Exception as e:
        print(f"\nHa ocurrido un error inesperado: {e}")

if __name__ == "__main__":
    principal()