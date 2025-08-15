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

# -- 1. CONFIGURACIÓN CENTRALIZADA (ACTUALIZADA PARA NDM) --
CONFIGURACION: Dict[str, Any] = {
    "rutas_modelos": {
        'particula_convexa': Path('/home/tdelorenzi/1_yolo/1_segregacion/4_modelos/4_discos-pmma2/train/weights/best.pt'),
        'particula_no_convexa': Path('/home/tdelorenzi/1_yolo/1_segregacion/4_modelos/3_discospmmaazul/train/weights/best.pt')
    },
    "umbrales_confianza": {
        'particula_convexa': 0.55,
        'particula_no_convexa': 0.7
    },
    "ruta_video": Path('/home/tdelorenzi/1_yolo/1_segregacion/1_imagenesvideos/1_procesados/20250806_154957.mp4'),
    "directorio_salida": Path('/home/tdelorenzi/1_yolo/1_segregacion/6_resultados/2025-08-06-NDM_k6'),
    "informacion_clases": {
        'nombres': ['particula convexa', 'particula no convexa'], # id_clase 0 y 1 respectivamente
        'colores': [(0, 255, 0), (0, 0, 255)] # Verde para convexas, Azul para no convexas
    },
    "conteos_particulas": {
        'total_convexas': 122,
        'total_no_convexas': 123
    },
    "parametros_analisis": {
        # K es el número de vecinos a considerar para cada partícula. 6 es un valor común.
        'k_vecinos': 6
    }
}

class AnalizadorSegregacionNDM:
    """
    Encapsula la lógica para el análisis de segregación usando el
    Índice de Mezcla Basado en la Distancia al Vecino (NDM).
    """
    def __init__(self, configuracion: Dict[str, Any]):
        self.configuracion = configuracion
        self.modelos = self._cargar_modelos()
        self.mostrar_etiquetas = self._obtener_preferencia_usuario()

        self.captura = cv2.VideoCapture(str(configuracion["ruta_video"]))
        if not self.captura.isOpened():
            raise IOError(f"Error al abrir el video: {configuracion['ruta_video']}")

        self.ancho = int(self.captura.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.alto = int(self.captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.captura.get(cv2.CAP_PROP_FPS)

        # Fracción global del tipo de partícula de referencia (convexas, id_clase=0)
        total_convexas = configuracion["conteos_particulas"]["total_convexas"]
        total_no_convexas = configuracion["conteos_particulas"]["total_no_convexas"]
        self.x_o = total_convexas / (total_convexas + total_no_convexas)

        self.datos_frames: List[Dict[str, Any]] = []
        self.total_frames = 0

        self.configuracion["directorio_salida"].mkdir(parents=True, exist_ok=True)
        self.ruta_video_salida = self.configuracion["directorio_salida"] / 'video_analisis_ndm.mp4'
        codificador = cv2.VideoWriter_fourcc(*'mp4v')
        self.escritor = cv2.VideoWriter(str(self.ruta_video_salida), codificador, self.fps, (self.ancho, self.alto))

    def _cargar_modelos(self) -> Dict[str, YOLO]:
        return {nombre: YOLO(ruta) for nombre, ruta in self.configuracion["rutas_modelos"].items()}

    def _obtener_preferencia_usuario(self) -> bool:
        print("\nOpciones de visualización:\n1. Mostrar puntos con etiquetas y confianza\n2. Mostrar solo puntos (sin texto)")
        return input("Seleccione el modo de visualización (1/2): ").strip() == '1'

    def _detectar_particulas(self, frame: np.ndarray, modelo: YOLO, id_clase: int, umbral_conf: float) -> List[Dict]:
        """Detecta partículas y devuelve sus centros, confianzas e id de clase."""
        detecciones = []
        resultados = modelo.predict(frame, conf=umbral_conf, verbose=False)

        for caja in resultados[0].boxes:
            x1, y1, x2, y2 = map(int, caja.xyxy[0])
            centro_p = ((x1 + x2) // 2, (y1 + y2) // 2)
            detecciones.append({
                'centro': centro_p,
                'confianza': float(caja.conf[0]),
                'id_clase': id_clase
            })
        return detecciones

    def _calcular_indice_ndm(self, detecciones: List[Dict]) -> float:
        """
        Calcula el Índice de Mezcla Basado en la Distancia al Vecino (NDM).
        Este índice varía de 0 (totalmente segregado) a 1 (perfectamente mezclado).
        """
        k = self.configuracion["parametros_analisis"]["k_vecinos"]
        
        if len(detecciones) < k + 1:
            return 0.0 # No hay suficientes partículas para el análisis

        # Separar detecciones y coordenadas
        puntos = np.array([d['centro'] for d in detecciones])
        clases = np.array([d['id_clase'] for d in detecciones])

        # Partículas de referencia (convexas, id_clase = 0)
        indices_referencia = np.where(clases == 0)[0]
        if len(indices_referencia) == 0:
            return 0.0 # No hay partículas de referencia para analizar

        # Construir el KDTree con todas las partículas para una búsqueda eficiente
        kdtree = KDTree(puntos)

        fracciones_locales = []
        # Para cada partícula de referencia, encontrar sus vecinos
        # Se buscan k+1 vecinos porque el punto mismo está incluido en los resultados
        distancias, indices_vecinos = kdtree.query(puntos[indices_referencia], k=k + 1)

        for i in range(len(indices_referencia)):
            # Excluir el punto mismo (que siempre es el vecino más cercano)
            vecinos_reales = indices_vecinos[i][1:]
            
            # Contar cuántos de los vecinos son de la clase de referencia (id_clase = 0)
            clases_vecinos = clases[vecinos_reales]
            conteo_mismo_tipo = np.sum(clases_vecinos == 0)
            
            # Calcular la fracción local de partículas de referencia
            fraccion_local = conteo_mismo_tipo / k
            fracciones_locales.append(fraccion_local)

        if not fracciones_locales:
            return 0.0

        # Calcular la fracción promedio observada en los vecindarios
        p_observado = np.mean(fracciones_locales)

        # Normalizar el índice
        # p_observado: Fracción promedio de partículas de referencia en los vecindarios
        # self.x_o: Fracción global de partículas de referencia (estado ideal/aleatorio)
        # 1.0: Fracción en un estado totalmente segregado (todos los vecinos son del mismo tipo)
        if (1.0 - self.x_o) == 0: return 1.0 # Si solo hay un tipo de partícula, está perfectamente mezclado
        
        ndm = (p_observado - self.x_o) / (1.0 - self.x_o)
        
        # El índice NDM suele definirse como 1 - ndm para que 1 sea mezclado
        # y 0 sea segregado.
        indice_final = 1.0 - ndm
        
        return max(0, min(1, indice_final)) # Asegurar que esté entre 0 y 1

    def _dibujar_anotaciones(self, frame: np.ndarray, detecciones: List[Dict], ndm_actual: float):
        """Dibuja los puntos de detección y el texto informativo."""
        for det in detecciones:
            centro_p = det['centro']
            id_clase = det['id_clase']
            color = self.configuracion['informacion_clases']['colores'][id_clase]
            cv2.circle(frame, centro_p, 4, color, -1) # Círculos un poco más grandes
            if self.mostrar_etiquetas:
                etiqueta = f"{self.configuracion['informacion_clases']['nombres'][id_clase]} {det['confianza']:.2f}"
                cv2.putText(frame, etiqueta, (centro_p[0] - 30, centro_p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        num_convexas = sum(1 for d in detecciones if d['id_clase'] == 0)
        num_no_convexas = len(detecciones) - num_convexas

        texto_info = [
            f"P. Convexas: {num_convexas}/{self.configuracion['conteos_particulas']['total_convexas']}",
            f"P. No Convexas: {num_no_convexas}/{self.configuracion['conteos_particulas']['total_no_convexas']}",
            f"Indice NDM (k={self.configuracion['parametros_analisis']['k_vecinos']}): {ndm_actual:.3f} (1=Mezcla, 0=Segreg.)"
        ]
        for i, texto in enumerate(texto_info):
            cv2.putText(frame, texto, (20, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    def procesar_video(self):
        print("\nProcesando video... (Presione 'q' para detener)")
        cv2.namedWindow("Análisis de Segregación con NDM", cv2.WINDOW_NORMAL)

        while self.captura.isOpened():
            ret, frame = self.captura.read()
            if not ret: break

            self.total_frames += 1

            detecciones_convexas = self._detectar_particulas(frame, self.modelos['particula_convexa'], 0, self.configuracion['umbrales_confianza']['particula_convexa'])
            detecciones_no_convexas = self._detectar_particulas(frame, self.modelos['particula_no_convexa'], 1, self.configuracion['umbrales_confianza']['particula_no_convexa'])
            todas_detecciones = detecciones_convexas + detecciones_no_convexas

            ndm_actual = self._calcular_indice_ndm(todas_detecciones)

            # Almacenar datos del frame
            self.datos_frames.append({
                'frame': self.total_frames,
                'tiempo': self.total_frames / self.fps,
                'indice_ndm': ndm_actual,
                'total_convexas_detectadas': len(detecciones_convexas),
                'total_no_convexas_detectadas': len(detecciones_no_convexas)
            })

            frame_anotado = self._dibujar_anotaciones(frame.copy(), todas_detecciones, ndm_actual)
            cv2.imshow("Análisis de Segregación con NDM", frame_anotado)
            self.escritor.write(frame_anotado)

            if cv2.waitKey(1) & 0xFF == ord('q'): break

        print("\nProcesamiento de video finalizado.")
        self._limpiar_recursos()

    def guardar_resultados(self):
        if not self.datos_frames:
            print("No se procesaron frames, no hay resultados que guardar.")
            return

        ruta_csv = self.configuracion["directorio_salida"] / 'datos_segregacion_ndm.csv'
        df = pd.DataFrame(self.datos_frames)
        df.to_csv(ruta_csv, index=False)
        print(f"\nDatos de segregación guardados en: {ruta_csv}")

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 6))
        plt.plot(df['tiempo'], df['indice_ndm'], label=f'Índice NDM (k={self.configuracion["parametros_analisis"]["k_vecinos"]})', color='darkcyan')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Índice de Mezcla NDM')
        plt.title('Evolución del Índice de Mezcla en el Tiempo')
        plt.ylim(0, 1.05) # Fijar el eje Y entre 0 y 1
        plt.legend()
        plt.grid(True)
        ruta_grafico = self.configuracion["directorio_salida"] / 'evolucion_ndm.png'
        plt.savefig(ruta_grafico)
        plt.close()
        print(f"Gráfico de evolución guardado en: {ruta_grafico}")

    def _limpiar_recursos(self):
        self.captura.release()
        self.escritor.release()
        cv2.destroyAllWindows()
        print(f"\nVideo de salida guardado en: {self.ruta_video_salida}")

def principal():
    try:
        analizador = AnalizadorSegregacionNDM(CONFIGURACION)
        analizador.procesar_video()
        analizador.guardar_resultados()
    except ImportError:
        print("\nERROR: La librería 'scipy' no está instalada.")
        print("Por favor, instálala ejecutando: pip install scipy")
    except Exception as e:
        print(f"\nHa ocurrido un error inesperado: {e}")

if __name__ == "__main__":
    principal()