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

# -- 1. CONFIGURACIÓN CENTRALIZADA (ACTUALIZADA A NOTACIÓN DE ARTÍCULO) --
CONFIGURACION: Dict[str, Any] = {
    "rutas_modelos": {
        'particula_A': Path('/home/tdelorenzi/1_yolo/1_segregacion/4_modelos/4_discos-pmma2/train/weights/best.pt'),
        'particula_B': Path('/home/tdelorenzi/1_yolo/1_segregacion/4_modelos/3_discospmmaazul/train/weights/best.pt')
    },
    "umbrales_confianza": {
        'particula_A': 0.55,
        'particula_B': 0.7
    },
    "ruta_video": Path('/home/tdelorenzi/1_yolo/1_segregacion/1_imagenesvideos/1_procesados/20250806_154957.mp4'),
    "directorio_salida": Path('/home/tdelorenzi/1_yolo/1_segregacion/6_resultados/2025-08-06-Analisis-Comparativo'),
    "informacion_clases": {
        'nombres': ['particula A (convexa)', 'particula B (no convexa)'], # id_clase 0 y 1 respectivamente
        'colores': [(0, 255, 0), (0, 0, 255)] # Verde para A, Azul para B
    },
    "conteos_particulas": {
        'total_A': 122,
        'total_B': 123
    },
    "parametros_analisis": {
        # k (o z) es el número de vecinos a considerar. 6 es un valor común.
        'k_vecinos': 6
    }
}

class AnalizadorSegregacion:
    """
    Encapsula la lógica para el análisis de segregación usando el
    Índice de Mezcla Basado en Vecinos (NDM) y el índice de Godlieb.
    Notación actualizada para reflejar conceptos de publicaciones académicas.
    - 'A' se refiere a partículas convexas.
    - 'B' se refiere a partículas no convexas.
    - 'M' denota un índice de mezcla.
    - 'p' denota una fracción de partículas.
    - 'k' o 'z' denota el número de vecinos.
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

        total_A = configuracion["conteos_particulas"]["total_A"]
        total_B = configuracion["conteos_particulas"]["total_B"]
        # p_global: Fracción global de cada tipo de partícula
        self.p_global_A = total_A / (total_A + total_B)
        self.p_global_B = total_B / (total_A + total_B)

        self.datos_frames: List[Dict[str, Any]] = []
        self.total_frames = 0

        self.configuracion["directorio_salida"].mkdir(parents=True, exist_ok=True)
        self.ruta_video_salida = self.configuracion["directorio_salida"] / 'video_analisis_comparativo.mp4'
        codificador = cv2.VideoWriter_fourcc(*'mp4v')
        self.escritor = cv2.VideoWriter(str(self.ruta_video_salida), codificador, self.fps, (self.ancho, self.alto))

    def _cargar_modelos(self) -> Dict[str, YOLO]:
        # Cambiamos los nombres para la nueva notación A/B
        modelos_cargados = {
            'particula_A': YOLO(self.configuracion["rutas_modelos"]["particula_A"]),
            'particula_B': YOLO(self.configuracion["rutas_modelos"]["particula_B"])
        }
        return modelos_cargados

    def _obtener_preferencia_usuario(self) -> bool:
        print("\nOpciones de visualización:\n1. Mostrar puntos con etiquetas y confianza\n2. Mostrar solo puntos (sin texto)")
        return input("Seleccione el modo de visualización (1/2): ").strip() == '1'

    def _detectar_particulas(self, frame: np.ndarray, modelo: YOLO, id_clase: int, umbral_conf: float) -> List[Dict]:
        detecciones = []
        resultados = modelo.predict(frame, conf=umbral_conf, verbose=False)
        for caja in resultados[0].boxes:
            x1, y1, x2, y2 = map(int, caja.xyxy[0])
            centro_p = ((x1 + x2) // 2, (y1 + y2) // 2)
            detecciones.append({'centro': centro_p, 'confianza': float(caja.conf[0]), 'id_clase': id_clase})
        return detecciones

    def _calcular_indice_ndm_por_tipo(self, detecciones: List[Dict], id_clase_referencia: int) -> float:
        """
        Calcula el Índice NDM desde la perspectiva de un tipo de partícula (A o B).
        M = 1 - I_segregacion
        Donde I_segregacion = (p_local_promedio - p_global_ref) / (1 - p_global_ref)
        """
        k = self.configuracion["parametros_analisis"]["k_vecinos"]
        if len(detecciones) < k + 1: return 0.0

        puntos = np.array([d['centro'] for d in detecciones])
        clases = np.array([d['id_clase'] for d in detecciones])

        indices_referencia = np.where(clases == id_clase_referencia)[0]
        if len(indices_referencia) == 0: return 0.0

        kdtree = KDTree(puntos)
        _, indices_vecinos = kdtree.query(puntos[indices_referencia], k=k + 1)
        
        fracciones_locales = []
        for i in range(len(indices_referencia)):
            vecinos_reales = indices_vecinos[i][1:]
            clases_vecinos = clases[vecinos_reales]
            conteo_mismo_tipo = np.sum(clases_vecinos == id_clase_referencia)
            fracciones_locales.append(conteo_mismo_tipo / k)

        if not fracciones_locales: return 0.0

        p_local_promedio = np.mean(fracciones_locales)
        
        # p_global_ref: Fracción global del tipo de partícula de referencia
        p_global_ref = self.p_global_A if id_clase_referencia == 0 else self.p_global_B
        
        if (1.0 - p_global_ref) == 0: return 1.0
        
        # I_segregacion: Intensidad de segregación
        I_segregacion = (p_local_promedio - p_global_ref) / (1.0 - p_global_ref)
        
        # M_ndm: Índice de mezcla (Mixing Index)
        M_ndm = 1.0 - I_segregacion
        
        return max(0, min(1, M_ndm))

    def _calcular_indice_godlieb(self, detecciones: List[Dict]) -> float:
        """
        Calcula el índice de mezcla según la fórmula de Godlieb et al. (2007).
        M = mean(2 * z_diff / k) para todas las partículas.
        """
        k = self.configuracion["parametros_analisis"]["k_vecinos"]
        if len(detecciones) < k + 1: return 0.0

        puntos = np.array([d['centro'] for d in detecciones])
        clases = np.array([d['id_clase'] for d in detecciones])
        kdtree = KDTree(puntos)

        M_locales_godlieb = []
        # Itera sobre TODAS las partículas
        for i in range(len(puntos)):
            _, indices_vecinos = kdtree.query(puntos[i], k=k + 1)
            vecinos_reales = indices_vecinos[1:]
            clase_actual = clases[i]
            clases_vecinos = clases[vecinos_reales]
            
            # z_diff: Número de vecinos de tipo DIFERENTE
            z_diff = np.sum(clases_vecinos != clase_actual)
            
            # Fórmula local de Godlieb
            M_locales_godlieb.append((2 * z_diff) / k)

        if not M_locales_godlieb: return 0.0
        
        # M_godlieb: Promedio de los índices locales
        M_godlieb = np.mean(M_locales_godlieb)
        return M_godlieb

    def _dibujar_anotaciones(self, frame: np.ndarray, detecciones: List[Dict], indices: Dict[str, float]):
        """Dibuja los puntos de detección y el texto informativo con todos los índices."""
        for det in detecciones:
            centro_p, id_clase = det['centro'], det['id_clase']
            color = self.configuracion['informacion_clases']['colores'][id_clase]
            cv2.circle(frame, centro_p, 4, color, -1)
            if self.mostrar_etiquetas:
                etiqueta = f"{self.configuracion['informacion_clases']['nombres'][id_clase]} {det['confianza']:.2f}"
                cv2.putText(frame, etiqueta, (centro_p[0] - 30, centro_p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        texto_info = [
            f"M_ndm (ref A): {indices.get('M_ndm_A', 0):.3f}",
            f"M_ndm (ref B): {indices.get('M_ndm_B', 0):.3f}",
            f"M_godlieb: {indices.get('M_godlieb', 0):.3f}",
            f"(1=Mezcla, 0=Segreg.)"
        ]
        for i, texto in enumerate(texto_info):
            cv2.putText(frame, texto, (20, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    def procesar_video(self):
        print("\nProcesando video para análisis comparativo... (Presione 'q' para detener)")
        cv2.namedWindow("Análisis Comparativo de Segregación", cv2.WINDOW_NORMAL)

        while self.captura.isOpened():
            ret, frame = self.captura.read()
            if not ret: break

            self.total_frames += 1
            
            # Usamos A y B para las detecciones
            detecciones_A = self._detectar_particulas(frame, self.modelos['particula_A'], 0, self.configuracion['umbrales_confianza']['particula_A'])
            detecciones_B = self._detectar_particulas(frame, self.modelos['particula_B'], 1, self.configuracion['umbrales_confianza']['particula_B'])
            todas_detecciones = detecciones_A + detecciones_B

            # Calculamos los tres índices de mezcla (M)
            M_ndm_A = self._calcular_indice_ndm_por_tipo(todas_detecciones, id_clase_referencia=0)
            M_ndm_B = self._calcular_indice_ndm_por_tipo(todas_detecciones, id_clase_referencia=1)
            M_godlieb = self._calcular_indice_godlieb(todas_detecciones)

            self.datos_frames.append({
                'frame': self.total_frames,
                'tiempo': self.total_frames / self.fps,
                'M_ndm_A': M_ndm_A,
                'M_ndm_B': M_ndm_B,
                'M_godlieb': M_godlieb,
                'total_A_detectadas': len(detecciones_A),
                'total_B_detectadas': len(detecciones_B)
            })
            
            indices_actuales = {
                'M_ndm_A': M_ndm_A, 
                'M_ndm_B': M_ndm_B, 
                'M_godlieb': M_godlieb
            }
            frame_anotado = self._dibujar_anotaciones(frame.copy(), todas_detecciones, indices_actuales)
            cv2.imshow("Análisis Comparativo de Segregación", frame_anotado)
            self.escritor.write(frame_anotado)

            if cv2.waitKey(1) & 0xFF == ord('q'): break

        print("\nProcesamiento de video finalizado.")
        self._limpiar_recursos()

    def guardar_resultados(self):
        if not self.datos_frames:
            print("No se procesaron frames, no hay resultados que guardar.")
            return

        ruta_csv = self.configuracion["directorio_salida"] / 'datos_segregacion_comparativo.csv'
        df = pd.DataFrame(self.datos_frames)
        df.to_csv(ruta_csv, index=False)
        print(f"\nDatos de segregación guardados en: {ruta_csv}")

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 7))
        plt.plot(df['tiempo'], df['M_ndm_A'], label='Índice NDM (Ref: Partículas A)', color='green', linestyle='-')
        plt.plot(df['tiempo'], df['M_ndm_B'], label='Índice NDM (Ref: Partículas B)', color='blue', linestyle='--')
        plt.plot(df['tiempo'], df['M_godlieb'], label='Índice Godlieb (2007)', color='red', linestyle=':')
        
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Índice de Mezcla (M)')
        plt.title(f'Evolución Comparativa de Índices de Mezcla (k={self.configuracion["parametros_analisis"]["k_vecinos"]})')
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        ruta_grafico = self.configuracion["directorio_salida"] / 'evolucion_comparativa.png'
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
        analizador = AnalizadorSegregacion(CONFIGURACION)
        analizador.procesar_video()
        analizador.guardar_resultados()
    except ImportError:
        print("\nERROR: La librería 'scipy' no está instalada.")
        print("Por favor, instálala ejecutando: pip install scipy")
    except Exception as e:
        print(f"\nHa ocurrido un error inesperado: {e}")

if __name__ == "__main__":
    principal()