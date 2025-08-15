import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, Any, List

# -- 1. CONFIGURACIÓN CENTRALIZADA (MODIFICADA) --
CONFIGURACION: Dict[str, Any] = {
    "rutas_modelos": {
        # Claves renombradas para mayor claridad
        'particula_convexa': Path('/home/tdelorenzi/1_yolo/1_segregacion/4_modelos/1_discos-pmma/train/weights/best.pt'),
        'particula_no_convexa': Path('/home/tdelorenzi/1_yolo/1_segregacion/4_modelos/2_noconvexas-1corte/train/weights/best.pt')
    },
    "umbrales_confianza": {
        # Claves renombradas para mayor claridad
        'particula_convexa': 0.55,
        'particula_no_convexa': 0.7
    },
    "ruta_video": Path('/home/tdelorenzi/1_yolo/1_segregacion/1_imagenesvideos/1_procesados/20250804_103822.mp4'),
    "directorio_salida": Path('/home/tdelorenzi/1_yolo/1_segregacion/6_resultados/2025-08-04-reticula_polar_ref_convexa'), # Directorio de salida actualizado
    "informacion_clases": {
        # Nombres de clases actualizados
        'nombres': ['particula convexa', 'particula no convexa'],
        'colores': [(0, 255, 0), (0, 0, 255)] # Verde para convexas, Rojo para no convexas
    },
    "conteos_particulas": {
        # Conteos renombrados para mayor claridad
        'total_convexas': 193,
        'total_no_convexas': 174
    },
    "parametros_analisis": {
        'num_anillos': 9,
        'num_sectores': 16
    }
}

class AnalizadorSegregacion:
    """
    Encapsula la lógica para el análisis de segregación usando una cuadrícula polar.
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
        self.centro = (self.ancho // 2, self.alto // 2)
        self.radio_maximo = min(self.centro)
        
        self.num_anillos = configuracion["parametros_analisis"]["num_anillos"]
        self.num_sectores = configuracion["parametros_analisis"]["num_sectores"]
        self.num_regiones = self.num_anillos * self.num_sectores
        
        self.ancho_anillo = self.radio_maximo / self.num_anillos
        self.angulo_por_sector = 360 / self.num_sectores

        # --- CAMBIO CLAVE: CÁLCULO DE x_o CON PARTÍCULAS CONVEXAS COMO REFERENCIA ---
        total_convexas = configuracion["conteos_particulas"]["total_convexas"]
        total_no_convexas = configuracion["conteos_particulas"]["total_no_convexas"]
        self.x_o = total_convexas / (total_convexas + total_no_convexas)

        self.datos_frames: List[Dict[str, Any]] = []
        self.total_frames = 0
         
        self.configuracion["directorio_salida"].mkdir(parents=True, exist_ok=True)
        self.ruta_video_salida = self.configuracion["directorio_salida"] / 'segregacion_polar_grid_ref_convexa.mp4'
        codificador = cv2.VideoWriter_fourcc(*'mp4v')
        self.escritor = cv2.VideoWriter(str(self.ruta_video_salida), codificador, self.fps, (self.ancho, self.alto))

    def _cargar_modelos(self) -> Dict[str, YOLO]:
        return { nombre: YOLO(ruta) for nombre, ruta in self.configuracion["rutas_modelos"].items() }

    def _obtener_preferencia_usuario(self) -> bool:
        print("\nOpciones de visualización:\n1. Mostrar puntos con etiquetas y confianza\n2. Mostrar solo puntos (sin texto)")
        return input("Seleccione el modo de visualización (1/2): ").strip() == '1'
         
    def _detectar_particulas(self, frame: np.ndarray, modelo: YOLO, id_clase: int, umbral_conf: float) -> List[Dict]:
        """Detecta partículas y las asigna a una celda de la cuadrícula polar."""
        detecciones = []
        resultados = modelo.predict(frame, conf=umbral_conf, verbose=False)
         
        for caja in resultados[0].boxes:
            x1, y1, x2, y2 = map(int, caja.xyxy[0])
            centro_p = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            dx = centro_p[0] - self.centro[0]
            dy = centro_p[1] - self.centro[1]
            distancia = math.sqrt(dx**2 + dy**2)
            angulo_rad = math.atan2(dy, dx)
            angulo_deg = (math.degrees(angulo_rad) + 360) % 360
            
            indice_anillo = min(int(distancia / self.ancho_anillo), self.num_anillos - 1)
            indice_sector = int(angulo_deg / self.angulo_por_sector)
            
            region = (indice_anillo * self.num_sectores) + indice_sector
             
            detecciones.append({
                'caja': (x1, y1, x2, y2), 'confianza': float(caja.conf[0]),
                'id_clase': id_clase, 'region': region
            })
        return detecciones

    def _dibujar_anotaciones(self, frame: np.ndarray, detecciones: List[Dict], si_actual: float, conteos: np.ndarray):
        """Dibuja la cuadrícula polar, puntos de detección y texto informativo."""
        for i in range(1, self.num_anillos + 1):
            radio = int(i * self.ancho_anillo)
            cv2.circle(frame, self.centro, radio, (200, 200, 200), 1)

        for i in range(self.num_sectores):
            angulo_linea = math.radians(i * self.angulo_por_sector)
            punto_final = (
                self.centro[0] + int(self.radio_maximo * math.cos(angulo_linea)),
                self.centro[1] + int(self.radio_maximo * math.sin(angulo_linea))
            )
            cv2.line(frame, self.centro, punto_final, (200, 200, 200), 1)

        for det in detecciones:
            x1, y1, x2, y2 = det['caja']
            centro_p = ((x1 + x2) // 2, (y1 + y2) // 2)
            id_clase = det['id_clase']
            color = self.configuracion['informacion_clases']['colores'][id_clase]
            cv2.circle(frame, centro_p, 3, color, -1)
            if self.mostrar_etiquetas:
                etiqueta = f"{self.configuracion['informacion_clases']['nombres'][id_clase]} {det['confianza']:.2f}"
                cv2.putText(frame, etiqueta, (centro_p[0] - 30, centro_p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # --- Textos en pantalla actualizados ---
        texto_info = [
            f"P. Convexas: {int(conteos[:,0].sum())}/{self.configuracion['conteos_particulas']['total_convexas']}",
            f"P. No Convexas: {int(conteos[:,1].sum())}/{self.configuracion['conteos_particulas']['total_no_convexas']}",
            f"SI (ref. convexas): {si_actual:.3f} (x_o={self.x_o:.3f})",]
        for i, texto in enumerate(texto_info):
            cv2.putText(frame, texto, (20, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    def procesar_frame(self, frame: np.ndarray) -> np.ndarray:
        self.total_frames += 1
        conteos_region_actual = np.zeros((self.num_regiones, 2))
        
        # Detecciones con nombres de variables actualizados
        detecciones_convexas = self._detectar_particulas(frame, self.modelos['particula_convexa'], 0, self.configuracion['umbrales_confianza']['particula_convexa'])
        detecciones_no_convexas = self._detectar_particulas(frame, self.modelos['particula_no_convexa'], 1, self.configuracion['umbrales_confianza']['particula_no_convexa'])
        todas_detecciones = detecciones_convexas + detecciones_no_convexas
        
        for det in todas_detecciones:
            if 0 <= det['region'] < self.num_regiones:
                conteos_region_actual[det['region'], det['id_clase']] += 1
        
        total_particulas = conteos_region_actual.sum(axis=1)

        # --- CAMBIO CLAVE: CÁLCULO DE x_i CON PARTÍCULAS CONVEXAS (índice 0) ---
        x_i = np.where(total_particulas > 0, conteos_region_actual[:,0] / total_particulas, 0)
        
        regiones_validas = total_particulas > 0
        si_actual = 0
        if np.any(regiones_validas) and self.x_o > 0:
            promedio_xi_xo = np.mean(x_i[regiones_validas] / self.x_o)
            # La fórmula del SI no cambia, pero ahora opera sobre las nuevas fracciones
            si_actual = np.sqrt(np.sum(((x_i[regiones_validas] / self.x_o - promedio_xi_xo) ** 2) / len(x_i[regiones_validas])))

        self._almacenar_datos_frame(si_actual, conteos_region_actual, x_i, total_particulas)
        frame_anotado = self._dibujar_anotaciones(frame.copy(), todas_detecciones, si_actual, conteos_region_actual)
        return frame_anotado

    def _almacenar_datos_frame(self, si, conteos, x_i, total_particulas):
        # --- Nombres de columnas en el DataFrame actualizados ---
        entrada_frame = { 
            'frame': self.total_frames, 
            'tiempo': self.total_frames / self.fps, 
            'SI_ref_convexa': si,
            'total_convexas_detectadas': int(conteos[:,0].sum()),
            'total_no_convexas_detectadas': int(conteos[:,1].sum()) 
        }
        for i in range(self.num_regiones):
            entrada_frame[f'region_{i}_convexas'] = int(conteos[i, 0])
            entrada_frame[f'region_{i}_no_convexas'] = int(conteos[i, 1])
            entrada_frame[f'region_{i}_fraccion_convexas'] = x_i[i] if total_particulas[i] > 0 else 0
        self.datos_frames.append(entrada_frame)

    def ejecutar_analisis(self):
        print("\nProcesando video... (Presione 'q' para detener)")
        cv2.namedWindow("Detección con Cuadrícula Polar", cv2.WINDOW_NORMAL)
        while self.captura.isOpened():
            ret, frame = self.captura.read()
            if not ret: break
            frame_anotado = self.procesar_frame(frame)
            cv2.imshow("Detección con Cuadrícula Polar", frame_anotado)
            self.escritor.write(frame_anotado)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        print("\nProcesamiento de video finalizado.")
        self._limpiar_recursos()

    def guardar_resultados(self):
        # --- Nombres de archivos de salida actualizados ---
        ruta_csv = self.configuracion["directorio_salida"] / 'datos_segregacion_ref_convexa.csv'
        df = pd.DataFrame(self.datos_frames)
        df.to_csv(ruta_csv, index=False)
        print(f"\nDatos de segregación guardados en: {ruta_csv}")

        plt.style.use('seaborn-v0_8-whitegrid')

        plt.figure(figsize=(12, 6))
        plt.plot(df['tiempo'], df['SI_ref_convexa'], label='Índice de Segregación (ref. convexas)', color='royalblue')
        plt.xlabel('Tiempo (s)'), plt.ylabel('SI'), plt.title('Evolución del Índice de Segregación en el Tiempo')
        plt.legend()
        plt.savefig(self.configuracion["directorio_salida"] / 'evolucion_si_ref_convexa.png')
        plt.close()

        # --- Gráfico de mapa de calor (Heatmap) actualizado ---
        # Se calcula la fracción promedio de la nueva partícula de referencia
        fraccion_promedio = [df[f'region_{i}_fraccion_convexas'].mean() for i in range(self.num_regiones)]
        
        grid_data = np.array(fraccion_promedio).reshape((self.num_anillos, self.num_sectores))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(grid_data, cmap='viridis', interpolation='nearest', origin='lower')
        
        cbar = ax.figure.colorbar(im, ax=ax)
        # Etiqueta de la barra de color actualizada
        cbar.ax.set_ylabel('Fracción promedio de "particula convexa"', rotation=-90, va="bottom")

        ax.set_title('Distribución Promedio de Partículas en Cuadrícula Polar')
        
        ax.set_xlabel('Sector Angular')
        ax.set_xticks(np.arange(self.num_sectores))
        ax.set_xticklabels([f'{int(i*self.angulo_por_sector)}°' for i in range(self.num_sectores)], rotation=45, ha="right")

        ax.set_ylabel('Anillo Radial (Centro -> Borde)')
        ax.set_yticks(np.arange(self.num_anillos))
        ax.set_yticklabels([f'{i+1}' for i in range(self.num_anillos)])
        
        plt.tight_layout()
        plt.savefig(self.configuracion["directorio_salida"] / 'distribucion_polar_heatmap_ref_convexa.png')
        plt.close()

        print(f"Gráficos guardados en: {self.configuracion['directorio_salida']}")

    def _limpiar_recursos(self):
        self.captura.release()
        self.escritor.release()
        cv2.destroyAllWindows()
        print(f"\nVideo de salida guardado en: {self.ruta_video_salida}")

def principal():
    try:
        analizador = AnalizadorSegregacion(CONFIGURACION)
        analizador.ejecutar_analisis()
        analizador.guardar_resultados()
    except Exception as e:
        print(f"Ha ocurrido un error: {e}")

if __name__ == "__main__":
    principal()