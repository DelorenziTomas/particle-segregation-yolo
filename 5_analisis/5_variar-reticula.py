import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, Any, List
import shutil
from tqdm import tqdm
import itertools
from datetime import datetime

# -- 1. CONFIGURACIÓN CENTRALIZADA --
CONFIGURACION: Dict[str, Any] = {
    "rutas_modelos": {
        'convexa-pmma': Path('/home/tdelorenzi/1_yolo/1_segregacion/4_modelos/1_discos-pmma/train/weights/best.pt'),
        'no-convexa1': Path('/home/tdelorenzi/1_yolo/1_segregacion/4_modelos/2_noconvexas-1corte/train/weights/best.pt')
    },
    "umbrales_confianza": {
        'convexa-pmma': 0.5877,
        'no-convexa1': 0.7
    },
    "ruta_video": Path('/home/tdelorenzi/1_yolo/1_segregacion/1_imagenesvideos/1_procesados/20250804_103822.mp4'),
    # MODIFICACIÓN: Apunta al directorio raíz de resultados.
    "directorio_salida_base": Path('/home/tdelorenzi/1_yolo/1_segregacion/6_resultados/'),
    "informacion_clases": {
        'nombres': ['convexa-pmma', 'no-convexa1'],
        'colores': [(0, 255, 0), (0, 0, 255)]
    },
    "conteos_particulas": {
        'total_discos': 194,
        'total_1corte': 173
    },
    "parametros_analisis": {
        'num_anillos': 5,
        'num_sectores': 12
    },
    # --- PARÁMETROS DE OPTIMIZACIÓN ---
    "parametros_optimizacion": {
        'device': 'cuda:0',
        'batch_size': 64,
        'half_precision': True,
        'guardar_video': False
    }
}

class AnalizadorSegregacion:
    """
    Encapsula la lógica para el análisis de segregación usando una cuadrícula polar.
    """
    def __init__(self, configuracion: Dict[str, Any]):
        self.configuracion = configuracion
        
        opt_params = self.configuracion["parametros_optimizacion"]
        self.device = opt_params['device']
        self.batch_size = opt_params['batch_size']
        self.half_precision = opt_params['half_precision']
        self.guardar_video = opt_params['guardar_video']
        
        self.modelos = self._cargar_modelos()
        self.mostrar_etiquetas = False
        
        self.captura = cv2.VideoCapture(str(configuracion["ruta_video"]))
        if not self.captura.isOpened():
            raise IOError(f"Error al abrir el video: {configuracion['ruta_video']}")

        self.ancho = int(self.captura.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.alto = int(self.captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.captura.get(cv2.CAP_PROP_FPS)
        self.total_video_frames = int(self.captura.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.centro = (self.ancho // 2, self.alto // 2)
        self.radio_maximo = min(self.centro)

        self.num_anillos = configuracion["parametros_analisis"]["num_anillos"]
        self.num_sectores = configuracion["parametros_analisis"]["num_sectores"]
        self.num_regiones = self.num_anillos * self.num_sectores

        self.ancho_anillo = self.radio_maximo / self.num_anillos
        self.angulo_por_sector = 360 / self.num_sectores

        total_discos = configuracion["conteos_particulas"]["total_discos"]
        total_1corte = configuracion["conteos_particulas"]["total_1corte"]
        self.x_o = total_1corte / (total_discos + total_1corte) if (total_discos + total_1corte) > 0 else 0

        self.datos_frames: List[Dict[str, Any]] = []
        self.total_frames = 0

        self.directorio_salida = self.configuracion["directorio_salida"]
        self.directorio_salida.mkdir(parents=True, exist_ok=True)

        if self.guardar_video:
            self.ruta_video_salida = self.directorio_salida / 'segregacion_polar_grid.mkv'
            codificador = cv2.VideoWriter_fourcc(*'FFV1')
            self.escritor = cv2.VideoWriter(str(self.ruta_video_salida), codificador, self.fps, (self.ancho, self.alto))
        else:
            self.escritor = None

    def _cargar_modelos(self) -> Dict[str, YOLO]:
        modelos_cargados = {}
        for nombre, ruta in self.configuracion["rutas_modelos"].items():
            model = YOLO(ruta)
            model.to(self.device)
            modelos_cargados[nombre] = model
        return modelos_cargados

    def _detectar_particulas_batch(self, frames_batch: List[np.ndarray], modelo: YOLO, id_clase: int, umbral_conf: float) -> List[List[Dict]]:
        resultados_batch = modelo.predict(frames_batch, conf=umbral_conf, verbose=False, device=self.device, half=self.half_precision)
        
        detecciones_totales = []
        for resultados in resultados_batch:
            detecciones_frame = []
            for caja in resultados.boxes:
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
                detecciones_frame.append({
                    'caja': (x1, y1, x2, y2), 'confianza': float(caja.conf[0]),
                    'id_clase': id_clase, 'region': region
                })
            detecciones_totales.append(detecciones_frame)
        return detecciones_totales

    def _dibujar_anotaciones(self, frame: np.ndarray, detecciones: List[Dict], si_actual: float, conteos: np.ndarray):
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

        texto_info = [
            f"1corte: {int(conteos[:,1].sum())}/{self.configuracion['conteos_particulas']['total_1corte']}",
            f"Discos: {int(conteos[:,0].sum())}/{self.configuracion['conteos_particulas']['total_discos']}",
            f"SI: {si_actual:.3f} (x_o={self.x_o:.3f})",
            f"Grid: {self.num_anillos} Anillos, {self.num_sectores} Sectores"
        ]
        for i, texto in enumerate(texto_info):
            cv2.putText(frame, texto, (20, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    def procesar_lote(self, frames: List[np.ndarray]):
        detecciones_discos_batch = self._detectar_particulas_batch(
            frames, self.modelos['convexa-pmma'], 0, self.configuracion['umbrales_confianza']['convexa-pmma']
        )
        detecciones_estrellas_batch = self._detectar_particulas_batch(
            frames, self.modelos['no-convexa1'], 1, self.configuracion['umbrales_confianza']['no-convexa1']
        )
        
        for i, frame in enumerate(frames):
            self.total_frames += 1
            conteos_region_actual = np.zeros((self.num_regiones, 2))
            
            todas_detecciones = detecciones_discos_batch[i] + detecciones_estrellas_batch[i]
            
            for det in todas_detecciones:
                if 0 <= det['region'] < self.num_regiones:
                    conteos_region_actual[det['region'], det['id_clase']] += 1
            
            total_particulas = conteos_region_actual.sum(axis=1)
            
            out_array = np.zeros_like(total_particulas, dtype=float)
            x_i = np.divide(conteos_region_actual[:,1], total_particulas, out=out_array, where=total_particulas!=0)
            
            regiones_validas = total_particulas > 0
            si_actual = 0
            if np.any(regiones_validas) and self.x_o > 0:
                promedio_xi_xo = np.mean(x_i[regiones_validas] / self.x_o)
                desviacion_estandar = np.sqrt(np.sum(((x_i[regiones_validas] / self.x_o - promedio_xi_xo) ** 2)) / len(x_i[regiones_validas]))
                si_actual = desviacion_estandar

            self._almacenar_datos_frame(si_actual, conteos_region_actual, x_i, total_particulas)
            
            if self.escritor is not None:
                frame_anotado = self._dibujar_anotaciones(frame.copy(), todas_detecciones, si_actual, conteos_region_actual)
                self.escritor.write(frame_anotado)

    def _almacenar_datos_frame(self, si, conteos, x_i, total_particulas):
        entrada_frame = { 'frame': self.total_frames, 'tiempo': self.total_frames / self.fps, 'SI': si,
                         'total_estrellas': int(conteos[:,1].sum()), 'total_discos': int(conteos[:,0].sum()) }
        for i in range(self.num_regiones):
            entrada_frame[f'region_{i}_estrellas'] = int(conteos[i, 1])
            entrada_frame[f'region_{i}_discos'] = int(conteos[i, 0])
            entrada_frame[f'region_{i}_fraccion'] = x_i[i]
        self.datos_frames.append(entrada_frame)

    def ejecutar_analisis(self):
        frames_batch = []
        with tqdm(total=self.total_video_frames, desc="Procesando Video (GPU)") as pbar_video:
            while self.captura.isOpened():
                ret, frame = self.captura.read()
                if not ret:
                    break
                
                frames_batch.append(frame)

                if len(frames_batch) == self.batch_size:
                    self.procesar_lote(frames_batch)
                    pbar_video.update(len(frames_batch))
                    frames_batch = []
            
            if frames_batch:
                self.procesar_lote(frames_batch)
                pbar_video.update(len(frames_batch))
        
        self._limpiar_recursos()

    def guardar_resultados(self):
        ruta_csv = self.directorio_salida / 'segregation_data_polar_grid.csv'
        df = pd.DataFrame(self.datos_frames)
        df.to_csv(ruta_csv, index=False)
        
        plt.style.use('seaborn-v0_8-whitegrid')

        plt.figure(figsize=(12, 6))
        plt.plot(df['tiempo'], df['SI'], label='Índice de Segregación (SI)', color='royalblue')
        plt.xlabel('Tiempo (s)'), plt.ylabel('SI')
        plt.title(f'Evolución del SI ({self.num_anillos} Anillos, {self.num_sectores} Sectores)')
        plt.legend()
        plt.savefig(self.directorio_salida / 'si_evolution_polar_grid.png')
        plt.close()

        fraccion_promedio = [df[f'region_{i}_fraccion'].mean() for i in range(self.num_regiones)]
        grid_data = np.array(fraccion_promedio).reshape((self.num_anillos, self.num_sectores))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(grid_data, cmap='viridis', interpolation='nearest', origin='lower')
        
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Fracción promedio de "no-convexa1"', rotation=-90, va="bottom")
        ax.set_title(f'Distribución Promedio de Partículas ({self.num_anillos} Anillos, {self.num_sectores} Sectores)')
        
        ax.set_xlabel('Sector Angular')
        ax.set_xticks(np.arange(self.num_sectores))
        ax.set_xticklabels([f'{int(i*self.angulo_por_sector)}°' for i in range(self.num_sectores)], rotation=45, ha="right")
        ax.set_ylabel('Anillo Radial (Centro -> Borde)')
        ax.set_yticks(np.arange(self.num_anillos))
        ax.set_yticklabels([f'{i+1}' for i in range(self.num_anillos)])
        
        plt.tight_layout()
        plt.savefig(self.directorio_salida / 'polar_distribution_heatmap.png')
        plt.close()

    def _limpiar_recursos(self):
        self.captura.release()
        if self.escritor is not None:
            self.escritor.release()
        cv2.destroyAllWindows()


def principal():
    valores_anillos = [ 3, 6, 9 ] # 10, 13, 15
    valores_sectores = [4, 8, 12, 16] # 8, 16, 18, 22

    # --- MODIFICACIÓN: Crear un directorio único para cada ejecución ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nombre_analisis = f"analisis_polar_{timestamp}"
    directorio_base_run = CONFIGURACION["directorio_salida_base"] / nombre_analisis
    
    # Se elimina el bloque que borraba directorios.
    
    combinaciones = list(itertools.product(valores_anillos, valores_sectores))
    
    print(f"Iniciando análisis de {len(combinaciones)} configuraciones...")
    print(f"Los resultados se guardarán en: {directorio_base_run}") # Informar al usuario
    
    with tqdm(total=len(combinaciones), desc="Progreso General") as pbar_general:
        for anillos, sectores in combinaciones:
            nombre_carpeta = f"{anillos}_anillos_{sectores}_sectores"
            pbar_general.set_description(f"Analizando: {nombre_carpeta}")

            config_actual = CONFIGURACION.copy()
            config_actual["parametros_analisis"] = {'num_anillos': anillos, 'num_sectores': sectores}
            # Se usa el directorio base único para esta ejecución
            config_actual["directorio_salida"] = directorio_base_run / nombre_carpeta
            
            try:
                analizador = AnalizadorSegregacion(config_actual)
                analizador.ejecutar_analisis()
                analizador.guardar_resultados()
                # Se mueven los prints informativos al final de la ejecución
            except Exception as e:
                print(f"\n❌ Ha ocurrido un error durante el análisis de {nombre_carpeta}: {e}")
            
            pbar_general.update(1)

    print("\n--- Resumen de la Ejecución ---")
    print(f"✅ Análisis completado.")
    print(f"Todos los resultados han sido guardados en la carpeta: {directorio_base_run}")


if __name__ == "__main__":
    principal()