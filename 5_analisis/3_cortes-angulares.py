import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, Any, List

# -- 1. CONFIGURACIÓN CENTRALIZADA --
# Agrupar todas las configuraciones en un diccionario para fácil acceso y modificación.
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
    "directorio_salida": Path('/home/tdelorenzi/1_yolo/1_segregacion/6_resultados/2025-08-04_angular'), # Directorio de salida modificado
    "informacion_clases": {
        'nombres': ['convexa-pmma', 'no-convexa1'],
        'colores': [(0, 255, 0), (0, 0, 255)]  # Verde y Rojo
    },
    "conteos_particulas": {
        'total_discos': 193,
        'total_1corte': 174
    },
    "parametros_analisis": {
        'num_regiones': 12  # Ahora representa el número de "porciones de pizza" o sectores angulares
    }
}

class AnalizadorSegregacion:
    """
    Encapsula toda la lógica para el análisis de segregación de partículas.
    Esto evita el uso de variables globales y organiza el código.
    """
    def __init__(self, configuracion: Dict[str, Any]):
        self.configuracion = configuracion
        self.modelos = self._cargar_modelos()
        self.mostrar_etiquetas = self._obtener_preferencia_usuario()

        # Inicialización de parámetros del video y análisis
        self.captura = cv2.VideoCapture(str(configuracion["ruta_video"]))
        if not self.captura.isOpened():
            raise IOError(f"Error al abrir el video: {configuracion['ruta_video']}")

        self.ancho = int(self.captura.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.alto = int(self.captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.captura.get(cv2.CAP_PROP_FPS)
        self.centro = (self.ancho // 2, self.alto // 2)
        self.radio_maximo = min(self.centro)
        self.num_regiones = configuracion["parametros_analisis"]["num_regiones"]
        self.angulo_por_region = 360 / self.num_regiones

        # Parámetros de segregación (sin cambios en el cálculo)
        total_discos = configuracion["conteos_particulas"]["total_discos"]
        total_1corte = configuracion["conteos_particulas"]["total_1corte"]
        self.x_o = total_1corte / (total_discos + total_1corte)

        # Contenedores de estado
        self.datos_frames: List[Dict[str, Any]] = []
        self.total_frames = 0
         
        # Configuración de salida
        self.configuracion["directorio_salida"].mkdir(parents=True, exist_ok=True)
        self.ruta_video_salida = self.configuracion["directorio_salida"] / '08-07-2025-segregacion_angular.mp4'
        codificador = cv2.VideoWriter_fourcc(*'mp4v')
        self.escritor = cv2.VideoWriter(str(self.ruta_video_salida), codificador, self.fps, (self.ancho, self.alto))

    def _cargar_modelos(self) -> Dict[str, YOLO]:
        """Carga los modelos YOLO desde las rutas especificadas en la configuración."""
        return {
            nombre: YOLO(ruta) for nombre, ruta in self.configuracion["rutas_modelos"].items()
        }

    def _obtener_preferencia_usuario(self) -> bool:
        """Obtiene la preferencia de visualización del usuario."""
        print("\nOpciones de visualización:")
        print("1. Mostrar puntos con etiquetas y confianza")
        print("2. Mostrar solo puntos (sin texto)")
        modo_visual = input("Seleccione el modo de visualización (1/2): ").strip()
        return modo_visual == '1'
         
    def _detectar_particulas(self, frame: np.ndarray, modelo: YOLO, id_clase: int, umbral_conf: float) -> List[Dict]:
        """Función auxiliar para detectar partículas y asignarles una región angular."""
        detecciones = []
        resultados = modelo.predict(frame, conf=umbral_conf, verbose=False)
         
        for caja in resultados[0].boxes:
            x1, y1, x2, y2 = map(int, caja.xyxy[0])
            centro_p = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # --- MODIFICACIÓN CLAVE: CÁLCULO DE REGIÓN ANGULAR ---
            # Calcular el ángulo de la partícula con respecto al centro
            dx = centro_p[0] - self.centro[0]
            dy = centro_p[1] - self.centro[1]
            angulo_rad = math.atan2(dy, dx)
            
            # Convertir ángulo a grados [0, 360]
            angulo_deg = (math.degrees(angulo_rad) + 360) % 360
            
            # Asignar la región basada en el ángulo
            region = int(angulo_deg / self.angulo_por_region)
             
            detecciones.append({
                'caja': (x1, y1, x2, y2),
                'confianza': float(caja.conf[0]),
                'id_clase': id_clase,
                'region': region
            })
        return detecciones

    def _dibujar_anotaciones(self, frame: np.ndarray, detecciones: List[Dict], si_actual: float, conteos: np.ndarray):
        """Dibuja todas las anotaciones en el frame (sectores, puntos, texto)."""
        # --- MODIFICACIÓN CLAVE: DIBUJAR LÍNEAS DE SECTORES ANGULARES ---
        # Dibujar el círculo exterior
        cv2.circle(frame, self.centro, self.radio_maximo, (200, 200, 200), 1)
        # Dibujar las líneas que definen los sectores
        for i in range(self.num_regiones):
            angulo_linea = math.radians(i * self.angulo_por_region)
            punto_final = (
                self.centro[0] + int(self.radio_maximo * math.cos(angulo_linea)),
                self.centro[1] + int(self.radio_maximo * math.sin(angulo_linea))
            )
            cv2.line(frame, self.centro, punto_final, (200, 200, 200), 1)

        # Dibujar puntos en el centro de las detecciones
        for det in detecciones:
            x1, y1, x2, y2 = det['caja']
            centro_p = ((x1 + x2) // 2, (y1 + y2) // 2)
            id_clase = det['id_clase']
            color = self.configuracion['informacion_clases']['colores'][id_clase]
            cv2.circle(frame, centro_p, 3, color, -1)
             
            if self.mostrar_etiquetas:
                etiqueta = f"{self.configuracion['informacion_clases']['nombres'][id_clase]} {det['confianza']:.2f}"
                cv2.putText(frame, etiqueta, (centro_p[0] - 30, centro_p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Dibujar texto de información
        texto_info = [
            f"1corte: {int(conteos[:,1].sum())}/{self.configuracion['conteos_particulas']['total_1corte']}",
            f"Discos: {int(conteos[:,0].sum())}/{self.configuracion['conteos_particulas']['total_discos']}",
            f"SI: {si_actual:.3f} (x_o={self.x_o:.3f})",
        ]
        for i, texto in enumerate(texto_info):
            cv2.putText(frame, texto, (20, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
         
        return frame

    def procesar_frame(self, frame: np.ndarray) -> np.ndarray:
        """Procesa un único frame: detecta, calcula métricas y prepara para visualización."""
        self.total_frames += 1
        conteos_region_actual = np.zeros((self.num_regiones, 2))
         
        # Detección para ambos modelos
        detecciones_discos = self._detectar_particulas(frame, self.modelos['convexa-pmma'], 0, self.configuracion['umbrales_confianza']['convexa-pmma'])
        detecciones_estrellas = self._detectar_particulas(frame, self.modelos['no-convexa1'], 1, self.configuracion['umbrales_confianza']['no-convexa1'])
        todas_detecciones = detecciones_discos + detecciones_estrellas
         
        # Actualizar contadores por región
        for det in todas_detecciones:
            # Asegurarse de que la región esté dentro de los límites
            if 0 <= det['region'] < self.num_regiones:
                conteos_region_actual[det['region'], det['id_clase']] += 1
         
        # Cálculo de métricas del frame (la lógica no cambia)
        total_particulas = conteos_region_actual.sum(axis=1)
        x_i = np.where(total_particulas > 0, conteos_region_actual[:,1] / total_particulas, 0)
         
        regiones_validas = total_particulas > 0
        si_actual = 0
        if np.any(regiones_validas):
            promedio_xi_xo = np.mean(x_i[regiones_validas] / self.x_o) if self.x_o > 0 else 0
            si_actual = np.sqrt(np.sum(((x_i[regiones_validas] / self.x_o - promedio_xi_xo) ** 2) / len(x_i[regiones_validas]))) if self.x_o > 0 else 0

        # Almacenar datos del frame
        self._almacenar_datos_frame(si_actual, conteos_region_actual, x_i, total_particulas)
         
        # Anotar el frame para visualización
        frame_anotado = self._dibujar_anotaciones(frame.copy(), todas_detecciones, si_actual, conteos_region_actual)
        return frame_anotado

    def _almacenar_datos_frame(self, si, conteos, x_i, total_particulas):
        """Almacena los datos calculados para el frame actual."""
        entrada_frame = {
            'frame': self.total_frames,
            'tiempo': self.total_frames / self.fps,
            'SI': si,
            'total_estrellas': int(conteos[:,1].sum()),
            'total_discos': int(conteos[:,0].sum())
        }
        for i in range(self.num_regiones):
            entrada_frame[f'region_{i}_estrellas'] = int(conteos[i, 1])
            entrada_frame[f'region_{i}_discos'] = int(conteos[i, 0])
            entrada_frame[f'region_{i}_fraccion'] = x_i[i] if total_particulas[i] > 0 else 0
        self.datos_frames.append(entrada_frame)

    def ejecutar_analisis(self):
        """Ejecuta el bucle principal de procesamiento de video."""
        print("\nProcesando video... (Presione 'q' para detener)")
        # --- MODIFICACIÓN: Título de la ventana ---
        cv2.namedWindow("Detección con Regiones Angulares", cv2.WINDOW_NORMAL)
         
        while self.captura.isOpened():
            ret, frame = self.captura.read()
            if not ret:
                break
             
            frame_anotado = self.procesar_frame(frame)
             
            cv2.imshow("Detección con Regiones Angulares", frame_anotado)
            self.escritor.write(frame_anotado)
             
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
         
        print("\nProcesamiento de video finalizado.")
        self._limpiar_recursos()

    def guardar_resultados(self):
        """Guarda los datos en CSV y genera los gráficos."""
        # Guardar CSV
        ruta_csv = self.configuracion["directorio_salida"] / '08-07-2025-segregation_data_angular.csv'
        df = pd.DataFrame(self.datos_frames)
        df.to_csv(ruta_csv, index=False)
        print(f"\nDatos de segregación guardados en: {ruta_csv}")

        # Generar gráficos
        plt.style.use('seaborn-v0_8-whitegrid')

        # Gráfico de evolución del SI
        plt.figure(figsize=(12, 6))
        plt.plot(df['tiempo'], df['SI'], label='Índice de Segregación (SI)', color='royalblue')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('SI')
        plt.title('Evolución del Índice de Segregación en el Tiempo')
        plt.legend()
        plt.savefig(self.configuracion["directorio_salida"] / '08-07-2025_si_evolution_angular.png')
        plt.close()

        # --- MODIFICACIÓN: Gráfico de distribución angular ---
        fraccion_promedio = [df[f'region_{i}_fraccion'].mean() for i in range(self.num_regiones)]
        angulos_regiones = [f'{int(i * self.angulo_por_region)}-{int((i+1) * self.angulo_por_region)}°' for i in range(self.num_regiones)]
        
        plt.figure(figsize=(12, 6))
        plt.bar(angulos_regiones, fraccion_promedio, color='skyblue', label='Fracción promedio observada')
        plt.axhline(y=self.x_o, color='r', linestyle='--', label=f'Fracción inicial $x_o$ = {self.x_o:.3f}')
        plt.xlabel('Sector Angular (grados)')
        plt.ylabel('Fracción promedio de "no-convexa1"')
        plt.title('Distribución Angular Promedio de Partículas')
        plt.xticks(rotation=45, ha="right") # Rotar etiquetas para mejor visualización
        plt.legend()
        plt.tight_layout() # Ajustar layout para que no se corten las etiquetas
        plt.savefig(self.configuracion["directorio_salida"] / '08-07-2025_angular_distribution.png')
        plt.close()

        print(f"Gráficos guardados en: {self.configuracion['directorio_salida']}")

    def _limpiar_recursos(self):
        """Libera recursos de video."""
        self.captura.release()
        self.escritor.release()
        cv2.destroyAllWindows()
        print(f"\nVideo de salida guardado en: {self.ruta_video_salida}")


def principal():
    """Punto de entrada principal del script."""
    try:
        analizador = AnalizadorSegregacion(CONFIGURACION)
        analizador.ejecutar_analisis()
        analizador.guardar_resultados()
    except Exception as e:
        print(f"Ha ocurrido un error: {e}")

if __name__ == "__main__":
    principal()