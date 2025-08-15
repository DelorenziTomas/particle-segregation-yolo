import cv2
import math
import copy # Importado para copiar configuraciones
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
    # Directorio base para todos los resultados del estudio
    "directorio_salida_base": Path('/home/tdelorenzi/1_yolo/1_segregacion/6_resultados/1_variacion-de-anillos'),
    "informacion_clases": {
        'nombres': ['convexa-pmma', 'no-convexa1'],
        'colores': [(0, 255, 0), (0, 0, 255)]  # Verde y Rojo
    },
    "conteos_particulas": {
        'total_discos': 193,
        'total_1corte': 174
    },
    "parametros_analisis": {
        # Define el rango del estudio: (máximo_anillos, mínimo_anillos).
        # El código iterará desde max hasta min (incluyéndolos).
        'rango_anillos': (9, 3), # Ejemplo: estudiará para 9, 8, 7, 6, 5, 4 y 3 anillos.
        'num_regiones': 9 # Valor inicial, será sobreescrito en cada ciclo
    }
}

class AnalizadorSegregacion:
    """
    Encapsula toda la lógica para el análisis de segregación de partículas.
    Modificado para manejar directorios de salida dinámicos por ciclo de análisis.
    """
    def __init__(self, configuracion_ciclo: Dict[str, Any], mostrar_etiquetas: bool):
        self.configuracion = configuracion_ciclo
        self.modelos = self._cargar_modelos()
        self.mostrar_etiquetas = mostrar_etiquetas

        # Crear directorios de salida específicos para este ciclo
        self.directorio_salida_ciclo = self.configuracion["directorio_salida"]
        self.ruta_dir_videos = self.directorio_salida_ciclo / 'videos'
        self.ruta_dir_tablas = self.directorio_salida_ciclo / 'tablas'
        self.ruta_dir_graficas = self.directorio_salida_ciclo / 'graficas'
        
        for d in [self.ruta_dir_videos, self.ruta_dir_tablas, self.ruta_dir_graficas]:
            d.mkdir(parents=True, exist_ok=True)

        # Inicialización de parámetros del video y análisis
        self.captura = cv2.VideoCapture(str(self.configuracion["ruta_video"]))
        if not self.captura.isOpened():
            raise IOError(f"Error al abrir el video: {self.configuracion['ruta_video']}")

        self.ancho = int(self.captura.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.alto = int(self.captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.captura.get(cv2.CAP_PROP_FPS)
        self.centro = (self.ancho // 2, self.alto // 2)
        self.radio_maximo = min(self.centro)
        self.num_regiones = self.configuracion["parametros_analisis"]["num_regiones"]
        
        # Parámetros de segregación
        total_discos = self.configuracion["conteos_particulas"]["total_discos"]
        total_1corte = self.configuracion["conteos_particulas"]["total_1corte"]
        self.x_o = total_1corte / (total_discos + total_1corte)

        # Contenedores de estado
        self.datos_frames: List[Dict[str, Any]] = []
        self.total_frames = 0
        
        # Configuración de salida de video
        nombre_base_archivo = f"analisis_{self.num_regiones}-anillos"
        self.ruta_video_salida = self.ruta_dir_videos / f'{nombre_base_archivo}.mp4'
        codificador = cv2.VideoWriter_fourcc(*'mp4v')
        self.escritor = cv2.VideoWriter(str(self.ruta_video_salida), codificador, self.fps, (self.ancho, self.alto))

    def _cargar_modelos(self) -> Dict[str, YOLO]:
        """Carga los modelos YOLO desde las rutas especificadas en la configuración."""
        return {
            nombre: YOLO(ruta) for nombre, ruta in self.configuracion["rutas_modelos"].items()
        }
        
    def _detectar_particulas(self, frame: np.ndarray, modelo: YOLO, id_clase: int, umbral_conf: float) -> List[Dict]:
        """Función auxiliar para detectar partículas de un tipo específico."""
        detecciones = []
        resultados = modelo.predict(frame, conf=umbral_conf, verbose=False)
        
        for caja in resultados[0].boxes:
            x1, y1, x2, y2 = map(int, caja.xyxy[0])
            centro_p = ((x1 + x2) // 2, (y1 + y2) // 2)
            distancia = math.sqrt((centro_p[0] - self.centro[0])**2 + (centro_p[1] - self.centro[1])**2)
            region = min(int(distancia / (self.radio_maximo / self.num_regiones)), self.num_regiones - 1)
            
            detecciones.append({
                'caja': (x1, y1, x2, y2),
                'confianza': float(caja.conf[0]),
                'id_clase': id_clase,
                'region': region
            })
        return detecciones

    def _dibujar_anotaciones(self, frame: np.ndarray, detecciones: List[Dict], si_actual: float, conteos: np.ndarray):
        """Dibuja todas las anotaciones en el frame (anillos, puntos, texto)."""
        # Dibujar anillos
        for i in range(self.num_regiones, 0, -1):
            radio = int(i * (self.radio_maximo / self.num_regiones))
            cv2.circle(frame, self.centro, radio, (200, 200, 200), 1)

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
            f"Anillos: {self.num_regiones}",
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
        
        detecciones_discos = self._detectar_particulas(frame, self.modelos['convexa-pmma'], 0, self.configuracion['umbrales_confianza']['convexa-pmma'])
        detecciones_estrellas = self._detectar_particulas(frame, self.modelos['no-convexa1'], 1, self.configuracion['umbrales_confianza']['no-convexa1'])
        todas_detecciones = detecciones_discos + detecciones_estrellas
        
        for det in todas_detecciones:
            conteos_region_actual[det['region'], det['id_clase']] += 1
        
        total_particulas = conteos_region_actual.sum(axis=1)
        x_i = np.where(total_particulas > 0, conteos_region_actual[:,1] / total_particulas, 0)
        
        regiones_validas = total_particulas > 0
        si_actual = 0
        if np.any(regiones_validas):
            promedio_xi_xo = np.mean(x_i[regiones_validas] / self.x_o) if self.x_o > 0 else 0
            if len(x_i[regiones_validas]) > 0:
                si_actual = np.sqrt(np.sum(((x_i[regiones_validas] / self.x_o - promedio_xi_xo) ** 2) / len(x_i[regiones_validas])))

        self._almacenar_datos_frame(si_actual, conteos_region_actual, x_i, total_particulas)
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
        print(f"\nProcesando video para {self.num_regiones} anillos... (Presione 'q' en la ventana para saltar este ciclo)")
        cv2.namedWindow("Detección con Anillos Concéntricos", cv2.WINDOW_NORMAL)
        
        while self.captura.isOpened():
            ret, frame = self.captura.read()
            if not ret:
                break
            
            frame_anotado = self.procesar_frame(frame)
            
            cv2.imshow("Detección con Anillos Concéntricos", frame_anotado)
            self.escritor.write(frame_anotado)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Ciclo saltado por el usuario.")
                break
        
        print(f"Procesamiento de video para {self.num_regiones} anillos finalizado.")
        self._limpiar_recursos()

    def guardar_resultados(self):
        """Guarda los datos en CSV y genera los gráficos en sus carpetas respectivas."""
        if not self.datos_frames:
            print("No se generaron datos, no se guardarán resultados.")
            return
            
        nombre_base_archivo = f"analisis_{self.num_regiones}-anillos"

        # Guardar CSV en la carpeta 'tablas'
        ruta_csv = self.ruta_dir_tablas / f'{nombre_base_archivo}_segregation_data.csv'
        df = pd.DataFrame(self.datos_frames)
        df.to_csv(ruta_csv, index=False)
        print(f"\nDatos de segregación guardados en: {ruta_csv}")

        # Generar gráficos en la carpeta 'graficas'
        plt.style.use('seaborn-v0_8-whitegrid')

        # Gráfico de evolución del SI
        plt.figure(figsize=(12, 6))
        plt.plot(df['tiempo'], df['SI'], label=f'Índice de Segregación (SI) - {self.num_regiones} anillos', color='royalblue')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('SI')
        plt.title(f'Evolución del Índice de Segregación - {self.num_regiones} Anillos')
        plt.legend()
        plt.savefig(self.ruta_dir_graficas / f'{nombre_base_archivo}_si_evolution.png')
        plt.close()

        # Gráfico de distribución radial promedio
        fraccion_promedio = [df[f'region_{i}_fraccion'].mean() for i in range(self.num_regiones)]
        plt.figure(figsize=(12, 6))
        plt.bar(range(1, self.num_regiones + 1), fraccion_promedio, color='skyblue', label='Fracción promedio observada')
        plt.axhline(y=self.x_o, color='r', linestyle='--', label=f'Fracción inicial $x_o$ = {self.x_o:.3f}')
        plt.xlabel(f'Región (1=centro, {self.num_regiones}=borde)')
        plt.ylabel('Fracción promedio de "no-convexa1"')
        plt.title(f'Distribución Radial Promedio - {self.num_regiones} Anillos')
        plt.xticks(range(1, self.num_regiones + 1))
        plt.legend()
        plt.savefig(self.ruta_dir_graficas / f'{nombre_base_archivo}_radial_distribution.png')
        plt.close()

        print(f"Gráficos guardados en: {self.ruta_dir_graficas}")

    def _limpiar_recursos(self):
        """Libera recursos de video."""
        self.captura.release()
        self.escritor.release()
        cv2.destroyAllWindows()
        print(f"Video de salida guardado en: {self.ruta_video_salida}\n")

def obtener_preferencia_usuario() -> bool:
    """Función global para obtener la preferencia de visualización del usuario una sola vez."""
    print("\nOpciones de visualización para todo el estudio:")
    print("1. Mostrar puntos con etiquetas y confianza")
    print("2. Mostrar solo puntos (sin texto)")
    while True:
        modo_visual = input("Seleccione el modo de visualización (1/2): ").strip()
        if modo_visual in ['1', '2']:
            return modo_visual == '1'
        print("Opción no válida. Por favor, ingrese 1 o 2.")

def principal():
    """
    Punto de entrada principal del script.
    Orquesta el estudio iterando sobre el número de anillos.
    """
    try:
        # Preguntar preferencias una sola vez
        mostrar_etiquetas = obtener_preferencia_usuario()
        
        # Obtener el rango de anillos del estudio desde la configuración
        max_anillos, min_anillos = CONFIGURACION["parametros_analisis"]["rango_anillos"]
        
        # Bucle de estudio: itera desde el máximo al mínimo de anillos
        for num_anillos_actual in range(max_anillos, min_anillos - 1, -1):
            print(f"\n{'='*60}")
            print(f"INICIANDO CICLO DE ANÁLISIS PARA {num_anillos_actual} ANILLOS")
            print(f"{'='*60}")
            
            # Crear una configuración específica para este ciclo
            config_ciclo = copy.deepcopy(CONFIGURACION)
            config_ciclo["parametros_analisis"]["num_regiones"] = num_anillos_actual
            # Establecer el directorio de salida para este ciclo específico
            config_ciclo["directorio_salida"] = CONFIGURACION["directorio_salida_base"] / f"{num_anillos_actual}_anillos"
            
            # Ejecutar el análisis para la configuración actual
            analizador = AnalizadorSegregacion(config_ciclo, mostrar_etiquetas)
            analizador.ejecutar_analisis()
            analizador.guardar_resultados()

        print(f"\n{'='*60}")
        print("Estudio paramétrico completado.")
        print(f"Todos los resultados se han guardado en el directorio base: {CONFIGURACION['directorio_salida_base']}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"Ha ocurrido un error inesperado durante el estudio: {e}")

if __name__ == "__main__":
    principal()