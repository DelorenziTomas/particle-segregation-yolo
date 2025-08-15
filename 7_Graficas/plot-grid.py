import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generar_grafica_combinada():\


    """
    Genera un único gráfico combinando todas las curvas de segregación vs. tiempo,
    con colores del arcoíris, eje Y ajustado, transparencia y una leyenda más grande.
    """
    # --- Configuración de Directorios ---
    directorio_base = os.path.expanduser('~/1_yolo/1_segregacion/6_resultados/2025-08-04_analisis_polar_iterativo')
    directorio_salida = os.path.expanduser('~/1_yolo/1_segregacion/7_Graficas/resultados')

    # --- Creación del Directorio de Salida ---
    try:
        os.makedirs(directorio_salida, exist_ok=True)
    except OSError as e:
        print(f"Error al crear el directorio de salida: {e}")
        return

    # --- 1. Identificar todos los archivos a procesar ANTES de graficar ---
    try:
        subdirectorios = sorted(os.listdir(directorio_base))
    except FileNotFoundError:
        print(f"Error: No se encontró el directorio base en '{directorio_base}'")
        return
        
    archivos_a_procesar = []
    for nombre_subdir in subdirectorios:
        ruta_completa_subdir = os.path.join(directorio_base, nombre_subdir)
        if os.path.isdir(ruta_completa_subdir):
            ruta_csv = os.path.join(ruta_completa_subdir, 'segregation_data_polar_grid.csv')
            if os.path.exists(ruta_csv):
                # Guardamos la ruta del csv y el nombre para la leyenda
                archivos_a_procesar.append((ruta_csv, nombre_subdir))

    if not archivos_a_procesar:
        print("No se encontraron archivos CSV para procesar. Saliendo.")
        return

    # Se crea la paleta de colores del arcoíris
    num_curvas = len(archivos_a_procesar)
    mapa_colores = plt.cm.rainbow
    colores = mapa_colores(np.linspace(0, 1, num_curvas))
    
    print(f"Se encontraron {num_curvas} configuraciones. Se generará una paleta de {num_curvas} colores.")

    # --- 2. Inicialización de la Gráfica y variables ---
    plt.figure(figsize=(15, 10))
    min_y_global = float('inf')
    max_y_global = float('-inf')

    # --- 3. Procesamiento y Graficado ---
    for i, (ruta_csv, nombre_subdir) in enumerate(archivos_a_procesar):
        try:
            df = pd.read_csv(ruta_csv)
            tiempo = df.iloc[:, 1]
            indice_segregacion = df.iloc[:, 2]

            # (Opcional) Suavizado
            indice_segregacion = indice_segregacion.rolling(window=70, min_periods=1).mean()

            etiqueta = nombre_subdir.replace('_', ' ').replace('anillos', 'A').replace('sectores', 'S')
            
            plt.plot(tiempo, indice_segregacion, linewidth=1.5, label=etiqueta, alpha=0.55, color=colores[i])
            
            if not indice_segregacion.empty:
                min_actual = indice_segregacion.min()
                max_actual = indice_segregacion.max()
                if min_actual < min_y_global:
                    min_y_global = min_actual
                if max_actual > max_y_global:
                    max_y_global = max_actual

            print(f"✔️ Curva añadida: {etiqueta}")

        except Exception as e:
            print(f"❌ ERROR al procesar '{ruta_csv}': {e}")
    
    # --- 4. Personalización y Guardado Final ---
    plt.title('Comparativa de Índices de Segregación vs. Tiempo', fontsize=18)
    plt.xlabel('Tiempo (s)', fontsize=14)
    plt.ylabel('Índice de Segregación', fontsize=14)
    
    # <--- MODIFICACIÓN: Se establece el límite del eje X (horizontal) ---
    plt.xlim(0, 600)
    
    # Se ajusta el eje Y
    if min_y_global != float('inf') and max_y_global != float('-inf'):
        rango = max_y_global - min_y_global
        margen = rango * 0.05
        plt.ylim(min_y_global - margen, max_y_global + margen)

    plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.7, color='gray')
    
    plt.legend(title="Configuración (Anillos / Sectores)", fontsize=12, ncol=2)
    
    plt.tight_layout()

    # Guarda la figura
    nombre_archivo_salida = 'comparativa_analisis_polar_colores_600s.png'
    ruta_salida = os.path.join(directorio_salida, nombre_archivo_salida)
    
    plt.savefig(ruta_salida, dpi=300)
    plt.close()

    print("\n¡Proceso completado!")
    print(f"⏱️ Gráfica con eje X hasta 600s guardada en: {ruta_salida}")

# --- Ejecución del Script ---
if __name__ == '__main__':
    generar_grafica_combinada()