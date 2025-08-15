import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuración (sin cambios) ---
file1_path = '/home/tdelorenzi/1_yolo/1_segregacion/6_resultados/2025-08-04-reticula_polar_ref_convexa/datos_segregacion_ref_convexa.csv'
file2_path = '/home/tdelorenzi/1_yolo/1_segregacion/6_resultados/2025-08-06-pmmavspmma/segregation_data_polar_grid.csv'
output_dir = '/home/tdelorenzi/1_yolo/1_segregacion/7_Graficas/resultados'
output_filename = 'ratio_suavizado_por_intervalos.png'

# --- Procesamiento y Graficación ---
os.makedirs(output_dir, exist_ok=True)

try:
    # 1. Carga y Limpieza de Datos (sin cambios)
    df1 = pd.read_csv(file1_path, header=None, usecols=[1, 2], names=['tiempo', 'si_ref'])
    df1['tiempo'] = pd.to_numeric(df1['tiempo'], errors='coerce')
    df1['si_ref'] = pd.to_numeric(df1['si_ref'], errors='coerce')

    df2 = pd.read_csv(file2_path, header=None, usecols=[1, 2], names=['tiempo', 'si_pmma'])
    df2['tiempo'] = pd.to_numeric(df2['tiempo'], errors='coerce')
    df2['si_pmma'] = pd.to_numeric(df2['si_pmma'], errors='coerce')

    df1.dropna(inplace=True)
    df2.dropna(inplace=True)

    # 2. Sincronización de los Datos (sin cambios)
    df1.sort_values('tiempo', inplace=True)
    df2.sort_values('tiempo', inplace=True)
    df_merged = pd.merge_asof(df2, df1, on='tiempo')
    
    # 3. Cálculo de la Relación (sin cambios)
    df_merged['ratio_si'] = df_merged['si_pmma'] / df_merged['si_ref']
    df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_merged.dropna(subset=['ratio_si'], inplace=True)

    # --- 👇 AQUÍ ESTÁ EL CAMBIO: PROMEDIAR PUNTOS POR INTERVALOS 👇 ---
    
    # Define el tamaño del intervalo de tiempo en segundos (puedes cambiar este valor)
    bin_size = 1.5  # segundos

    # Crea una columna que asigna cada punto a un intervalo de tiempo
    df_merged['time_bin'] = (df_merged['tiempo'] // bin_size) * bin_size
    
    # Agrupa por intervalo y calcula el promedio del ratio y del tiempo para cada grupo
    df_binned = df_merged.groupby('time_bin').agg(
        ratio_mean=('ratio_si', 'mean'), 
        time_mean=('tiempo', 'mean')
    ).reset_index()

    print(f"Se agruparon los puntos en intervalos de {bin_size} segundos.")
    # --- FIN DE LA MODIFICACIÓN ---

    # 4. Graficación del Scatter Plot con Puntos Promediados
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Graficamos los PUNTOS PROMEDIADOS en lugar de los originales
    ax.scatter(df_binned['time_mean'], df_binned['ratio_mean'], 
               s=50, label=f'Ratio Promedio (cada {bin_size}s)', zorder=10)
    
    # Mantenemos la línea de tendencia general para confirmar el comportamiento
    ratio_suavizado = df_merged['ratio_si'].rolling(window=65, center=True).mean()
    ax.plot(df_merged['tiempo'], ratio_suavizado, color='red', linewidth=2.5, 
            label='Tendencia General (promedio móvil)', alpha=0.8)

    # Línea de referencia en y=1
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1.5, label='Ratio = 1 (SI iguales)')

    # Configuración de la gráfica
    ax.set_title(r'Relación de Índices de Segregación (Promediada por Intervalos): $\frac{SI_{04/08/2025}}{SI_{07/08/2025}}$',fontsize=18, fontweight='bold'), ax.set_xlabel('Tiempo (s)', fontsize=16)
    ax.set_ylabel('Ratio del Índice de Segregación', fontsize=16)
    ax.legend(fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    fig.tight_layout()
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300)

    print(f"✅ Gráfica suavizada guardada exitosamente en: {output_path}")

except Exception as e:
    print(f"❌ Ocurrió un error inesperado: {e}")