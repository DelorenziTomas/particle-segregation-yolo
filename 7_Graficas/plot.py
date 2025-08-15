import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Configuración de Archivos y Gráfica ---

# Rutas de los archivos de datos de entrada
file1_path = '/home/tdelorenzi/1_yolo/1_segregacion/6_resultados/2025-08-04-reticula_polar_ref_convexa/datos_segregacion_ref_convexa.csv'
file2_path = '/home/tdelorenzi/1_yolo/1_segregacion/6_resultados/2025-08-06-pmmavspmma/segregation_data_polar_grid.csv'

# Directorio de salida para la gráfica
output_dir = '/home/tdelorenzi/1_yolo/1_segregacion/7_Graficas/resultados'
output_filename = 'comparacion_indice_segregacion_final.png'

# Etiquetas para la leyenda de la gráfica
label_file1 = r'07/08/2025 - 245 particulas convexas (123 azules, 122 blancas), 0 particulas no convexas, $x_0 = \frac{123}{123+122} = 0.502$'
label_file2 = r'04/08/2025 - 193 particulas convexas, 174 particulas no convexas, $x_0 = \frac{193}{193+174} = 0.525$'

# Parámetro de suavizado
window_size = 65

# --- Procesamiento y Graficación ---

os.makedirs(output_dir, exist_ok=True)

try:
    # Carga y Limpieza de Datos
    df1 = pd.read_csv(file1_path, header=None, usecols=[1, 2], names=['tiempo', 'si'])
    df1['tiempo'] = pd.to_numeric(df1['tiempo'], errors='coerce')
    df1['si'] = pd.to_numeric(df1['si'], errors='coerce')

    df2 = pd.read_csv(file2_path, header=None, usecols=[1, 2], names=['tiempo', 'si'])
    df2['tiempo'] = pd.to_numeric(df2['tiempo'], errors='coerce')
    df2['si'] = pd.to_numeric(df2['si'], errors='coerce')

    df1.dropna(inplace=True)
    df2.dropna(inplace=True)

    # Truncar los datos al tiempo máximo de la curva más corta
    max_time_1 = df1['tiempo'].max()
    max_time_2 = df2['tiempo'].max()
    cutoff_time = min(max_time_1, max_time_2)
    df1 = df1[df1['tiempo'] <= cutoff_time]
    df2 = df2[df2['tiempo'] <= cutoff_time]
    
    print(f"Ambas gráficas se plotearán hasta el tiempo máximo común: {cutoff_time:.2f} segundos.")
    
    # --- 👇 AQUÍ SE AÑADE EL CÁLCULO DE LAS MEDIAS 👇 ---

    # 1. Calcular el valor medio del SI para t >= 70s para cada curva
    mean_si_1 = df1[df1['tiempo'] >= 70]['si'].mean()
    mean_si_2 = df2[df2['tiempo'] >= 70]['si'].mean()
    
    print(f"Valor medio (t>=70s) para '{label_file1[:20]}...': {mean_si_1:.4f}")
    print(f"Valor medio (t>=70s) para '{label_file2[:20]}...': {mean_si_2:.4f}")

    # --- FIN DE LA SECCIÓN DE CÁLCULO ---

    # Calcular el promedio móvil sobre los datos truncados
    df1['si_suavizado'] = df1['si'].rolling(window=window_size, center=True).mean()
    df2['si_suavizado'] = df2['si'].rolling(window=window_size, center=True).mean()

    # Configurar el estilo
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(14, 8))

    # --- 👇 AQUÍ SE PLOTEAN LAS LÍNEAS PUNTEADAS 👇 ---
    
    # Plotear los datos suavizados (curvas principales)
    # Se guarda la referencia del color de cada línea (C0 para la primera, C1 para la segunda)
    line1, = ax.plot(df1['tiempo'], df1['si_suavizado'], label=label_file1, linewidth=2.5)
    line2, = ax.plot(df2['tiempo'], df2['si_suavizado'], label=label_file2, linewidth=2.5)
    
    # Plotear las líneas de promedio usando el mismo color que la curva correspondiente
    ax.axhline(y=mean_si_1, color=line1.get_color(), linestyle='--', linewidth=1.5, label=f'Media (07/08/2025): {mean_si_1:.3f}')
    ax.axhline(y=mean_si_2, color=line2.get_color(), linestyle='--', linewidth=1.5, label=f'Media (04/08/2025): {mean_si_2:.3f}')

    # --- FIN DE LA SECCIÓN DE PLOTEO ---

    # Configurar Títulos
    fig.suptitle('Evolución Temporal del Índice de Segregación', fontsize=20, fontweight='bold')
    ax.set_title('Condiciones del experimento: pwm=70 ; 10rpm', fontsize=16, pad=10)

    # Configurar etiquetas con letras grandes
    ax.set_xlabel('Tiempo (s)', fontsize=16)
    ax.set_ylabel('Índice de Segregación (SI)', fontsize=16)
    ax.legend(fontsize=15) # Reducido un poco para que quepan las nuevas etiquetas
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Ajustar layout
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Guardar la gráfica
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    print(f"✅ Gráfica final con líneas de promedio guardada en: {output_path}")

except FileNotFoundError as e:
    print(f"❌ Error: No se pudo encontrar el archivo: {e.filename}")
except Exception as e:
    print(f"❌ Ocurrió un error inesperado: {e}")