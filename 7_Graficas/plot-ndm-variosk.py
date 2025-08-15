import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path

# --- CONFIGURACIÓN ---

# Directorio donde están tus archivos CSV
directorio_datos = '/home/tdelorenzi/1_yolo/1_segregacion/6_resultados/analisis_sensibilidad_k'

# Donde guardar el gráfico
ruta_salida = '/home/tdelorenzi/1_yolo/1_segregacion/7_Graficas/resultados-ndm/grafico_comparativo_sensibilidad_k_suavizado.png'

# Parámetros del gráfico
limite_tiempo = 300
ventana_suavizado = 60

# --- CÓDIGO PRINCIPAL ---

# Buscar todos los archivos CSV de datos
patron_archivos = 'datos_segregacion_k_*.csv'
ruta_busqueda = os.path.join(directorio_datos, patron_archivos)
lista_archivos = sorted(glob.glob(ruta_busqueda))

if not lista_archivos:
    print(f"Error: No se encontraron archivos CSV en el directorio '{directorio_datos}'")
else:
    print(f"Archivos encontrados: {len(lista_archivos)}")
    
    # Crear la figura
    fig, ax = plt.subplots(figsize=(12, 8))
    archivos_exitosos = 0

    for archivo in lista_archivos:
        try:
            print(f"Procesando: {os.path.basename(archivo)}")
            
            # Leer el archivo CSV
            df = pd.read_csv(archivo)
            
            # Verificar que tiene las columnas necesarias
            if 'tiempo' not in df.columns or 'indice_ndm' not in df.columns:
                print(f"  ⚠️  Archivo sin columnas 'tiempo' e 'indice_ndm': {os.path.basename(archivo)}")
                continue
            
            # Filtrar por tiempo
            df_filtrado = df[df['tiempo'] <= limite_tiempo].copy()
            
            if len(df_filtrado) == 0:
                print(f"  ⚠️  No hay datos dentro del límite de tiempo para {os.path.basename(archivo)}")
                continue
            
            # Aplicar suavizado
            df_filtrado['ndm_suavizado'] = df_filtrado['indice_ndm'].rolling(
                window=ventana_suavizado, 
                min_periods=1
            ).mean()
            
            # Extraer el valor de k del nombre del archivo
            try:
                k_valor = os.path.basename(archivo).split('_')[-1].split('.')[0]
                etiqueta = f'k = {k_valor}'
            except:
                etiqueta = os.path.basename(archivo)
            
            # Plotear la curva suavizada
            ax.plot(
                df_filtrado['tiempo'], 
                df_filtrado['ndm_suavizado'], 
                label=etiqueta,
                linewidth=2
            )
            
            archivos_exitosos += 1
            print(f"  ✅ {len(df_filtrado)} puntos procesados y suavizados")
            
        except Exception as e:
            print(f"  ❌ Error con {os.path.basename(archivo)}: {str(e)}")
            continue

    if archivos_exitosos > 0:
        # Configurar el gráfico
        ax.set_title('Análisis de Sensibilidad del Parámetro k', fontsize=16)
        ax.set_xlabel('Tiempo (s)', fontsize=12)
        ax.set_ylabel('Índice NDM (suavizado)', fontsize=12)
        ax.legend(title='Valor de k')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        # Crear directorio de salida si no existe
        directorio_salida = os.path.dirname(ruta_salida)
        os.makedirs(directorio_salida, exist_ok=True)
        
        # Guardar gráfico
        plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
        print(f"\n✅ Gráfico guardado en: '{ruta_salida}'")
        print(f"✅ Curvas procesadas exitosamente: {archivos_exitosos}/{len(lista_archivos)}")
        
        # Mostrar el gráfico
        plt.show()
        
    else:
        print("\n❌ No se pudo procesar ningún archivo correctamente")
        
    print("\n--- Graficado Completado ---")