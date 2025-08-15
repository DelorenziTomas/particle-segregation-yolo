import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pandas as pd
import time
from tqdm import tqdm
from scipy.stats import gmean

def procesar_video_con_confianza(model, video_path, confianza, fps):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Listas para almacenar resultados
    frames_procesados = []
    detecciones_disco = []
    detecciones_estrella = []
    confianzas_disco = []
    confianzas_estrella = []
    
    for frame_id in tqdm(range(total_frames), desc=f"Conf {confianza:.2f}"):
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.predict(frame, conf=confianza, verbose=False)
        
        conteo_disco = 0
        conteo_estrella = 0
        confs_disco = []
        confs_estrella = []
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                if class_id == 0:  # Disco blanco
                    conteo_disco += 1
                    confs_disco.append(conf)
                else:  # Estrella blanca
                    conteo_estrella += 1
                    confs_estrella.append(conf)
        
        frames_procesados.append(frame_id)
        detecciones_disco.append(conteo_disco)
        detecciones_estrella.append(conteo_estrella)
        confianzas_disco.append(np.mean(confs_disco) if confs_disco else 0)
        confianzas_estrella.append(np.mean(confs_estrella) if confs_estrella else 0)
    
    cap.release()
    
    return {
        'frames': frames_procesados,
        'disco': detecciones_disco,
        'estrella': detecciones_estrella,
        'confianza_disco': confianzas_disco,
        'confianza_estrella': confianzas_estrella,
        'confianza_umbral': confianza,
        'fps': fps
    }

def calcular_metricas_avanzadas(resultados):
    disco = np.array(resultados['disco'])
    estrella = np.array(resultados['estrella'])
    frames = np.array(resultados['frames'])
    conf_disco = np.array(resultados['confianza_disco'])
    conf_estrella = np.array(resultados['confianza_estrella'])
    fps = resultados['fps']
    
    # Cálculos básicos
    delta_disco = np.insert(np.diff(disco), 0, 0)
    delta_estrella = np.insert(np.diff(estrella), 0, 0)
    acumulado_disco = np.cumsum(disco)
    acumulado_estrella = np.cumsum(estrella)
    segundos = frames / fps
    
    # Detecciones por intervalo de tiempo
    intervalo_5s = (segundos // 5).astype(int)
    intervalo_1s = (segundos // 1).astype(int)
    
    # Ratios y porcentajes
    ratio_de = np.divide(disco, estrella + 1e-6)  # Evitar división por cero
    porcentaje_frames_con_disco = np.mean(disco > 0) * 100
    porcentaje_frames_con_estrella = np.mean(estrella > 0) * 100
    
    # Estadísticas de confianza
    confianza_promedio_disco = np.mean(conf_disco[disco > 0]) if np.any(disco > 0) else 0
    confianza_promedio_estrella = np.mean(conf_estrella[estrella > 0]) if np.any(estrella > 0) else 0
    
    # Cambios bruscos (umbral del 50% de cambio)
    cambios_bruscos_disco = np.abs(delta_disco) > (0.5 * np.mean(disco[disco > 0])) if np.any(disco > 0) else np.zeros_like(delta_disco)
    cambios_bruscos_estrella = np.abs(delta_estrella) > (0.5 * np.mean(estrella[estrella > 0])) if np.any(estrella > 0) else np.zeros_like(delta_estrella)
    
    # Agregar todas las métricas al diccionario de resultados
    resultados.update({
        'delta_disco': delta_disco.tolist(),
        'delta_estrella': delta_estrella.tolist(),
        'acumulado_disco': acumulado_disco.tolist(),
        'acumulado_estrella': acumulado_estrella.tolist(),
        'segundos': segundos.tolist(),
        'intervalo_5s': intervalo_5s.tolist(),
        'intervalo_1s': intervalo_1s.tolist(),
        'ratio_de': ratio_de.tolist(),
        'porcentaje_frames_con_disco': porcentaje_frames_con_disco,
        'porcentaje_frames_con_estrella': porcentaje_frames_con_estrella,
        'confianza_promedio_disco': confianza_promedio_disco,
        'confianza_promedio_estrella': confianza_promedio_estrella,
        'cambios_bruscos_disco': cambios_bruscos_disco.astype(int).tolist(),
        'cambios_bruscos_estrella': cambios_bruscos_estrella.astype(int).tolist(),
    })
    
    return resultados

def guardar_csv_detallado(resultados, output_dir):
    confianza = resultados['confianza_umbral']
    datos = {
        'frame': resultados['frames'],
        'segundo': resultados['segundos'],
        'minuto': [s/60 for s in resultados['segundos']],
        'intervalo_5s': resultados['intervalo_5s'],
        'intervalo_1s': resultados['intervalo_1s'],
        'disco_blanco': resultados['disco'],
        'delta_disco': resultados['delta_disco'],
        'acumulado_disco': resultados['acumulado_disco'],
        'confianza_promedio_disco': [resultados['confianza_disco'][i] if resultados['disco'][i] > 0 else 0 for i in range(len(resultados['frames']))],
        'estrella_blanca': resultados['estrella'],
        'delta_estrella': resultados['delta_estrella'],
        'acumulado_estrella': resultados['acumulado_estrella'],
        'confianza_promedio_estrella': [resultados['confianza_estrella'][i] if resultados['estrella'][i] > 0 else 0 for i in range(len(resultados['frames']))],
        'ratio_de': resultados['ratio_de'],
        'cambio_brusco_disco': resultados['cambios_bruscos_disco'],
        'cambio_brusco_estrella': resultados['cambios_bruscos_estrella'],
        'umbral_confianza': [resultados['confianza_umbral']] * len(resultados['frames'])
    }
    
    df = pd.DataFrame(datos)
    
    # Guardar CSV detallado por frame
    output_path = os.path.join(output_dir, f'resultados_detallados_conf_{confianza:.2f}.csv')
    df.to_csv(output_path, index=False)
    
    # Crear y guardar resumen estadístico
    resumen_stats = {
        'umbral_confianza': confianza,
        'total_frames': len(resultados['frames']),
        'total_disco': sum(resultados['disco']),
        'total_estrella': sum(resultados['estrella']),
        'media_disco': np.mean(resultados['disco']),
        'media_estrella': np.mean(resultados['estrella']),
        'std_disco': np.std(resultados['disco']),
        'std_estrella': np.std(resultados['estrella']),
        'max_disco': max(resultados['disco']),
        'max_estrella': max(resultados['estrella']),
        'min_disco': min(resultados['disco']),
        'min_estrella': min(resultados['estrella']),
        'porcentaje_frames_con_disco': resultados['porcentaje_frames_con_disco'],
        'porcentaje_frames_con_estrella': resultados['porcentaje_frames_con_estrella'],
        'confianza_promedio_disco': resultados['confianza_promedio_disco'],
        'confianza_promedio_estrella': resultados['confianza_promedio_estrella'],
        'frames_con_cambios_bruscos_disco': sum(resultados['cambios_bruscos_disco']),
        'frames_con_cambios_bruscos_estrella': sum(resultados['cambios_bruscos_estrella']),
        'ratio_total_de': sum(resultados['disco'])/(sum(resultados['estrella']) + 1e-6),
        'geometric_mean_disco': gmean([x+1 for x in resultados['disco']])-1 if all(x >= 0 for x in resultados['disco']) else 0,
        'geometric_mean_estrella': gmean([x+1 for x in resultados['estrella']])-1 if all(x >= 0 for x in resultados['estrella']) else 0
    }
    
    df_resumen = pd.DataFrame([resumen_stats])
    resumen_path = os.path.join(output_dir, f'resumen_estadistico_conf_{confianza:.2f}.csv')
    df_resumen.to_csv(resumen_path, index=False)
    
    return output_path, resumen_path

def generar_grafico(resultados, output_dir):
    confianza = resultados['confianza_umbral']
    disco = resultados['disco']
    estrella = resultados['estrella']
    frames = resultados['frames']
    
    # Configuración de la figura con 4 subplots
    fig, axs = plt.subplots(4, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [2, 1, 1, 1]})
    
    # Gráfico principal de detecciones
    axs[0].plot(frames, disco, label='Disco Blanco', color='green')
    axs[0].plot(frames, estrella, label='Estrella Blanca', color='red')
    axs[0].set_title(f'Detecciones por Frame (Confianza={confianza:.2f})')
    axs[0].set_xlabel('Frame')
    axs[0].set_ylabel('Cantidad')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Gráfico de acumulados
    axs[1].plot(frames, resultados['acumulado_disco'], label='Disco Acumulado', color='green', linestyle='--')
    axs[1].plot(frames, resultados['acumulado_estrella'], label='Estrella Acumulada', color='red', linestyle='--')
    axs[1].set_title('Detecciones Acumuladas')
    axs[1].set_xlabel('Frame')
    axs[1].set_ylabel('Total')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    # Gráfico de ratios
    axs[2].plot(frames, resultados['ratio_de'], label='Ratio Disco/Estrella', color='blue')
    axs[2].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    axs[2].set_title('Ratio de Detecciones')
    axs[2].set_xlabel('Frame')
    axs[2].set_ylabel('Ratio')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    
    # NUEVO SUBPLOT: Distribución de confianzas por partícula detectada
    axs[3].hist(resultados['confianza_disco'], bins=20, alpha=0.6, 
               label=f'Disco Blanco (μ={np.mean(resultados["confianza_disco"]):.2f})', 
               color='green', range=(0, 1))
    axs[3].hist(resultados['confianza_estrella'], bins=20, alpha=0.6, 
               label=f'Estrella Blanca (μ={np.mean(resultados["confianza_estrella"]):.2f})', 
               color='red', range=(0, 1))
    axs[3].axvline(x=resultados['confianza_umbral'], color='black', linestyle='--', 
                  label=f'Umbral actual ({resultados["confianza_umbral"]:.2f})')
    axs[3].set_title('Distribución de Confianzas por Partícula Detectada')
    axs[3].set_xlabel('Confianza de Detección')
    axs[3].set_ylabel('Frecuencia')
    axs[3].legend()
    axs[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'grafico_detecciones_conf_{confianza:.2f}.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path

def main():
    # Configuración inicial
    model_path = 'runs/detect/train9/weights/best.pt'
    video_path = '/home/tdelorenzi/testYolo/1-imagenesvideos/tambor_recortado_36s_rotado2.mp4'
    output_dir = '/home/tdelorenzi/testYolo/2-resultados'
    
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)
    
    # Obtener FPS del video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    confianzas = np.round(np.arange(1.00, 0.00, -0.05), 2)
    resultados_totales = []
    
    print(f"\nIniciando procesamiento con FPS={fps:.2f}")
    
    for confianza in confianzas:
        print(f"\nProcesando confianza {confianza:.2f}")
        
        # Procesar video
        resultados = procesar_video_con_confianza(model, video_path, confianza, fps)
        
        # Calcular métricas avanzadas
        resultados = calcular_metricas_avanzadas(resultados)
        
        # Guardar resultados
        csv_det_path, csv_res_path = guardar_csv_detallado(resultados, output_dir)
        img_path = generar_grafico(resultados, output_dir)
        
        # Almacenar para resumen comparativo
        resultados_totales.append(resultados)
        
        print(f"  - CSV detallado guardado: {os.path.basename(csv_det_path)}")
        print(f"  - Resumen estadístico guardado: {os.path.basename(csv_res_path)}")
        print(f"  - Gráfico guardado: {os.path.basename(img_path)}")
    
    # Generar resumen comparativo entre umbrales
    generar_resumen_comparativo(resultados_totales, output_dir)
    
    print("\nProceso completado exitosamente")

def generar_resumen_comparativo(resultados_totales, output_dir):
    datos_comparativos = []
    
    for res in resultados_totales:
        datos_comparativos.append({
            'umbral_confianza': res['confianza_umbral'],
            'total_disco': sum(res['disco']),
            'total_estrella': sum(res['estrella']),
            'ratio_total': sum(res['disco'])/(sum(res['estrella']) + 1e-6),
            'media_disco': np.mean(res['disco']),
            'media_estrella': np.mean(res['estrella']),
            'std_disco': np.std(res['disco']),
            'std_estrella': np.std(res['estrella']),
            'confianza_promedio_disco': res['confianza_promedio_disco'],
            'confianza_promedio_estrella': res['confianza_promedio_estrella'],
            'porcentaje_frames_con_disco': res['porcentaje_frames_con_disco'],
            'porcentaje_frames_con_estrella': res['porcentaje_frames_con_estrella'],
            'frames_con_cambios_bruscos_disco': sum(res['cambios_bruscos_disco']),
            'frames_con_cambios_bruscos_estrella': sum(res['cambios_bruscos_estrella']),
            'geometric_mean_disco': gmean([x+1 for x in res['disco']])-1 if all(x >= 0 for x in res['disco']) else 0,
            'geometric_mean_estrella': gmean([x+1 for x in res['estrella']])-1 if all(x >= 0 for x in res['estrella']) else 0,
            'particulas_perdidas_disco': res['confianza_promedio_disco'] < res['confianza_umbral'] if res['confianza_promedio_disco'] > 0 else 0,
            'particulas_perdidas_estrella': res['confianza_promedio_estrella'] < res['confianza_umbral'] if res['confianza_promedio_estrella'] > 0 else 0
        })
    
    df_comparativo = pd.DataFrame(datos_comparativos)
    output_path = os.path.join(output_dir, 'resumen_comparativo_completo.csv')
    df_comparativo.to_csv(output_path, index=False)
    
    # Gráfico comparativo adicional
    plt.figure(figsize=(14, 8))
    
    # Gráfico de partículas perdidas estimadas
    plt.subplot(1, 2, 1)
    plt.plot(df_comparativo['umbral_confianza'], df_comparativo['confianza_promedio_disco'], 'o-', label='Disco Blanco', color='green')
    plt.plot(df_comparativo['umbral_confianza'], df_comparativo['confianza_promedio_estrella'], 'o-', label='Estrella Blanca', color='red')
    plt.plot(df_comparativo['umbral_confianza'], df_comparativo['umbral_confianza'], '--', label='Umbral', color='gray')
    plt.xlabel('Umbral de Confianza')
    plt.ylabel('Confianza Promedio')
    plt.title('Confianza Promedio vs Umbral')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico de diferencia con el umbral
    plt.subplot(1, 2, 2)
    plt.bar(df_comparativo['umbral_confianza'], 
            df_comparativo['confianza_promedio_disco'] - df_comparativo['umbral_confianza'], 
            width=0.03, label='Disco Blanco', color='green', alpha=0.6)
    plt.bar(df_comparativo['umbral_confianza'], 
            df_comparativo['confianza_promedio_estrella'] - df_comparativo['umbral_confianza'], 
            width=0.03, label='Estrella Blanca', color='red', alpha=0.6)
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Umbral de Confianza')
    plt.ylabel('Diferencia con Umbral')
    plt.title('Margen de Confianza sobre Umbral')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparativo_img_path = os.path.join(output_dir, 'comparativo_confianzas.png')
    plt.savefig(comparativo_img_path, dpi=300)
    plt.close()
    
    print(f"\nResumen comparativo guardado: {os.path.basename(output_path)}")
    print(f"Gráfico comparativo adicional: {os.path.basename(comparativo_img_path)}")

if __name__ == "__main__":
    main()