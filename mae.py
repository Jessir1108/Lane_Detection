import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
import cv2

# Cargar los datos reales y predichos desde sus respectivos archivos CSV
real_data = pd.read_csv("reales_3.csv")
pred_data = pd.read_csv("estimadas_3.csv")

# Leer el primer frame del video
vidcap = cv2.VideoCapture("movement_videos/3.MP4")
success, image = vidcap.read()

if success:
    # Redimensionar el frame a 640x480
    image = cv2.resize(image, (640, 480))

    # Calcular MAE y MAD
    video_width = 640
    lines = ['Left', 'Right', 'Middle']
    total_mae = 0
    total_mad = 0
    valid_lines = 0

    for line in lines:
        real_coords = real_data[real_data['Line'] == line][['X', 'Y']].values
        pred_coords = pred_data[pred_data['Line'] == line][['X', 'Y']].values

        if len(real_coords) == 0 or len(pred_coords) == 0:
            print(f"No hay datos suficientes para la línea {line}.")
            continue

        tree = cKDTree(pred_coords)
        distances, indices = tree.query(real_coords)

        mae = np.mean(distances) * 4
        mad = np.mean(np.abs(distances - np.mean(distances))) * 4

        total_mae += mae
        total_mad += mad
        valid_lines += 1

        mae_percentage = (mae / video_width) * 100
        mad_percentage = (mad / video_width) * 100

        print(f"Métricas para la línea {line}:")
        print(f"  MAE: {mae:.2f} px ({mae_percentage:.2f}%)")
        print(f"  MAD: {mad:.2f} px ({mad_percentage:.2f}%)")

        # Dibujar las líneas estimadas basadas en MAE y MAD
        for i in range(len(real_coords) - 1):
            start_point = (int(real_coords[i][0] + mae), int(real_coords[i][1] + mae))
            end_point = (int(real_coords[i + 1][0] + mae), int(real_coords[i + 1][1] + mae))
            cv2.line(image, start_point, end_point, (255, 255, 0), int(mae * 0.3))  # MAE line in yellow

            start_point = (int(real_coords[i][0] + mad), int(real_coords[i][1] + mad))
            end_point = (int(real_coords[i + 1][0] + mad), int(real_coords[i + 1][1] + mad))
            cv2.line(image, start_point, end_point, (255, 0, 255), int(mad * 0.3))  # MAD line in magenta

    if valid_lines > 0:
        avg_mae = total_mae / valid_lines
        avg_mad = total_mad / valid_lines

        avg_mae_percentage = (avg_mae / video_width) * 100
        avg_mad_percentage = (avg_mad / video_width) * 100

        print(f"MAE promedio: {avg_mae:.2f} px ({avg_mae_percentage:.2f}%)")
        print(f"MAD promedio: {avg_mad:.2f} px ({avg_mad_percentage:.2f}%)")

    # Añadir leyendas para MAE y MAD en la imagen
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'MAE', (10, 30), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, 'MAD', (10, 70), font, 1, (255, 0, 255), 2, cv2.LINE_AA)

    # Guardar la imagen con las líneas y leyendas dibujadas
    cv2.imwrite("MAE_MAD.png", image)

    # Mostrar el frame con las líneas y leyendas dibujadas
    cv2.imshow("First Frame with Lines and Legends", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

vidcap.release()
