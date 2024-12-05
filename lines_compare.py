import cv2
import pandas as pd

# Cargar datos de los archivos CSV
prueba2_data = pd.read_csv('Prueba2.csv')
predicted_positions_data = pd.read_csv('PredictedPositions.csv')

# Leer el primer frame del video
vidcap = cv2.VideoCapture("movement_videos/3.MP4")
success, image = vidcap.read()

if success:
    # Redimensionar el frame a 640x480
    image = cv2.resize(image, (640, 480))

    # Dibujar las líneas de "Prueba2" en color verde, azul y rojo dependiendo de la línea
    for i in range(len(prueba2_data) - 1):
        if prueba2_data.iloc[i]['Line'] == 'left':
            cv2.line(image, (prueba2_data.iloc[i]['X'], prueba2_data.iloc[i]['Y']),
                     (prueba2_data.iloc[i + 1]['X'], prueba2_data.iloc[i + 1]['Y']), (0, 255, 0), 2)
        elif prueba2_data.iloc[i]['Line'] == 'middle':
            cv2.line(image, (prueba2_data.iloc[i]['X'], prueba2_data.iloc[i]['Y']),
                     (prueba2_data.iloc[i + 1]['X'], prueba2_data.iloc[i + 1]['Y']), (255, 0, 0), 2)
        elif prueba2_data.iloc[i]['Line'] == 'right':
            cv2.line(image, (prueba2_data.iloc[i]['X'], prueba2_data.iloc[i]['Y']),
                     (prueba2_data.iloc[i + 1]['X'], prueba2_data.iloc[i + 1]['Y']), (0, 0, 255), 2)

    # Dibujar las líneas de "PredictedPositions" en color rojo
    for i in range(len(predicted_positions_data) - 1):
        cv2.line(image, (predicted_positions_data.iloc[i]['X'], predicted_positions_data.iloc[i]['Y']),
                 (predicted_positions_data.iloc[i + 1]['X'], predicted_positions_data.iloc[i + 1]['Y']), (0, 0, 255), 2)

    # Mostrar el frame con las líneas dibujadas
    cv2.imshow("First Frame with Lines", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

vidcap.release()