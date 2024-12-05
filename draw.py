import cv2
import numpy as np
import csv

# Variables globales para almacenar las posiciones de las líneas
lines = {"left": [], "middle": [], "right": []}
current_line = "left"
drawing = False  # Variable para indicar si se está dibujando

# Función para manejar los eventos del mouse
def draw_line(event, x, y, flags, param):
    global lines, current_line, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        lines[current_line].append((x, y))
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            lines[current_line].append((x, y))
            if current_line == "left":
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            elif current_line == "middle":
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            elif current_line == "right":
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
            cv2.imshow("Frame", frame)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# Función para guardar las posiciones en un archivo CSV
def save_positions_to_csv(filename, lines):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Line", "X", "Y"])
        for line_name, points in lines.items():
            for point in points:
                writer.writerow([line_name, point[0], point[1]])

# Cargar el video
video_path = "movement_videos/3.MP4"
vidcap = cv2.VideoCapture(video_path)
success, frame = vidcap.read()

if not success:
    print("Error al cargar el video")
    exit()

# Redimensionar el frame
new_width = 640
new_height = 480
frame = cv2.resize(frame, (new_width, new_height))

# Mostrar el primer frame
cv2.imshow("Frame", frame)
cv2.setMouseCallback("Frame", draw_line)

print("Dibuja las líneas en el siguiente orden: left, middle, right")
print("Presiona 'n' para cambiar a la siguiente línea")
print("Presiona 's' para guardar y salir")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):
        if current_line == "left":
            current_line = "middle"
            print("Cambiado a la línea: middle")
        elif current_line == "middle":
            current_line = "right"
            print("Cambiado a la línea: right")
        elif current_line == "right":
            print("Todas las líneas dibujadas. Presiona 's' para guardar y salir")
    elif key == ord('s'):
        save_positions_to_csv("reales_3.csv", lines)
        print("Posiciones guardadas en DibujadasPosiciones.csv")
        break

vidcap.release()
cv2.destroyAllWindows()