import cv2
import numpy as np
import torch
from scipy.interpolate import splprep, splev

# Cargar el modelo preentrenado de YOLOv5 para la detección de vehículos
vehicle_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Función para calcular la pendiente de una línea
def calcular_pendiente(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')

# Función para calcular el punto medio entre dos puntos
def punto_medio(p1, p2):
    return [(p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2]

# Función para aplicar Canny Edge y ROI
def aplicar_canny_y_roi(frame):
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque Gaussiano
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detectar bordes usando Canny
    edges = cv2.Canny(blur, 50, 150)
    
    # Definir una región de interés (ROI)
    height = frame.shape[0]
    polygons = np.array([[(0, height), (frame.shape[1], height), (frame.shape[1] // 2, int(height * 0.6))]])
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, polygons, 255)
    
    # Aplicar la máscara a la imagen de bordes
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges

# Función para procesar cada frame
def procesar_frame(frame):
    # Detectar vehículos en el frame
    results = vehicle_model(frame)
    vehicle_boxes = results.xyxy[0].cpu().numpy()

    # Crear una máscara para excluir los vehículos
    vehicle_mask = np.zeros_like(frame[:, :, 0])
    for box in vehicle_boxes:
        x1, y1, x2, y2, conf, cls = box
        if cls in [2, 3, 5, 7]:  # Clases para coche, moto, bus, camión
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            vehicle_mask[y1:y2, x1:x2] = 255

    # Invertir la máscara para usarla en la detección de bordes
    vehicle_mask_inv = cv2.bitwise_not(vehicle_mask)

    # Aplicar Canny y ROI a la imagen
    edges = aplicar_canny_y_roi(frame)
    edges = cv2.bitwise_and(edges, edges, mask=vehicle_mask_inv)  # Excluir vehículos

    # Detectar líneas usando la Transformada de Hough
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=50)

    # Crear una imagen para dibujar las líneas
    line_image = np.zeros_like(frame)

    left_lines, right_lines = [], []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = calcular_pendiente(x1, y1, x2, y2)
                if slope < 0:  # Línea izquierda
                    left_lines.append([x1, y1, x2, y2])
                elif slope > 0:  # Línea derecha
                    right_lines.append([x1, y1, x2, y2])

    # Promediar las líneas izquierda y derecha
    def promedio_lineas(lines, frame_height):
        if not lines:
            return None
        x_coords, y_coords = [], []
        for line in lines:
            x1, y1, x2, y2 = line
            x_coords += [x1, x2]
            y_coords += [y1, y2]
        slope, intercept = np.polyfit(x_coords, y_coords, 1)
        y1 = frame_height
        y2 = int(y1 * 0.6)  
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return [x1, y1, x2, y2]

    left_line = promedio_lineas(left_lines, frame.shape[0])
    right_line = promedio_lineas(right_lines, frame.shape[0])

    # Dibujar las líneas promedio
    if left_line is not None:
        cv2.line(line_image, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (255, 0, 0), 10)
    if right_line is not None:
        cv2.line(line_image, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 0), 10)

    # Superponer las líneas sobre la imagen original
    output = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    return output

# Cargar el video
cap = cv2.VideoCapture('movement_videos/3.MP4')

# Verificar si el video se cargó correctamente
if not cap.isOpened():
    raise FileNotFoundError("No se pudo cargar el video. Verifique la ruta del archivo.")

# Obtener las dimensiones del video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Procesando video a {fps} FPS y resolución {width}x{height}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Procesar el frame
    processed_frame = procesar_frame(frame)

    # Mostrar el frame procesado
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', width, height)
    cv2.imshow('Frame', processed_frame)
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
