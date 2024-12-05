import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def calcular_pendiente(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')

def punto_medio(p1, p2):
    return [(p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2]

image = cv2.imread('highway.png')

if image is None:
    raise FileNotFoundError("No se pudo cargar la imagen. Verifique la ruta del archivo.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_yellow = np.array([15, 100, 100], dtype=np.uint8)
upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
yellow_lines = cv2.HoughLinesP(mask_yellow, rho=1, theta=np.pi/180, threshold=30, minLineLength=50, maxLineGap=10)
edge_lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30, minLineLength=50, maxLineGap=10)

left_lines = []
right_lines = []

if edge_lines is not None:
    for line in edge_lines:
        for x1, y1, x2, y2 in line:
            slope = calcular_pendiente(x1, y1, x2, y2)
            if slope < 0 and x1 <= 500 and x2 <= 500:
                left_lines.append([x1, y1, x2, y2])

if yellow_lines is not None:
    for line in yellow_lines:
        for x1, y1, x2, y2 in line:
            slope = calcular_pendiente(x1, y1, x2, y2)
            if slope > 0:
                right_lines.append([x1, y1, x2, y2])

line_image = np.copy(image) * 0

for line in left_lines:
    x1, y1, x2, y2 = line
    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 8)

for line in right_lines:
    x1, y1, x2, y2 = line
    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 8)

bottom_y = image.shape[0]
top_y = int(image.shape[0] * 0.6)

if left_lines:
    left_bottom = max(left_lines, key=lambda line: max(line[1], line[3]))
    left_top = min(left_lines, key=lambda line: min(line[1], line[3]))
    left_point_bottom = [left_bottom[0], left_bottom[1]] if left_bottom[1] > left_bottom[3] else [left_bottom[2], left_bottom[3]]
    left_point_top = [left_top[0], left_top[1]] if left_top[1] < left_top[3] else [left_top[2], left_top[3]]
else:
    left_point_bottom = [image.shape[1] * 0.1, bottom_y]
    left_point_top = [image.shape[1] * 0.1, top_y]

if right_lines:
    right_bottom = max(right_lines, key=lambda line: max(line[1], line[3]))
    right_top = min(right_lines, key=lambda line: min(line[1], line[3]))
    right_point_bottom = [right_bottom[0], right_bottom[1]] if right_bottom[1] > right_bottom[3] else [right_bottom[2], right_bottom[3]]
    right_point_top = [right_top[0], right_top[1]] if right_top[1] < right_top[3] else [right_top[2], right_top[3]]
else:
    right_point_bottom = [image.shape[1] * 0.9, bottom_y]
    right_point_top = [image.shape[1] * 0.9, top_y]

mid_bottom = punto_medio(left_point_bottom, right_point_bottom)
mid_top = punto_medio(left_point_top, right_point_top)

mid_points = np.array([mid_bottom, mid_top])

if len(mid_points) >= 4:
    tck, u = splprep([mid_points[:, 0], mid_points[:, 1]], s=0)
    u_new = np.linspace(u.min(), u.max(), 100)
    x_new, y_new = splev(u_new, tck)
    mid_points = np.vstack((x_new, y_new)).T.astype(int)

cv2.polylines(line_image, [mid_points.reshape((-1, 1, 2))], isClosed=False, color=(0, 255, 0), thickness=8)

overlay = np.zeros_like(image, dtype=np.uint8)

pts_left = np.array([left_point_bottom, mid_bottom, mid_top, left_point_top], np.int32)
pts_left = pts_left.reshape((-1, 1, 2))
cv2.fillPoly(overlay, [pts_left], (255, 0, 255))

pts_right = np.array([right_point_bottom, mid_bottom, mid_top, right_point_top], np.int32)
pts_right = pts_right.reshape((-1, 1, 2))
cv2.fillPoly(overlay, [pts_right], (0, 255, 255))

alpha = 0.2
cv2.addWeighted(overlay, alpha, line_image, 1 - alpha, 0, line_image)

lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)
thickness = 2

left_text_position = (int(mid_bottom[0] * 0.5), int(mid_bottom[1] * 0.9))
right_text_position = (int(mid_bottom[0] * 1.5), int(mid_bottom[1] * 0.9))

cv2.putText(lines_edges, 'Carril Derecho', left_text_position, font, font_scale, font_color, thickness, cv2.LINE_AA)
cv2.putText(lines_edges, 'Carril Izquierdo', right_text_position, font, font_scale, font_color, thickness, cv2.LINE_AA)

plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(lines_edges, cv2.COLOR_BGR2RGB))
plt.title("Detección de carriles y línea divisoria suavizada extendida")
plt.show()
