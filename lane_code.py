import cv2
import numpy as np
import csv

def nothing(x):
    pass

def read_real_positions(csv_filename):
    real_positions = {"left": [], "middle": [], "right": []}
    with open(csv_filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            line = row["Line"]
            x = int(row["X"])
            y = int(row["Y"])
            real_positions[line].append((x, y))
    return real_positions

def transform_real_positions(real_positions, matrix):
    transformed_positions = {"left": [], "middle": [], "right": []}
    for line_name, points in real_positions.items():
        for point in points:
            transformed_point = cv2.perspectiveTransform(np.array([[point]], dtype='float32'), matrix)
            transformed_positions[line_name].append((int(transformed_point[0][0][0]), int(transformed_point[0][0][1])))
    return transformed_positions

def save_positions_to_csv(filename, positions):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Line", "X", "Y"])
        for line_name, points in positions.items():
            for point in points:
                writer.writerow([line_name.capitalize(), point[0], point[1]])

def transform_predicted_positions(predicted_positions, inv_matrix):
    transformed_positions = {"left": [], "middle": [], "right": []}
    for line_name, points in predicted_positions.items():
        for point in points:
            transformed_point = cv2.perspectiveTransform(np.array([[point]], dtype='float32'), inv_matrix)
            transformed_positions[line_name].append((int(transformed_point[0][0][0]), int(transformed_point[0][0][1])))
    return transformed_positions

# Variables para almacenar las posiciones reales de los carriles
csv_filename = "reales_3.csv"
real_positions = read_real_positions(csv_filename)

vidcap = cv2.VideoCapture("movement_videos/3.MP4")
success, image = vidcap.read()

cv2.namedWindow("Trackbars")
cv2.namedWindow("Bird's Eye View")
cv2.namedWindow("Sliding Windows")

cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 21, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 219, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# Loop para procesar el video
repetitions = 1
predicted_positions = {"left": [], "middle": [], "right": []}
for i in range(repetitions):
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the beginning
    while True:
        success, image = vidcap.read()
        if not success:
            break
        frame = cv2.resize(image, (640,480))

        tl = (130, 0)
        bl = (0, 480)
        tr = (520, 0)
        br = (640, 480)

        pts1 = np.float32([tl, bl, tr, br]) 
        pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]]) 
        
        matrix = cv2.getPerspectiveTransform(pts1, pts2) 
        inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
        transformed_frame = cv2.warpPerspective(frame, matrix, (640,480))

        hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
        
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")
        
        lower = np.array([l_h,l_s,l_v])
        upper = np.array([u_h,u_s,u_v])
        mask = cv2.inRange(hsv_transformed_frame, lower, upper)

        histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
        midpoint = int(histogram.shape[0] / 2)
        left_base = np.argmax(histogram[:midpoint])
        right_base = np.argmax(histogram[midpoint:]) + midpoint

        y = 472
        lx = []
        mx = []
        rx = []

        msk = mask.copy()

        while y > 0:
            img = mask[y-20:y, left_base-25:left_base+25]
            contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    cx = int(M["m10"]/M["m00"])
                    left_base = left_base - 25 + cx
                    lx.append(left_base)
                    predicted_positions["left"].append((left_base, y))
            else:
                predicted_positions["left"].append((left_base, y))

            middle_base = (left_base + right_base) // 2
            img = mask[y-20:y, middle_base-25:middle_base+25]
            contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    cx = int(M["m10"]/M["m00"])
                    middle_base = middle_base - 25 + cx
                    mx.append(middle_base)
                    predicted_positions["middle"].append((middle_base, y))
            else:
                predicted_positions["middle"].append((middle_base, y))

            img = mask[y-20:y, right_base-25:right_base+25]
            contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    cx = int(M["m10"]/M["m00"])
                    right_base = right_base - 25 + cx
                    rx.append(right_base)
                    predicted_positions["right"].append((right_base, y))
            else:
                predicted_positions["right"].append((right_base, y))

            cv2.rectangle(msk, (left_base-25, y), (left_base+25, y-20), (255,255,255), 2)
            cv2.rectangle(msk, (middle_base-25, y), (middle_base+25, y-20), (255,255,255), 2)
            cv2.rectangle(msk, (right_base-25, y), (right_base+25, y-20), (255,255,255), 2)
            y -= 20

        lane1_mask = np.zeros_like(transformed_frame)
        lane2_mask = np.zeros_like(transformed_frame)
        cv2.rectangle(lane1_mask, (left_base-25, 0), (middle_base+25, 480), (0, 255, 0), -1)
        cv2.rectangle(lane2_mask, (middle_base-25, 0), (right_base+25, 480), (255, 0, 0), -1)
        lanes_birdseye = cv2.addWeighted(transformed_frame, 1, lane1_mask, 0.3, 0)
        lanes_birdseye = cv2.addWeighted(lanes_birdseye, 1, lane2_mask, 0.3, 0)

        lane1_original = cv2.warpPerspective(lane1_mask, inv_matrix, (640, 480))
        lane2_original = cv2.warpPerspective(lane2_mask, inv_matrix, (640, 480))
        lanes_original = cv2.addWeighted(frame, 1, lane1_original, 0.3, 0)
        lanes_original = cv2.addWeighted(lanes_original, 1, lane2_original, 0.3, 0)

        cv2.putText(lanes_original, f"Repetition: {i+1}/{repetitions}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        transformed_real_positions = transform_real_positions(real_positions, matrix)

        for line_name, points in transformed_real_positions.items():
            for point in points:
                cv2.circle(lanes_birdseye, point, 5, (0, 255, 255), -1)

        # Transform predicted positions back to original coordinates
        transformed_predicted_positions = transform_predicted_positions(predicted_positions, inv_matrix)

        # Draw the transformed predicted positions on the original frame
        for line_name, points in transformed_predicted_positions.items():
            for point in points:
                if line_name == "left":
                    cv2.circle(lanes_original, point, 5, (0, 0, 255), -1)
                elif line_name == "middle":
                    cv2.circle(lanes_original, point, 5, (0, 255, 0), -1)
                elif line_name == "right":
                    cv2.circle(lanes_original, point, 5, (255, 0, 0), -1)

        cv2.imshow("Sliding Windows", msk)
        cv2.imshow("Bird's Eye View", lanes_birdseye)
        cv2.imshow("Lanes", lanes_original)

        cv2.imshow("Trackbars", np.zeros((1, 400), np.uint8))

        if cv2.waitKey(10) == 27:
            break

vidcap.release()
cv2.destroyAllWindows()

# Save the predicted positions to a CSV file in original coordinates
save_positions_to_csv("estimadas_3.csv", transformed_predicted_positions)