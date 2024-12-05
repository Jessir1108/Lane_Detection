import csv

def is_point_in_polygon(x, y, polygon):
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def get_area_coordinates(lines, width, height):
    area_coordinates = {"left": [], "right": []}
    for line_name, points in lines.items():
        for x in range(width):
            for y in range(height):
                if is_point_in_polygon(x, y, points):
                    area_coordinates[line_name].append((x, y))
    return area_coordinates

def save_area_coordinates_to_csv(area_coordinates, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Line", "X", "Y"])
        for line_name, points in area_coordinates.items():
            for point in points:
                writer.writerow([line_name, point[0], point[1]])

def get_rectangle_coordinates(rectangles):
    rectangle_coordinates = []
    for rect in rectangles:
        x1, y1, x2, y2 = rect
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                rectangle_coordinates.append((x, y))
    return rectangle_coordinates

def save_rectangle_coordinates_to_csv(rectangle_coordinates, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["X", "Y"])
        for point in rectangle_coordinates:
            writer.writerow([point[0], point[1]])

def main():
    # Load the realCoordinates.csv
    lines = {"left": [], "right": []}
    with open("realCoordinates.csv", mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            line_name, x, y = row
            lines[line_name].append((int(x), int(y)))

    # Define the width and height of the area (assuming 640x480 as in the original script)
    width, height = 640, 480

    # Get the area coordinates
    area_coordinates = get_area_coordinates(lines, width, height)

    # Save the area coordinates to CSV
    save_area_coordinates_to_csv(area_coordinates, "areaRealCoordinates.csv")

    # Load the rectangle coordinates from rectangleCoordinates.csv
    rectangles = []
    with open("rectangleCoordinates.csv", mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            x1, y1, x2, y2 = map(int, row)
            rectangles.append((x1, y1, x2, y2))

    # Get the rectangle coordinates
    rectangle_coordinates = get_rectangle_coordinates(rectangles)

    # Save the rectangle coordinates to CSV
    save_rectangle_coordinates_to_csv(rectangle_coordinates, "areaRectangleOutput.csv")

if __name__ == "__main__":
    main()