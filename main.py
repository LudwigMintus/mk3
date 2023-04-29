import cv2
import numpy as np
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
def detect_shapes(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define color ranges for red, green, and blue
    lower_red = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    upper_red = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    green = cv2.inRange(hsv, (36, 25, 25), (86, 255, 255))
    blue = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
    # Combine color ranges
    red = cv2.addWeighted(lower_red, 1.0, upper_red, 1.0, 0.0)
    colors = [green, red, blue]
    shapes = []

    # Detect shapes in each color range
    for i, color in enumerate(colors):
        color_name = ''
        if i == 0:
            color_name = 'Green'
        elif i == 1:
            color_name = 'Red'
        elif i == 2:
            color_name = 'Blue'
        contours, _ = cv2.findContours(color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1500:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                if len(approx) == 3:
                    shapes.append(('triangle', contour, color_name))
                elif len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    if abs(w - h) < 0.1 * max(w, h):
                        shapes.append(('square', contour, color_name))
                    else:
                        shapes.append(('rectangle', contour, color_name))

    return shapes


def draw_shapes(frame, shapes):
    for shape, contour, color_name in shapes:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, shape +" "+ color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=4), cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()

    if ret:
        shapes = detect_shapes(frame)
        draw_shapes(frame, shapes)
        cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()