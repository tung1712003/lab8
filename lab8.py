import cv2

def image_processing():
    img = cv2.imread("3.png")
    if img is None:
        raise FileNotFoundError("Failed to load the image")

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  
    cv2.imshow('Original Image', img)
    cv2.imshow('HSV Image', hsv_img)

    height, width = img.shape[:2]
    center_x, center_y = width // 2, height // 2
    square_size = 200

    top_left = (center_x - square_size // 2, center_y - square_size // 2)
    bottom_right = (center_x + square_size // 2, center_y + square_size // 2)
    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

    cv2.imshow('Labeled Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def video_processing():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise IOError("Failed to open the camera")

    frame_counter = 0
    down_points = (640, 480)

    fly_image = cv2.imread("fly64.png", cv2.IMREAD_UNCHANGED)
    if fly_image is None:
        raise FileNotFoundError("Image does not exist!")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture a frame")
            break

        frame = cv2.resize(frame, down_points, interpolation=cv2.INTER_LINEAR)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        ret, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if frame_counter % 5 == 0:
                center_x = x + (w // 2)
                center_y = y + (h // 2)
                print("Object center coordinates:", center_x, center_y)

                fly_center_x = center_x
                fly_center_y = center_y

                fly_top_left_x = fly_center_x - fly_image.shape[1] // 2
                fly_top_left_y = fly_center_y - fly_image.shape[0] // 2

                overlay_image(frame, fly_image, fly_top_left_x, fly_top_left_y)

        cv2.imshow('Frame', frame)
        cv2.imshow('HSV Frame', hsv_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_counter += 1

    cam.release()
    cv2.destroyAllWindows()
# Доп
def overlay_image(background, overlay, x, y):
    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        background[y:y+overlay.shape[0], x:x+overlay.shape[1], c] = \
            (1 - alpha) * background[y:y+overlay.shape[0], x:x+overlay.shape[1], c] + \
            alpha * overlay[:, :, c]

if __name__ == '__main__':
    image_processing()
    video_processing()