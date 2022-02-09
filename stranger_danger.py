# import the necessary packages
import math

import cv2
import numpy as np

# construct the argument parse and parse the arguments
radius_trackbar = 5
diagonal_aspect = math.hypot(640, 480)
middle_fov = math.radians(75)
hor_focal = 640 / (2 * (math.tan(middle_fov / 2) * (640 / diagonal_aspect)))
ver_focal = 480 / (2 * (math.tan(middle_fov / 2) * (480 / diagonal_aspect)))
redLower = np.array([4, 108, 52])
redUpper = np.array([11, 255, 255])
blueLower = np.array([0, 104, 0])
blueUpper = np.array([180, 255, 255])
is_blue = False


def callback(_):
    pass


def get_pitch(vertical_focal_length, y_pixels):
    return -math.degrees(math.atan((y_pixels - 240) / vertical_focal_length))


def get_yaw(horizontal_focal_length, x_pixels):
    return math.degrees(math.atan((x_pixels - 320) / horizontal_focal_length))


def split_cargos(cnt):
    if len(cnt) == 1:
        return
    print(cnt)
    hull = cv2.convexHull(cnt, returnPoints=False)
    if hull is not None:
        # print(cv2.isContourConvex(hull))
        hull[::-1].sort(axis=0)
        defects = cv2.convexityDefects(cnt, hull)
        if defects is not None:
            # print(defects, len(defects))
            pass

def update_trackbars():
    global radius_trackbar, is_blue

    radius_trackbar = cv2.getTrackbarPos('radius_trackbars', 'trackbars')
    is_blue = bool(cv2.getTrackbarPos('is_blue', 'trackbars'))

    if not is_blue:
        global redLower, redUpper
        redLower[0] = cv2.getTrackbarPos('min_h', 'trackbars')
        redLower[1] = cv2.getTrackbarPos('min_s', 'trackbars')
        redLower[2] = cv2.getTrackbarPos('min_v', 'trackbars')

        redUpper[0] = cv2.getTrackbarPos('max_h', 'trackbars')
        redUpper[1] = cv2.getTrackbarPos('max_s', 'trackbars')
        redUpper[2] = cv2.getTrackbarPos('max_v', 'trackbars')
    else:
        global blueLower, blueUpper
        blueLower[0] = cv2.getTrackbarPos('min_h', 'trackbars')
        blueLower[1] = cv2.getTrackbarPos('min_s', 'trackbars')
        blueLower[2] = cv2.getTrackbarPos('min_v', 'trackbars')

        blueUpper[0] = cv2.getTrackbarPos('max_h', 'trackbars')
        blueUpper[1] = cv2.getTrackbarPos('max_s', 'trackbars')
        blueUpper[2] = cv2.getTrackbarPos('max_v', 'trackbars')


cv2.namedWindow('trackbars')

cv2.createTrackbar('is_blue', 'trackbars', 0, 1, callback)
cv2.createTrackbar('radius_trackbars', 'trackbars', 0, 200, callback)
cv2.createTrackbar('min_h', 'trackbars', redLower[0], 180, callback)
cv2.createTrackbar('min_s', 'trackbars', redLower[1], 255, callback)
cv2.createTrackbar('min_v', 'trackbars', redLower[2], 255, callback)
cv2.createTrackbar('max_h', 'trackbars', redUpper[0], 180, callback)
cv2.createTrackbar('max_s', 'trackbars', redUpper[1], 255, callback)
cv2.createTrackbar('max_v', 'trackbars', redUpper[2], 255, callback)

vs = cv2.VideoCapture(2)
vs.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cv2.waitKey(1) & 0xFF != 27:
    # grab the current frame
    has_frame, frame = vs.read()

    if not has_frame:
        print("You are a loser, Why? Because...")
        continue
    update_trackbars()

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    if is_blue:
        mask = cv2.inRange(hsv, blueUpper, blueLower)
    else:
        mask = cv2.inRange(hsv, redLower, redUpper)

    mask = cv2.erode(mask, (3, 3), iterations=2)
    mask = cv2.dilate(mask, (3, 3), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (3, 3), iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (3, 3), iterations=3)

    cv2.imshow("hsv", cv2.bitwise_and(frame, frame, mask=mask))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # only proceed if at least one contour was found
    if len(contours) > 0:
        for contour in contours:
            split_cargos(contour)
            area = cv2.contourArea(contour)
            if area < 300:
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)

            # only proceed if the radius meets a minimum size
            if radius > radius_trackbar:
                area = cv2.contourArea(contour)
                circle_area = math.pi * radius ** 2
                solidity = float(area) / circle_area
                # _, _, w, h = cv2.boundingRect(c)
                # aspect_ratio = float(w) / h

                if solidity > 0.6:
                    cv2.circle(frame, (int(x), int(y)), int(radius),
                               (0, 255, 255), 2)

                    M = cv2.moments(contour)
                    cX = int(M["m10"] / (M["m00"] + 1e-54))
                    cY = int(M["m01"] / (M["m00"] + 1e-54))

                    pitch = get_pitch(ver_focal, cY)
                    yaw = get_yaw(hor_focal, cX)
                    # cv2.putText(frame, f"pitch: {pitch:2}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, (209, 80, 255), 3)
                    # cv2.putText(frame, f"yaw: {yaw:2}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, (209, 80, 255), 3)

    cv2.imshow("frame", frame)

vs.release()
cv2.destroyAllWindows()
