# import the necessary packages
import math

import cv2
import imutils
import numpy as np

# construct the argument parse and parse the arguments
cv2.namedWindow('trackbars')
radius_trackbar = 5
diagonal_aspect = math.hypot(640, 480)
middle_fov = math.radians(75)
hor_focal = 640 / (2 * (math.tan(middle_fov / 2) * (640 / diagonal_aspect)))
ver_focal = 480 / (2 * (math.tan(middle_fov / 2) * (480 / diagonal_aspect)))


def callback(_):
    pass


def get_pitch(vertical_focal_length, y):
    return -math.degrees(math.atan((y - 240) / vertical_focal_length))


def get_yaw(horizontal_focal_length, x):
    return math.degrees(math.atan((x - 320) / horizontal_focal_length))


def get_ratio


redLower = np.array([4, 108, 52])
redUpper = np.array([11, 255, 255])
blueLower = (0, 104, 0)
blueUpper = (180, 255, 255)

cv2.createTrackbar('radius_trackbars', 'trackbars', 0, 200, callback)
cv2.createTrackbar('min_h', 'trackbars', redLower[0], 180, callback)
cv2.createTrackbar('min_s', 'trackbars', redLower[1], 255, callback)
cv2.createTrackbar('min_v', 'trackbars', redLower[2], 255, callback)
cv2.createTrackbar('max_h', 'trackbars', redUpper[0], 180, callback)
cv2.createTrackbar('max_s', 'trackbars', redUpper[1], 255, callback)
cv2.createTrackbar('max_v', 'trackbars', redUpper[2], 255, callback)

vs = cv2.VideoCapture(0)
vs.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cv2.waitKey(1) & 0xFF != 27:
    # grab the current frame
    has_frame, frame = vs.read()

    if not has_frame:
        print("You are a loser, Why? Because...")
        break

    radius_trackbar = cv2.getTrackbarPos('radius_trackbars', 'trackbars')
    redLower[0] = cv2.getTrackbarPos('min_h', 'trackbars')
    redLower[1] = cv2.getTrackbarPos('min_s', 'trackbars')
    redLower[2] = cv2.getTrackbarPos('min_v', 'trackbars')

    redUpper[0] = cv2.getTrackbarPos('max_h', 'trackbars')
    redUpper[1] = cv2.getTrackbarPos('max_s', 'trackbars')
    redUpper[2] = cv2.getTrackbarPos('max_v', 'trackbars')

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    isBlue = False

    if isBlue:
        mask = cv2.inRange(hsv, blueUpper, blueLower)
    else:
        mask = cv2.inRange(hsv, redLower, redUpper)

    mask = cv2.erode(mask, (3, 3), iterations=2)
    mask = cv2.dilate(mask, (3, 3), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (3, 3), iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (3, 3), iterations=3)

    cv2.imshow("hsv", cv2.bitwise_and(frame, frame, mask=mask))

    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # hull = cv2.convexHull(contours)

    adam = cv2.Canny(frame, 90, 90 * 3)

    cv2.imshow("Gaia2", adam)

    # only proceed if at least one contour was found
    if len(contours) > 0:
        for c in contours:
            area = cv2.contourArea(c)
            if area < 300:
                continue
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if cv2.minEnclosingCircle(c):

            M = cv2.moments(c)
            center = (int(M["m10"] / (M["m00"] + 1e-54)), int(M["m01"] / (M["m00"] + 1e-54)))

            # only proceed if the radius meets a minimum size
            if radius > radius_trackbar:
                area = cv2.contourArea(c)
                circle_area = math.pi * radius ** 2
                solidity = float(area) / circle_area
                # _, _, w, h = cv2.boundingRect(c)
                # aspect_ratio = float(w) / h

                if solidity > 0.6:
                    cv2.circle(frame, (int(x), int(y)), int(radius),
                               (0, 255, 255), 2)
                    cX = int(M["m10"] / (M["m00"] + 1e-5))
                    cY = int(M["m01"] / (M["m00"] + 1e-5))
                    pitch = get_pitch(ver_focal, cY)
                    yaw = get_yaw(hor_focal, cX)
                    print("pitch", pitch)
                    print("yaw", yaw)

    cv2.imshow("frame", frame)

vs.release()
cv2.destroyAllWindows()
