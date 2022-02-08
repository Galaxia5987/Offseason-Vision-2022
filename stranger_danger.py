# import the necessary packages
import math
from typing import Union

import cv2
import imutils
import numpy as np

# construct the argument parse and parse the arguments
cv2.namedWindow('trackbars')
radius_trackbar = 5


def callback(_):
    pass


def draw_contours(filtered_contours, original):
    for cnt in filtered_contours:
        (a, b), radius = cv2.minEnclosingCircle(cnt)
        center = (int(a), int(b))
        cv2.circle(original, center, int(radius), (255, 255, 0), 5)
        distance = utils.distance(constants.FOCAL_LENGTHS['lifecam'],
                                  constants.GAME_PIECE_SIZES['fuel']['diameter'],
                                  radius * 2)
        cv2.putText(original, str(int(distance * 100)), (int(a), int(b + 2 * radius)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 0, 0), 3)


def get_pitch(vertical_focal_length, y):
    return -math.degrees(math.atan(-y / vertical_focal_length))


def get_yaw(horizontal_focal_length, x):
    return math.degrees(math.atan(-x / horizontal_focal_length))


def get_distance(target_height, camera_height, camera_pitch_radians, target_pitch_radians):
    return (target_height - camera_height) / math.tan(camera_pitch_radians + target_pitch_radians)


redLower = np.array([0, 121, 83])
redUpper = np.array([180, 194, 255])
blueLower = (0, 104, 0)
blueUpper = (180, 255, 255)

cv2.createTrackbar('radius_trackbars', 'trackbars', 0, 10, callback)
cv2.createTrackbar('min_h', 'trackbars', redLower[0], 180, callback)
cv2.createTrackbar('min_s', 'trackbars', redLower[1], 255, callback)
cv2.createTrackbar('min_v', 'trackbars', redLower[2], 255, callback)
cv2.createTrackbar('max_h', 'trackbars', redUpper[0], 180, callback)
cv2.createTrackbar('max_s', 'trackbars', redUpper[1], 255, callback)
cv2.createTrackbar('max_v', 'trackbars', redUpper[2], 255, callback)

vs = cv2.VideoCapture(1)


# if a video path was not supplied, grab the reference
# to the webcam
def is_circle(cnt: np.array, minimum: Union[float, int]) -> bool:
    ratio = circle_ratio(cnt)
    return minimum <= ratio <= 1


def filtered_contours(contours):
    correct_contours = []

    if contours is not None:

        for cnt in contours:
            if len(cnt) < 50:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / h
            area_circle_from_rect = math.pi * ((w / 2) ** 2)
            _, radius = cv2.minEnclosingCircle(cnt)

            area_circle = math.pi * (radius ** 2)

            area_ratio = area_circle / area_circle_from_rect

            if 0.75 < ratio < 1.25 and 0.75 < area_ratio < 1.25 and radius > 5:
                correct_contours.append(cnt)

    return correct_contours


while cv2.waitKey(1) & 0xFF != 27:
    # grab the current frame
    has_frame, frame = vs.read()

    if not has_frame:
        break

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
    # contours = filtered_contours(contours)
    # for cnt in contours:
    #     ((x, y), radius) = cv2.minEnclosingCircle(cnt)
    #     cv2.circle(frame, (int(x), int(y)), int(radius),
    #                (0, 255, 255), 2)

    # only proceed if at least one contour was found
    if len(contours) > 0:
        for c in contours:
            area = cv2.contourArea(c)
            if area < 300:
                continue
            ((x, y), radius) = cv2.minEnclosingCircle(c)
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

                    diagonal_aspect = math.hypot(640, 480)
                    middle_fov = math.radians(40) / 2  # 75
                    hor_focal = 640 / (2 * (math.tan(middle_fov) * (640 / diagonal_aspect)))
                    ver_focal = 480 / (2 * (math.tan(middle_fov) * (480 / diagonal_aspect)))
                    cX = int(M["m10"] / (M["m00"] + 1e-5))
                    cY = int(M["m01"] / (M["m00"] + 1e-5))
                    pitch = get_pitch(ver_focal, cY)
                    yaw = get_yaw(hor_focal, cX)
                    distance = get_distance(0.23, 0.11, 1e-5, math.radians(pitch))
                    print("pitch", pitch)
                    print("yaw", yaw)
                    print("distance", distance)

    cv2.imshow("frame", frame)

vs.release()
cv2.destroyAllWindows()
