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


cv2.createTrackbar('radius_trackbars', 'trackbars', 0, 10, callback)

redLower = (0, 121, 0)
redUpper = (9, 255, 255)
blueLower = (0, 104, 0)
blueUpper = (180, 255, 255)
vs = cv2.VideoCapture(0)


# if a video path was not supplied, grab the reference
# to the webcam
def is_circle(cnt: np.array, minimum: Union[float, int]) -> bool:
    ratio = circle_ratio(cnt)
    return minimum <= ratio <= 1


def filter_contours(contours, hierarchy):
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

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    isBlue = False

    if isBlue:
        mask = cv2.inRange(hsv, blueUpper, blueLower)
    else:
        mask = cv2.inRange(hsv, redLower, redUpper)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cv2.imshow("hsv", cv2.bitwise_and(frame, frame, mask=mask))

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        for c in cnts:
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # only proceed if the radius meets a minimum size

            if radius > radius_trackbar:
                area = cv2.contourArea(c)
                circle_area = math.pi * radius ** 2
                solidity = float(area) / circle_area
                # _, _, w, h = cv2.boundingRect(c)
                # aspect_ratio = float(w) / h

                if solidity > 0.8:
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


                    cv2.circle(frame, (int(x), int(y)), int(radius),
                               (0, 255, 255), 2)

    cv2.imshow("frame", frame)

if not args.get("video", False):
    vs.stop()

else:
    vs.release()

cv2.destroyAllWindows()
