#!/usr/bin/python3
import sys
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt

"""Image processing 2nd assignment

Circle detector using Hough Transform

Authors: Tom Mansion <tom.mansion@universite-paris-saclay.fr>, Sophie Nguyen <sophie.nguyen@universite-paris-saclay.fr>
"""

IMAGES_FOLDER = "images/"
IMG = "fourn.png"

MAX_KERNEL_LENGTH = 9

R_MIN = 1
R_MAX = 100

C_MIN = 1
C_MAX = 100

RAD_MIN = 5

N_CIRCLES = 5
DELTA_R = 10
DELTA_C = 10
DELTA_RAD = 10

CIRCLE_THICKNESS = 1

def is_local_maximum(acc, i, j, k):
    ref = acc[i, j, k]
    local_area = acc[i-1:i+2, j-1:j+2, k-1:k+2]

    if not np.any(local_area):
        return False
       
    maximum = np.amax(local_area)
    return maximum == ref

def get_detected_circles(acc, N_circles=N_CIRCLES):
    rows, cols, depth = np.shape(acc)
    local_maxima = []
    print("Storing local maxima...")
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            for k in range(1, depth - 1):
                if is_local_maximum(acc, i, j, k):
                    local_maxima.append({ "value": acc[i, j, k], "r": j, "c": i, "rad": k})

    sorted_maxima = sorted(local_maxima, key=lambda maximum: maximum["value"])[::-1]

    detected_circles = []

    print("Retrieve N biggest local maxima")
    for i in range(len(sorted_maxima)):
        ignore = False
        if len(detected_circles) == N_circles:
            break
        for circle in detected_circles:
            if (abs(circle['r'] - sorted_maxima[i]['r']) < DELTA_R 
                and abs(circle['c'] - sorted_maxima[i]['c']) < DELTA_C 
                and abs(circle['rad'] - sorted_maxima[i]['rad']) < DELTA_RAD):
                ignore = True
                break
        if not ignore:
            detected_circles.append(sorted_maxima[i])

    return detected_circles

def hough_circles(img, rows, cols, rad_max):
    acc = np.zeros((rows, cols, int(rad_max - RAD_MIN)))

    for y in range(0, rows):
        for x in range(0, cols):
            print(f"Current pixel: {y}, {x}")
            if img[y, x] > 0:
                for r in range(1, rows):
                    for c in range (1, cols):
                        rad = int(math.sqrt(((y - r) ** 2) + ((x - c) ** 2)))
                        if rad >= RAD_MIN and rad < rad_max:
                            acc[r, c, rad - RAD_MIN] += 1.0/rad

    detected_circles = get_detected_circles(acc)

    return detected_circles
    
def draw_circles(img, circles):
    modified_img = img

    for circle in circles:
        modified_img = cv.circle(img=modified_img, center=(circle['r'], circle['c']), radius=circle['rad'] + RAD_MIN - 1, color=(255, 0, 0), thickness=CIRCLE_THICKNESS)
        modified_img = cv.drawMarker(position=(circle['r'], circle['c']), img=modified_img, color=(255, 0, 0), markerType=cv.MARKER_CROSS, markerSize=5)
    
    return modified_img

if __name__ == "__main__":
    if len(sys.argv) != 2:
        img_name = IMG
    else:
        img_name = sys.argv[1]

    # Load target image
    img = cv.imread(IMAGES_FOLDER + img_name)
    filtered_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rows, cols = filtered_img.shape

    # Apply gaussian blur to remove noise
    # TODO: Improve blur (not working well with noisy images for now)
    # filtered_img = cv.GaussianBlur(src=filtered_img, ksize=(3, 3), sigmaX=0)
    filtered_img = cv.medianBlur(src=filtered_img, ksize=3)
    title = "Image with gaussian blur"
    plt.title(title)
    plt.imshow(filtered_img, cmap="gray")
    plt.waitforbuttonpress()

    # Apply Sobel filters
    filtered_img = cv.Sobel(src=filtered_img, ddepth=cv.CV_8U, dx=1, dy=1)
    # XXX: Better than Sobel filters ?
    # filtered_img = cv.Canny(filtered_img, 50, 150)

    title = "Image with Sobel filters applied"
    plt.title(title)
    plt.imshow(filtered_img, cmap="gray")
    plt.waitforbuttonpress()

    rad_max = math.sqrt(rows ** 2 + cols ** 2) # Pythagore
    detected_circles = hough_circles(filtered_img, rows, cols, rad_max)

    drawn = draw_circles(img, detected_circles)
    plt.imshow(drawn)
    plt.waitforbuttonpress()