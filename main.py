#!/usr/bin/python3
import sys
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
import os

from utils import progress_bar

"""Image processing 2nd assignment

Circle detector using Hough Transform

Usage : [python3] main.py [-h] [--images IMAGES [IMAGES ...]] [-s]

Authors: Tom Mansion <tom.mansion@universite-paris-saclay.fr>, Sophie Nguyen <sophie.nguyen@universite-paris-saclay.fr>
"""

IMAGES_FOLDER = "images"
# TODO: For now coins2.png is excluded (see exercise 3)
IMAGES = ["coins.png", "four.png", "fourn.png", "MoonCoin.png"]
OUTPUT_FOLDER = "outputs"

MAX_KERNEL_LENGTH = 9

# Row
R_MIN = 1
R_MAX = 100

# Column
C_MIN = 1
C_MAX = 100

RAD_MIN = 3

# Circle selection
N_CIRCLES = 4
DELTA_R = 3
DELTA_C = 3
DELTA_RAD = 3

CIRCLE_THICKNESS = 1
MARKER_SIZE = 5
ERODE_KERNEL = (3, 3)

def remove_noise(img) -> np.array:
    """Remove noise (if it exists) with median blur to make circle detection easier

    Args:
        img (np.array)      : Input image

    Returns:
        np.array            : Cleaned image
    """
    return cv.medianBlur(src=img, ksize=5)

def sobelize(img, kernel=ERODE_KERNEL) -> np.array:
    """Apply Sobel filter on an image for edge detection

    Args:
        img     (np.array)  : Input image

    Returns:
        np.array            : Output image
    """
    # Compute horizontal (grad_x) and vertical (grad_y) derivatives
    grad_x = cv.Sobel(src=img, ddepth=cv.CV_32F, dx=1, dy=0)
    grad_y = cv.Sobel(src=img, ddepth=cv.CV_32F, dx=0, dy=1)

    # Combine both derivatives to get an approximation of the gradient
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    # magnitude = math.sqrt(abs_grad_x ** 2 + abs_grad_y ** 2)
    filtered = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    ret, filtered = cv.threshold(filtered, 100, 255, cv.THRESH_BINARY)

    return cv.erode(filtered, kernel, iterations=1)


def is_local_maximum(acc, i, j, k, shape) -> bool:
    """Define if current element in 3D-accumulator is local maximum
    (according to its adjacent neighbors)

    Args:
        acc (np.array): 3D-accumulator
        i (int): Index i
        j (int): Index j
        k (int): Index k
        shape (tuple): 3D-accumulator shape

    Returns:
        bool: Is local maximum
    """
    ref = acc[i, j, k]

    if ref == 0:
        return False

    i_slice = np.s_[i - 1:i + 2] if i > 0 and i < shape[0] - \
        1 else (np.s_[i:i + 2] if i == 0 else np.s_[i - 1:])
    j_slice = np.s_[j - 1:j + 2] if j > 0 and j < shape[1] - \
        1 else (np.s_[j:j + 2] if j == 0 else np.s_[j - 1:])
    k_slice = np.s_[k - 1:k + 2] if k > 0 and k < shape[2] - \
        1 else (np.s_[k:k + 2] if k == 0 else np.s_[k - 1:])

    indices = (i_slice, j_slice, k_slice)

    local_area = acc[indices]

    maximum = np.amax(local_area)
    return maximum == ref


def get_detected_circles(acc, N_circles=N_CIRCLES) -> list:
    """Get N_circles detected circles from computed accumulator

    Args:
        acc (np.array): 3D accumulator
        TODO: Find a way to compute N_circles in image without using fixed number
        N_circles (int): Number of detected circles (the largest local maxima found) Defaults to N_CIRCLES.

    Returns:
        list: List of N_circles detected circles
    """
    rows, cols, depth = np.shape(acc)
    local_maxima = []

    get_detected_circles_progress_bar = progress_bar(
        "Finding the local maximum", rows, f"rows, {rows * cols} total pixels")

    for i in range(rows):
        for j in range(cols):
            for k in range(depth):
                if is_local_maximum(acc, i, j, k, (rows, cols, depth)):
                    local_maxima.append(
                        {"value": acc[i, j, k], "r": j, "c": i, "rad": k})

        get_detected_circles_progress_bar.update(i + 1)

    sorted_maxima = sorted(
        local_maxima, key=lambda maximum: maximum["value"])[::-1]

    detected_circles = []

    print(f"Retrieving {N_circles} largest local maxima...")
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


def hough_circles(img, rows, cols, r_min, r_max, c_min, c_max, rad_min, rad_max) -> list:
    """Fill the accumulator for each possible circle that passes through an edge pixel

    Args:
        img (np.array)  : Image of detected edges
        rows (int)      : Image height
        cols (int)      : Image width
        r_min (int)     : Minimum r axis value
        r_max (int)     : Maximum r axis value
        c_min (int)     : Minimum c axis value
        c_max (int)     : Maximum c axis value
        rad_min (int)   : Minimum circle radius
        rad_max (int)   : Maximum circle radius

    Returns:
        list            : List of N_circles detected circles
    """
    acc = np.zeros((r_max - r_min + 1, c_max - c_min + 1,
                    int(rad_max - rad_min) + 1), dtype=float)

    hough_circles_progress_bar = progress_bar(
        "Filling the accumulator", rows, f"rows, {rows * cols} total pixels")

    for y in range(0, rows):
        for x in range(0, cols):
            pixel = img[y, x]
            if pixel > 0:
                for r in range(r_min - 1, r_max):
                    for c in range(c_min - 1, c_max):
                        rad = int(math.sqrt(((y - r) ** 2) + ((x - c) ** 2)))
                        if rad >= rad_min and rad < rad_max:
                            acc[r - r_min, c - c_max, rad -
                                rad_min] += pixel / rad

        hough_circles_progress_bar.update(y + 1)

    detected_circles = get_detected_circles(acc)

    return detected_circles


def draw_circles(img, circles, r_min, c_min, rad_min, thickness=CIRCLE_THICKNESS, marker_size=MARKER_SIZE) -> np.array:
    """Draw circles (with markers on the center) to an image

    Args:
        img (np.array)      : Source image
        circles (list)      : List of circles parameters (format: {"r": int, "c": int, "rad": int})
        r_min (int)         : Minimum r axis value (used for gap as accumulator starts by 0 on each axis)
        c_min (int)         : Minimum c axis value
        rad_min (int)       : Minimum circle radius
        thickness (int)     : Circle line thickness
        marker_size (int)   : Centred marker size

    Returns:
        np.array            : Output image with drawn circles
    """
    modified_img = img.copy()

    for circle in circles:
        center = (circle['r'] + r_min - 1, circle['c'] + c_min - 1)
        modified_img = cv.circle(img=modified_img, center=center,
            radius=circle['rad'] + rad_min, color=(0, 0, 255), thickness=thickness)
        modified_img = cv.drawMarker(position=center, img=modified_img, color=(
            0, 0, 255), markerType=cv.MARKER_CROSS, markerSize=marker_size)

    return modified_img


parser = argparse.ArgumentParser(
    description="Detect circles on images in /images folder")
parser.add_argument("--images",
                    type=str,
                    nargs='+',
                    default=IMAGES,
                    help="an image to detect circles")
parser.add_argument("-s", "--save",
                    dest="save",
                    action="store_true",
                    default=False,
                    help="save output images into /output folder (default: disabled)")

if __name__ == "__main__":
    args = parser.parse_args()
    images = args.images

    # Load target image
    for image_name in images:
        img = cv.imread(f"{IMAGES_FOLDER}/{image_name}")
        print(f"CURRENT IMAGE : {image_name}")

        # Apply median blur to remove noise
        cleaned_img = remove_noise(cv.cvtColor(img, cv.COLOR_BGR2GRAY))

        # Apply Sobel filter
        filtered_img = sobelize(cleaned_img)

        # Set accumulator parameters
        rows, cols = filtered_img.shape
        # XXX: For now, r_max = rows and c_min = cols
        r_max, c_max = rows, cols
        r_min, c_min = R_MIN, C_MIN

        # Compute maximum radius using Pythagorean theorem
        rad_max = math.sqrt(r_max ** 2 + c_max ** 2)
        rad_min = RAD_MIN
        detected_circles = hough_circles(
            filtered_img, rows, cols, r_min, r_max, c_min, c_max, rad_min, rad_max)

        drawn_image = draw_circles(
            img, detected_circles, r_min, c_min, rad_min)

        plt.figure(num=image_name)
        plt.subplot(2, 2, 1)
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        plt.title("Original")
        plt.subplot(2, 2, 2)
        plt.imshow(cleaned_img, cmap="gray")
        plt.title("Image with median blur")
        plt.subplot(2, 2, 3)
        plt.imshow(filtered_img, cmap="gray")
        plt.title("Image with Sobel filter applied")
        plt.subplot(2, 2, 4)
        plt.imshow(cv.cvtColor(drawn_image, cv.COLOR_BGR2RGB))
        plt.title("Detected circles on image")

        if args.save:
            print("Saving output image")
            if not os.path.isdir(OUTPUT_FOLDER):
                os.mkdir(OUTPUT_FOLDER)
            status = cv.imwrite(f"{OUTPUT_FOLDER}/{image_name}", drawn_image)

        plt.tight_layout()
        print("Showing outputs")
        plt.show()
