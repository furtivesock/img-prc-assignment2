#!/usr/bin/python3
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt

from main import remove_noise, sobelize, get_local_maxima, get_top_detected_circles, get_most_detected_circles, draw_circles
from utils import progress_bar

"""Image processing 2nd assignment Exercise 3

Circle detector using Hough Transform but optimized for speed using the pyramidal approach

Usage : [python3] main_optimized.py

Authors: Tom Mansion <tom.mansion@universite-paris-saclay.fr>, Sophie Nguyen <sophie.nguyen@universite-paris-saclay.fr>
"""

IMAGES_FOLDER = "images"
IMAGES = ["coins2.jpg"]
OUTPUT_FOLDER = "outputs"

# Pyramid parameters
NUMBER_OF_REDUCTIONS = 4
REDUCTION_FACTOR = 2

# Radius
RAD_MIN = 3

# Circle selection
N_CIRCLES = 4
DELTA_R = 3
DELTA_C = 3
DELTA_RAD = 3
SEUIL_RATIO = 0.75

CIRCLE_THICKNESS = 1
MARKER_SIZE = 5
ERODE_KERNEL = (20, 20)


def hough_circles(img, rows, cols,
                  rows_max_search_range,
                  rows_min_search_range,
                  columns_max_search_range,
                  columns_min_search_range,
                  rad_max, rad_min) -> list:
    """Fill the accumulator for each possible circle that passes through an edge pixel

    Args:
        img (np.array)  : Image of detected edges
        rows (int)      : Image height
        cols (int)      : Image width

    Returns:
        list            : List of N_circles detected circles
    """
    acc = np.zeros((rows, cols, rad_max), dtype=float)

    hough_circles_progress_bar = progress_bar(
        "Filling the accumulator", rows, f"rows, {rows * cols} total pixels")

    for y in range(0, rows):
        for x in range(0, cols):
            if img[y, x] > 0:
                for r in range(-rows_max_search_range, rows_max_search_range):
                    if abs(r) < rows_min_search_range or y + r < 0 or y + r >= rows:
                        continue

                    for c in range(-columns_max_search_range, columns_max_search_range):
                        if abs(c) < columns_min_search_range or x + c < 0 or x + c >= cols:
                            continue

                        rad = int(math.sqrt(r ** 2 + c ** 2))
                        if rad >= rad_min and rad <= rad_max:
                            acc[y + r, x + c, rad - 1] += 1.0 / rad

        hough_circles_progress_bar.update(y + 1)

    local_maxima = get_local_maxima(acc, display_plot=False)
    top_detected_circles = get_top_detected_circles(local_maxima[:])
    most_detected_circles = get_most_detected_circles(local_maxima[:])

    return top_detected_circles, most_detected_circles


def reduce_image(img):
    """Reduce image size by half

    Args:
        img (np.array): Input image

    Returns:
        np.array: Reduced image
    """
    return cv.resize(img, None, fx=(1 / REDUCTION_FACTOR), fy=(1 / REDUCTION_FACTOR))


def draw_circles(img, circles, thickness=CIRCLE_THICKNESS, marker_size=MARKER_SIZE) -> np.array:
    """Draw circles (with markers on the center) to an image
    Args:
        img (np.array)      : Source image
        circles (list)      : List of circles parameters (format: {"r": int, "c": int, "rad": int})
        thickness (int)     : Circle line thickness
        marker_size (int)   : Centred marker size
    Returns:
        np.array            : Output image with drawn circles
    """
    modified_img = img.copy()

    for circle in circles:
        center = (circle['r'], circle['c'])
        modified_img = cv.circle(img=modified_img, center=center, radius=circle['rad'], color=(
            0, 0, 255), thickness=thickness)
        modified_img = cv.drawMarker(position=center, img=modified_img, color=(
            0, 0, 255), markerType=cv.MARKER_CROSS, markerSize=marker_size)

    return modified_img


if __name__ == "__main__":
    # Load target image
    for image_name in IMAGES:
        plt.figure(num=image_name)
        img = cv.imread(f"{IMAGES_FOLDER}/{image_name}")
        print(f"CURRENT IMAGE : {image_name}")
        total_top_detected_circles = []
        total_most_detected_circles = []

        # Apply median blur to remove noise
        cleaned_img = remove_noise(cv.cvtColor(img, cv.COLOR_BGR2GRAY))

        # Resize the image NUMBER_OF_REDUCTIONS times
        resized_images = [cleaned_img]

        for iteration in range(0, NUMBER_OF_REDUCTIONS):
            resized_images.append(reduce_image(resized_images[iteration]))

        # Detect circles on each image
        for iteration in range(NUMBER_OF_REDUCTIONS, -1, -1):
            print(f"\tIteration {iteration+1}/{NUMBER_OF_REDUCTIONS + 1}")
            image_to_process = resized_images[iteration]

            # Apply Sobel filter
            filtered_img = sobelize(image_to_process)

            rows, cols = image_to_process.shape

            # Calculate the iteration's parameters
            rows_max_search_range = 8
            rows_min_search_range = 4

            columns_max_search_range = 8
            columns_min_search_range = 4

            rad_max = 8
            rad_min = 4

            top_detected_circles, most_detected_circles = hough_circles(
                img=filtered_img,
                rows=rows,
                cols=cols,
                rows_max_search_range=rows_max_search_range,
                rows_min_search_range=rows_min_search_range,
                columns_max_search_range=columns_max_search_range,
                columns_min_search_range=columns_min_search_range,
                rad_max=rad_max,
                rad_min=rad_min
            )

            # Draw circles on the image
            top_detected_circles_image = draw_circles(
                image_to_process, top_detected_circles,
                thickness=NUMBER_OF_REDUCTIONS - iteration + 1,
                marker_size=NUMBER_OF_REDUCTIONS - iteration + 5
            )
            most_detected_circles_image = draw_circles(
                image_to_process, most_detected_circles,
                thickness=NUMBER_OF_REDUCTIONS - iteration + 1,
                marker_size=NUMBER_OF_REDUCTIONS - iteration + 5,
            )

            # Show image
            plt.subplot(7, 2, 1 + iteration * 2)
            plt.imshow(cv.cvtColor(
                top_detected_circles_image, cv.COLOR_BGR2RGB))
            plt.title(f"Top detected circles (iteration {iteration+1})")
            plt.subplot(7, 2, 2 + iteration * 2)
            plt.imshow(cv.cvtColor(
                most_detected_circles_image, cv.COLOR_BGR2RGB))
            plt.title(f"Most detected circles (iteration {iteration+1})")

            # Add the detected circles to the list
            # Multiply the radius and the coordinates by the reduction factor
            for j in range(0, len(most_detected_circles)):
                most_detected_circles[j]['r'] *= int(math.pow(
                    REDUCTION_FACTOR, iteration))
                most_detected_circles[j]['c'] *= int(math.pow(
                    REDUCTION_FACTOR, iteration))
                most_detected_circles[j]['rad'] *= int(math.pow(
                    REDUCTION_FACTOR, iteration))

            # Add the detected circles to the list
            total_top_detected_circles += top_detected_circles
            total_most_detected_circles += most_detected_circles
            print(
                f"\t\tTop detected circles: {len(total_top_detected_circles)}")
            print(
                f"\t\tMost detected circles: {len(total_most_detected_circles)}")

        # Draw circles on the final image
        total_top_detected_circles_image = draw_circles(
            img, total_top_detected_circles,
            thickness=3, marker_size=5
        )
        total_most_detected_circles_image = draw_circles(
            img, total_most_detected_circles,
            thickness=3, marker_size=5
        )

        plt.subplot(7, 2, NUMBER_OF_REDUCTIONS * 2 + 3)
        plt.imshow(cv.cvtColor(
            total_top_detected_circles_image, cv.COLOR_BGR2RGB))
        plt.title(f"Final top detected circles")
        plt.subplot(7, 2, NUMBER_OF_REDUCTIONS * 2 + 4)
        plt.imshow(cv.cvtColor(
            total_most_detected_circles_image, cv.COLOR_BGR2RGB))
        plt.title(f"Final most detected circles")

        plt.show()
