#!/usr/bin/python3
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt

from utils import progress_bar

"""Image processing 2nd assignment Exercise 3

Circle detector using Hough Transform but optimized for speed using the pyramidal approach

Usage : [python3] main_optimized.py

Authors: Tom Mansion <tom.mansion@universite-paris-saclay.fr>, Sophie Nguyen <sophie.nguyen@universite-paris-saclay.fr>
"""

IMAGES_FOLDER = "images"
IMAGES = ["coins2.jpg"]
# IMAGES = ["coins2.jpg", "coins.png", "four.png", "fourn.png", "MoonCoin.png"]
OUTPUT_FOLDER = "outputs"

# Pyramid parameters
NUMBER_OF_REDUCTIONS = 4
REDUCTION_FACTOR = 2


# Rad
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


def remove_noise(img) -> np.array:
    """Remove noise (if it exists) with median blur to make circle detection easier

    Args:
        img (np.array)      : Input image

    Returns:
        np.array            : Cleaned image
    """
    # XXX: Quick fix (temp) so that small images are not much blurred as bigger ones
    ksize = 1 if img.shape[1] < 50 and img.shape[0] < 50 else 3
    return cv.medianBlur(src=img, ksize=ksize)


def sobelize(img, kernel=ERODE_KERNEL) -> np.array:
    """Apply Sobel filter on an image for edge detection

    Args:
        img     (np.array)  : Input image

    Returns:
        np.array            : Output image
    """
    # Compute horizontal (grad_x) and vertical (grad_y) derivatives
    grad_x = cv.Sobel(src=img, ddepth=cv.CV_16S, dx=1, dy=0)
    grad_y = cv.Sobel(src=img, ddepth=cv.CV_16S, dx=0, dy=1)

    # Combine both derivatives to get an approximation of the gradient
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    edges_img = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    cleaned_img = cv.threshold(edges_img, 100, 255, cv.THRESH_BINARY)[1]
    return cv.erode(cleaned_img, kernel, iterations=1)


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


def get_local_maximum(acc):
    """Get local maximum from computed accumulator

    Args:
        acc (np.array): 3D accumulator

    Returns:
        list: List of local maximum
    """
    rows, cols, depth = np.shape(acc)
    local_maxima = []

    get_local_maximum_progress_bar = progress_bar(
        "Finding the local maximum", rows, f"rows, {rows * cols} total pixels")

    for i in range(rows):
        for j in range(cols):
            for k in range(depth):
                if is_local_maximum(acc, i, j, k, (rows, cols, depth)):
                    local_maxima.append(
                        {"value": acc[i, j, k], "r": j, "c": i, "rad": k})

        get_local_maximum_progress_bar.update(i + 1)

    sorted_maxima = sorted(
        local_maxima, key=lambda maximum: maximum["value"])[::-1]

    # Display a plot of the local maximum
    # plt.figure()
    # plt.plot(range(len(sorted_maxima)), list(
    #     map(lambda lm: lm["value"], sorted_maxima)))
    # plt.title("Sorted local maximum")

    return sorted_maxima


def get_top_detected_circles(local_maxima, N_circles=N_CIRCLES) -> list:
    """Get N_circles detected circles from computed local_maxima

    Args:
        local_maxima (list): List of local maximum
        N_circles (int): Number of detected circles (the largest local maxima found) Defaults to N_CIRCLES.

    Returns:
        list: List of N_circles detected circles
    """

    detected_circles = []

    print(f"Retrieving {N_circles} largest local maxima...")
    for local_max in local_maxima:
        ignore = False
        if len(detected_circles) == N_circles:
            break
        for circle in detected_circles:
            if (abs(circle['r'] - local_max['r']) < DELTA_R
                and abs(circle['c'] - local_max['c']) < DELTA_C
                    and abs(circle['rad'] - local_max['rad']) < DELTA_RAD):
                ignore = True
                break
        if not ignore:
            detected_circles.append(local_max)
    return detected_circles


def get_most_detected_circles(local_maxima, seuil_ratio=SEUIL_RATIO) -> list:
    """Get all the most detected circles from the computed local_maxima

    Args:
        local_maxima (list): List of local maximum
        seuil_ratio (float): ratio (between 0 and 1) of detection
            For exemple, if the max local maximum is 100, and the seuil_ratio is 0.5,
            all the local maxima with a value > 50 will be considered as detected circles
            Defaults to SEUIL_RATIO.

    Returns:
        list: List of detected circles
    """
    if local_maxima is None or len(local_maxima) == 0:
        return []
    max_value = max(lm["value"] for lm in local_maxima)
    level_of_acceptance = max_value * seuil_ratio
    detected_circles = []

    print(f"Retrieving top {seuil_ratio*100}% largest local maxima...")
    for local_max in local_maxima:
        if local_max['value'] < level_of_acceptance:
            continue

        ignore = False
        for circle in detected_circles:
            if (abs(circle['r'] - local_max['r']) < DELTA_R
                and abs(circle['c'] - local_max['c']) < DELTA_C
                    and abs(circle['rad'] - local_max['rad']) < DELTA_RAD):
                ignore = True
                break
        if not ignore:
            detected_circles.append(local_max)
    print(f"{len(detected_circles)} circles detected with a level of acceptance of {seuil_ratio*100}%")
    return detected_circles


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

    local_maxima = get_local_maximum(acc)
    top_detected_circles = get_top_detected_circles(local_maxima[:])
    most_detected_circles = get_most_detected_circles(local_maxima[:])

    return top_detected_circles, most_detected_circles


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


def reduce_image(img):
    """Reduce image size by half

    Args:
        img (np.array): Input image

    Returns:
        np.array: Reduced image
    """
    return cv.resize(img, None, fx=(1 / REDUCTION_FACTOR), fy=(1 / REDUCTION_FACTOR))


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
            #     Multiply the radius and the coordinates by the reduction factor
            for j in range(0, len(most_detected_circles)):
                most_detected_circles[j]['r'] *= int(math.pow(
                    REDUCTION_FACTOR, iteration))
                most_detected_circles[j]['c'] *= int(math.pow(
                    REDUCTION_FACTOR, iteration))
                most_detected_circles[j]['rad'] *= int(math.pow(
                    REDUCTION_FACTOR, iteration))

            #   Add the detected circles to the list
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
