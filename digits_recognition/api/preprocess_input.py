"""
Code for pre-processing the inputs to the model.
"""
import torch
import cv2
import numpy as np


def _pad_digit(digit_image):
    h, w = digit_image.shape
    max_dim = max(h, w)

    final_dim = 28

    if max_dim > final_dim:
        if max_dim == h:
            new_h = final_dim
            new_w = int((final_dim / h) * w)
        else:
            new_w = final_dim
            new_h = int((final_dim / w) * h)

        digit_image = cv2.resize(digit_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        w = new_w
        h = new_h

    top_pad = (final_dim - h) // 2
    bottom_pad = final_dim - h - top_pad
    left_pad = (final_dim - w) // 2
    right_pad = final_dim - w - left_pad

    square_digit = np.pad(
        digit_image,
        (
            (top_pad, bottom_pad),
            (left_pad, right_pad)
        ),
        mode='constant',
        constant_values=0  # Black padding
    )

    return square_digit


def find_digits(image):
    """
    Extract digits in an image and resizes them into padded 28x28 squares.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    min_area = 10

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    coords = []
    digits_deskewed = []

    for contour in digit_contours:
        x, y, w, h = cv2.boundingRect(contour)
        digit = binary[y:y + h, x:x + w]
        digit_resized = _pad_digit(digit)
        moments = cv2.moments(digit)
        if moments["mu02"] != 0:
            skew = moments["mu11"] / moments["mu02"]
            moment = np.float32([[1, skew, -0.5 * 28 * skew], [0, 1, 0]])
            digit_deskewed = cv2.warpAffine(
                digit_resized,
                moment,
                (28, 28),
                flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
            )

            digits_deskewed.append(digit_deskewed)

        coords.append((x, y))

    digits_array = np.array(digits_deskewed, dtype=np.float32)
    digits_array = np.expand_dims(digits_array, axis=1)  # add back the channel dimension
    digits_tensor = torch.tensor(digits_array)

    return coords, digits_tensor
