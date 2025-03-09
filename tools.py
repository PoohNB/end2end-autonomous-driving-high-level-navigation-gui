import cv2
import numpy as np
import os
# from scipy import stats

def camera_blank(image_array):
    # Check if the image is completely blank
    first_pixel_value = image_array[0, 0]
    if np.all(image_array == first_pixel_value):
        return True

    
def camera_blankorcover(image_array,th=1):
    # Check if the image is completely blank
    first_pixel_value = image_array[0, 0]
    if np.all(image_array == first_pixel_value):
        return True

    # Convert the RGB image to grayscale
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Calculate the threshold for binary conversion
    _, thresh = cv2.threshold(gray_image, th, 255, cv2.THRESH_BINARY_INV)

    # Calculate the percentage of the image that is blank (white in the inverted binary image)
    blank_percentage = (np.sum(thresh) / 255) / (gray_image.shape[0] * gray_image.shape[1]) * 100

    # # Calculate the mean value of the pixel intensities
    # mean_value = np.mean(gray_image)
    # print(f"Mean pixel value: {mean_value:.2f}")

    # # Flatten the image array and calculate the mode (most frequent value)
    # pixel_values = gray_image.flatten()
    # mode_value, count = stats.mode(pixel_values)
    # print(f"Majority pixel value (mode): {mode_value[0]}, Count: {count[0]}")
    # print("blank_percentage",blank_percentage)
    if blank_percentage > 50:
        return True
    else:
        return False
    
def agent_vision_blank(image_array,percent = 90):
    # Check if the input is a valid grayscale image
    if len(image_array.shape) != 2:
        raise ValueError("Input image must be a grayscale image")

    # Calculate the total number of pixels
    total_pixels = image_array.size

    # Calculate the number of blank pixels (value 0)
    blank_pixels = np.sum(image_array == 0)

    # Calculate the percentage of blank pixels
    blank_percentage = (blank_pixels / total_pixels) * 100

    if blank_percentage > percent:
        return True
    else:
        return False
    
from system.modules.agent import Agent
import json

def check_agent(model_dir):

    model_path = os.listdir(model_dir)

    model_path = None
    for file in os.listdir(model_dir):
        if file.endswith(".zip"):
            model_path = os.path.join(model_dir,file)
            break

    try:
        agent = Agent(model_path)
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError, AttributeError) as e:
        print(f"Error during agent initialization: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
    else:
        agent.close()

    return True


