import re
import cv2 
import numpy as np
import pytesseract
from pytesseract import Output
from matplotlib import pyplot as plt

image = cv2.imread('1.jpeg')



# Get OCR output using Pytesseract

custom_config = r'--oem 3 --psm 6'

file = open("out.txt", "w+")
file.write(pytesseract.image_to_string(image, config=custom_config))
file.close()
