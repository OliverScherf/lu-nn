import numpy as np
import sys
import math
import PIL.Image
from IPython.display import clear_output, Image, display
from io import StringIO

def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)


def gaussian(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * np.exp(- (x ** 2) / (2 * sigma ** 2))


def apply_bilateral_filter(source, filtered_image, x, y, diameter, sigma_i, sigma_s):
    hl = diameter/2
    i_filtered = 0
    Wp = 0
    i = 0
    while i < diameter:
        j = 0
        while j < diameter:
            neighbour_x = x - (hl - i)
            neighbour_y = y - (hl - j)
            neighbour_x = round(neighbour_x)
            neighbour_y = round(neighbour_y)
            if neighbour_x >= len(source):
                neighbour_x -= len(source)
            if neighbour_y >= len(source[0]):
                neighbour_y -= len(source[0])
            gi = gaussian(source[neighbour_x][neighbour_y] - source[x][y], sigma_i)
            gs = gaussian(distance(neighbour_x, neighbour_y, x, y), sigma_s)
            w = gi * gs
            i_filtered += source[neighbour_x][neighbour_y] * w
            Wp += w
            j += 1
        i += 1
    i_filtered = i_filtered / Wp
    filtered_image[x][y] = np.around(i_filtered)


def bilateral_filter_own(source, filter_diameter, sigma_i, sigma_s):
    filtered_image = np.zeros(source.shape)
    print("filtered image is", filtered_image)
    i = 0
    while i < len(source):
        j = 0
        while j < len(source[0]):
            apply_bilateral_filter(source, filtered_image, i, j, filter_diameter, sigma_i, sigma_s)
            j += 1
        i += 1
    return filtered_image

def showarray(a, fmt='jpeg', fileName='deep'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    arr = PIL.Image.fromarray(a)
    arr.save(fileName + ".jpg", fmt)
    display(Image(data=f.getvalue()))

if __name__ == "__main__":
    src = np.float32(PIL.Image.open('test.jpg'))
    showarray(src, fileName="readin")
    filtered_image_own = bilateral_filter_own(src, 5, 12.0, 16.0)
    showarray(filtered_image_own, fileName="filtered_bilateral")







