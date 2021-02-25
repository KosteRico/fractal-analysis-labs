from os import listdir

import numpy as np
from skimage import io, color
from skimage.filters import threshold_otsu


def boxcount(bin_img, k):
    S = np.add.reduceat(
        np.add.reduceat(bin_img, np.arange(0, bin_img.shape[0], k), axis=0),
        np.arange(0, bin_img.shape[1], k), axis=1)

    return len(np.where((S > 0) & (S < k * k))[0])


def fractal_dimension(img):
    thresh = threshold_otsu(img)

    binary = img < thresh

    p = min(binary.shape)

    n = 2 ** np.floor(np.log(p) / np.log(2))

    n = int(np.log(n) / np.log(2))

    sizes = 2 ** np.arange(n, 1, -1)

    counts = []
    for size in sizes:
        counts.append(boxcount(binary, size))

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


def get_filepaths(dirpath):
    files = listdir(dirpath)
    for file in files:
        yield '%s/%s' % (dirpath, file)


for path in get_filepaths('../images'):
    img = io.imread(path)
    img = color.rgb2gray(img)

    res = fractal_dimension(img)
    print("{:20}: {}".format(path, res))
