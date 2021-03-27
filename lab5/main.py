import cv2
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series


def get_cover_base(img, func, add):
    k, l = img.shape
    u = np.ndarray(img.shape)

    for i in range(k):
        for j in range(l):
            minmax = []

            if i > 0:
                minmax.append(img[i - 1][j])
            if i < img.shape[0] - 1:
                minmax.append(img[i + 1][j])
            if j < img.shape[1] - 1:
                minmax.append(img[i][j + 1])
            if j > 0:
                minmax.append(img[i][j - 1])
            u[i][j] = func(img[i][j] + add, func(minmax))
            # print(minmax)
    return u


def get_u(img):
    return get_cover_base(img, max, 1)


def get_b(img):
    return get_cover_base(img, min, -1)


def get_vol(u, b):
    k, l = u.shape
    vol = 0.0
    for i in range(k):
        for j in range(l):
            vol += u[i][j] - b[i][j]
    return vol


def get_A(img):
    u1 = get_u(img)
    u2 = get_u(u1)

    b1 = get_b(img)
    b2 = get_b(b1)

    vol1 = get_vol(u1, b1)
    vol2 = get_vol(u2, b2)

    return (vol2 - vol1) / 2


SEGMENT_SIZE = 5
A_BORDER = 101


# used for finding out A border
def calc_stats(l):
    s = Series(A_s)
    return s.describe()


if __name__ == '__main__':
    img = cv2.imread('../text/img.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.imwrite('gray.jpg', img)

    segmented_img = np.full(img.shape, 255)

    A_s = []

    for i in range(0, img.shape[0], SEGMENT_SIZE):
        for j in range(0, img.shape[1], SEGMENT_SIZE):
            A = get_A(img[i:i + SEGMENT_SIZE, j: j + SEGMENT_SIZE])
            # print(A)
            A_s.append(A)
            # print(A)
            if A >= A_BORDER:
                segmented_img[i:i + SEGMENT_SIZE,
                j:j + SEGMENT_SIZE].fill(0)

    plt.figure(figsize=(10, 10))

    print(calc_stats(A_s))

    plt.hist(A_s)
    plt.show()

    print('DONE!')
    cv2.imwrite('segmented.jpg', segmented_img)
