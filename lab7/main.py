import cv2
import matplotlib.pyplot as plt
import numpy as np


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


def get_D(img):
    u1 = get_u(img)
    u2 = get_u(u1)
    u3 = get_u(u2)

    b1 = get_b(img)
    b2 = get_b(b1)
    b3 = get_b(b2)

    vol1 = get_vol(u1, b1)
    vol2 = get_vol(u2, b2)
    vol3 = get_vol(u3, b3)

    A_2 = (vol2 - vol1) / 2
    A_3 = (vol3 - vol2) / 2

    return 2 - (np.log(A_2) - np.log(A_3)) / (np.log(2) - np.log(3))


if __name__ == '__main__':
    img = cv2.imread('../text/img.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sizes = [i for i in range(10, 100, 10)]

    D_s = []

    for size in sizes:
        D = 0
        for i in range(0, img.shape[0], size):
            for j in range(0, img.shape[1], size):
                D += get_D(img[i:i + size, j:j + size])

        print(f'For size={size} D={D}')
        D_s.append(D)

    f = plt.figure(figsize=(10, 10))
    plt.plot(sizes, D_s)
    plt.xlabel('Sizes')
    plt.ylabel('Minkowski distance')
    plt.show()
    plt.savefig(f, 'plot.png')

    print('DONE!')
