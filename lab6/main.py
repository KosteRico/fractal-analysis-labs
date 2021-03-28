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


def get_A_s(img, betas):
    u_s = [img]
    b_s = [img]

    u_s.append(get_u(u_s[-1]))
    b_s.append(get_b(b_s[-1]))

    for beta in betas:
        u_s.append(get_u(u_s[-1]))
        b_s.append(get_b(b_s[-1]))

        vol1 = get_vol(u_s[-2], b_s[-2])
        vol2 = get_vol(u_s[-1], b_s[-1])

        A = (vol2 - vol1) / 2

        print(vol1, vol2)

        res = np.log(A) / np.log(beta)

        print(f'beta={beta}: A={res}')

        yield res


if __name__ == '__main__':
    img = cv2.imread('../text/img.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    betas = [i for i in range(2, 11)]

    A_s = [i for i in get_A_s(img, betas)]

    plt.figure(figsize=(10, 10))
    line, = plt.plot(betas, A_s)

    plt.xlabel('beta')
    plt.ylabel('log(A) / log(delta)')

    plt.show()
