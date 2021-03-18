from os import listdir

from skimage import io, color

N = 200


def measure_part(arr):
    v_ol = 0

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            minmax = []
            if i > 0:
                minmax.append(arr[i - 1][j])
            if i < arr.shape[0] - 1:
                minmax.append(arr[i + 1][j])
            if j < arr.shape[1] - 1:
                minmax.append(arr[i][j + 1])
            if j > 0:
                minmax.append(arr[i][j - 1])

            ma = max(minmax)
            mi = min(minmax)

            u = max(arr[i][j] + 1, ma)
            b = min(arr[i][j] - 1, mi)
            v_ol += u - b

    return v_ol / 2


def measure_image(img):
    res = 0

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res += measure_part(img[i: i + N, j: j + N])
            print(res)

    return res


def get_filepaths(dirpath):
    files = listdir(dirpath)
    for file in files:
        yield '%s/%s' % (dirpath, file), file


for path, filename in get_filepaths('../images'):
    img = io.imread(path)
    img = color.rgb2gray(img)

    print(f'{filename}: {measure_image(img)}')
