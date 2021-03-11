from os import listdir

from skimage import io, color


def get_filepaths(dirpath):
    files = listdir(dirpath)
    for file in files:
        yield '%s/%s' % (dirpath, file), file


for path, filename in get_filepaths('../images'):
    img = io.imread(path)
    img = color.rgb2gray(img)

    io.imsave(filename, img, check_contrast=False)
