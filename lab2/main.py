from os import listdir

from skimage import io

# R=0 G=1 B=2
COLOR = 0


def get_filepaths(dirpath):
    files = listdir(dirpath)
    for file in files:
        yield '%s/%s' % (dirpath, file), file


for path, filename in get_filepaths('../images'):
    img = io.imread(path)

    colors_to_delete = [0, 1, 2]
    colors_to_delete.pop(COLOR)

    img[:, :, colors_to_delete] = 0.0

    io.imsave(filename, img, check_contrast=False)
