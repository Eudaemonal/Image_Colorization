from os import listdir
from os.path import join, isfile, isdir
from typing import Tuple

import numpy as np
from PIL import Image
from resizeimage import resizeimage
import itertools
from os import makedirs
from os.path import expanduser, join


def maybe_create_folder(folder):
    makedirs(folder, exist_ok=True)

class ImagenetResizer:
    def __init__(self, source_dir: str, dest_dir: str):
        if not isdir(source_dir):
            raise Exception('Input folder does not exists: {}'
                            .format(source_dir))
        self.source_dir = source_dir

        # Destination folder
        maybe_create_folder(dest_dir)
        self.dest_dir = dest_dir

    def resize_img(self, filename: str, size: Tuple[int, int] = (224, 224)):
        """
        Resizes the image using padding
        :param filename:
        :param size:
        """
        img = Image.open(join(self.source_dir, filename))
        width, height = img.size
        orig_shape = np.array(img.size)
        wanted_shape = np.array(size)
        ratios = wanted_shape / orig_shape
        wanted_width, wanted_height = size
        ratio_w, ratio_h = wanted_width / width, wanted_height / height

        if np.alltrue(ratios > 1):
            factor = min(ratio_h, ratio_w)
            img = img.resize((int(width * factor), int(height * factor)))

        cover = resizeimage.resize_contain(img, size)
        cover.save(join(self.dest_dir, filename), 'JPEG')

    def resize_all(self, size=(224, 224)):
        for filename in listdir(self.source_dir):
            if isfile(join(self.source_dir, filename)):
                self.resize_img(filename, size)


# Run from the top folder as:
# python3 -m resize <args>
if __name__ == '__main__':
    import argparse
    from shared import dir_originals, dir_resized

    # Argparse setup
    parser = argparse.ArgumentParser(
        description='Resize images from a folder to 224x224')
    parser.add_argument('-s', '--source-folder',
                        default=dir_originals,
                        type=str,
                        metavar='FOLDER',
                        dest='source',
                        help='use FOLDER as source of the images (default: {})'
                        .format(dir_originals))
    parser.add_argument('-o', '--output-folder',
                        default=dir_resized,
                        type=str,
                        metavar='FOLDER',
                        dest='output',
                        help='use FOLDER as destination (default: {})'
                        .format(dir_resized))

    args = parser.parse_args()
    ImagenetResizer(args.source, args.output).resize_all((224, 224))
