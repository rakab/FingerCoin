from __future__ import absolute_import
from __future__ import print_function
from PIL import Image, ImageDraw
import os
from . import utils

class Gabor(object):
    """Oriented Gabor filter for improving contrast and quality

    Attributes:
        image: input image
        block: block size
    """

    def __init__(self, image, block):
        self.image = image
        self.block = block

    def process(self):
        print('Calculating orientation...')
        orientation = utils.calculate_angles(self.image, self.block)
        print('Done!')

        print('Smoothing angles...')
        orientation = utils.smooth_angles(orientation)
        print('Done!')

        self.orientation = orientation

        (x, y) = self.image.size
        im_load = self.image.load()

        print('Calculating local ridge frequency...')
        freq = utils.freq(self.image, self.block, self.orientation)
        print('Done!')

        gauss = utils.gauss_kernel(3)
        utils.apply_kernel(freq, gauss)
        for i in range(1, x / self.block - 1):
            for j in range(1, y / self.block - 1):
                kernel = utils.gabor_kernel(self.block, self.orientation[i][j], freq[i][j])
                for k in range(0, self.block):
                    for l in range(0, self.block):
                        im_load[i * self.block + k, j * self.block + l] = int(utils.apply_kernel_at( lambda x, y: im_load[x, y],
                            kernel,
                            i * self.block + k,
                            j * self.block + l))

        return self.image

class Thinning(object):
    """Make ridges 1 pixel wide

    Attributes:
        image: input image
    """

    def __init__(self,image):
        self.image = image
        self.usage = False

    def structurize(self, pixels, structure, result):
        self.usage = False

        def choose(old, new):
            if new == result:
                self.usage = True
                return 0.0
            return old

        utils.apply_kernel_with_f(pixels, structure, choose)

        return self.usage

    def process(self):
        imload = utils.load_image(self.image)
        utils.apply_to_each_pixel(imload, lambda x: 0.0 if x > 10 else 1.0)

        print('Image loading done!')

        th1 = [[1, 1, 1], [0, 1, 0], [0.1, 0.1, 0.1]]
        th2 = utils.transpose(th1)
        th3 = utils.reverse(th1)
        th4 = utils.transpose(th3)
        th5 = [[0, 1, 0], [0.1, 1, 1], [0.1, 0.1, 0]]
        th7 = utils.transpose(th5)
        th6 = utils.reverse(th7)
        th8 = utils.reverse(th5)

        thinners = [th1, th2, th3, th4, th5, th6, th7]

        self.usage = True
        while(self.usage):
            self.usage = False
            for structure in thinners:
                self.usage |= self.structurize(imload, structure, utils.flatten(structure).count(1))
            print('Single thining done')

        print('Thining phase completed!')

        utils.apply_to_each_pixel(imload, lambda x: 255.0 * (1 - x))
        utils.load_pixels(self.image, imload)

        return self.image

class Minutiae(object):
    """Find minutiae points: ridge endings and bifurcations

    Attributes:
        image: input image
    """

    def __init__(self,image):
        self.image = image

    @staticmethod
    def localminutiae(pixels, i, j):
        cells = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        values = [pixels[i + k][j + l] for k, l in cells]
        crossings = 0
        for k in range(0, 8):
            crossings += abs(values[k] - values[k + 1])
        crossings /= 2

        if pixels[i][j] == 1:
            if crossings == 1:
                return "ending"
            if crossings == 3:
                return "bifurcation"
        return "none"

    def process(self):
        pixels = utils.load_image(self.image)
        utils.apply_to_each_pixel(pixels, lambda x: 0.0 if x > 10 else 1.0)

        (x, y) = self.image.size
        result = self.image.convert("RGB")

        draw = ImageDraw.Draw(result)

        colors = {"ending" : (150, 0, 0), "bifurcation" : (0, 150, 0)}

        ellipse_size = 2
        for i in range(1, x - 1):
            for j in range(1, y - 1):
                minutiae = self.localminutiae(pixels, i, j)
                if minutiae != "none":
                    draw.ellipse([(i - ellipse_size, j - ellipse_size), (i + ellipse_size, j + ellipse_size)], outline = colors[minutiae])

        del draw

        return result
