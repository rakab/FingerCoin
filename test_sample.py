from __future__ import absolute_import
from PIL import Image, ImageDraw
import argparse

from fingercoin.preprocess import enhance


parser = argparse.ArgumentParser(description="Unit test")
parser.add_argument("image", nargs=1, help = "Path to image")
#parser.add_argument("--save", action='store_true', help = "Save result image as src_image_enhanced.gif")
args = parser.parse_args()

im = Image.open(args.image[0])
im = im.convert("L")
#im.show()

#Gabor filtering
enhanced = enhance.Gabor(im,16).process()
enhanced.show()

#Thinning
thinned = enhance.Thinning(im).process()
thinned.show()

#Minutiae extraction
final = enhance.Minutiae(im).process()
final.show()
