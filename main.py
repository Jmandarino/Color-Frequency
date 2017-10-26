# Python 3.6
from PIL import Image
import scipy
import scipy.misc
import scipy.cluster
import codecs
import numpy as np


#### UTIL ####
def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb

def nparray_to_rbg(arr):
    # first 3 elements some cases we get a 4th rgb value (presumed for alpha)
    arr = arr[:3]
    return tuple(arr.astype('int'))

# the more clusters the more accurate, however it will be slower
NUM_CLUSTERS = 5

# read image
im = Image.open('test-fut.PNG') # change the file path
im = im.resize((500,500))
ar = scipy.misc.fromimage(im)
shape = ar.shape
ar = ar.reshape(scipy.product(shape[:2]), shape[2])

# kmeans algo
# Codes is an array for the colors
# counts is the counts of those colors
codes, dist = scipy.cluster.vq.kmeans(ar.astype(float), NUM_CLUSTERS)

vecs, dist = scipy.cluster.vq.vq(ar, codes)
counts, bins = scipy.histogram(vecs, len(codes))

# reverse sort the color codes by most occurring to least
perm = counts.argsort()
codes[perm]

for colorValue in codes:
    # print out the color in RGB and HEX
    colour = rgb_to_hex(nparray_to_rbg(colorValue))
    print ('most frequent is %s (#%s)' % (colorValue, colour))



