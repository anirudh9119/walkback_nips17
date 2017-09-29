import imageio
import glob
import numpy as np
from tqdm import tqdm

home = '/data/lisatmp4/ballasn/'
home2 = '/data/lisatmp3/anirudhg/coco_walkback/'
path = 'inpainting/val2014/'

images = []

for fname in tqdm(glob.glob('{}/*.jpg'.format(home+path))):
    img = imageio.imread(fname)
    if img.shape == (64, 64, 3) and img.dtype == np.uint8:
        images.append(img)

images = np.array(images)
np.savez_compressed(home2+'images.valid.npz', images)
