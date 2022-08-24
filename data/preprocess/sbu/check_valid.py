import sys
from PIL import Image

import warnings
from glob import glob
import os

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

def check_image_size(img_path):
    try:
        im = Image.open(img_path).convert('RGB')
        # remove images with too small or too large size
        if (im.size[0] < 10 or im.size[1] < 10 or im.size[0] > 10000 or im.size[1] > 10000):
            raise Exception('')
        
    except:
        # print(sys.argv[1])
        return img_path
    else: 
        return None

def main():
    image_already_dl = list(glob("images/*"))
    print('already download {} images.'.format(len(image_already_dl)))
    for image_path in image_already_dl:
        ret = check_image_size(image_path)
        if ret is not None:
            os.remove(ret)
    
    image_already_dl = list(glob("images/*"))
    print('after check size, {} images left.'.format(len(image_already_dl)))

if __name__ == "__main__":
    print('remove images with too small or too large size')
    main()