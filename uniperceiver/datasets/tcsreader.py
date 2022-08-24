import io
from PIL import Image
import cv2
import numpy as np
try:
    from petrel_client.client import Client
except ImportError as E:
    "petrel_client.client cannot be imported"
    pass


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    return Image.open(buff)

def cv2_loader(img_bytes):
    # assert(img_bytes is not None)
    img_mem_view = memoryview(img_bytes)
    img_array = np.frombuffer(img_mem_view, np.uint8)
    imgcv2=cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    imgcv2=cv2.cvtColor(imgcv2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(imgcv2)

class TCSLoader(object):

    def __init__(self, conf_path):
        self.client = Client(conf_path)

    def __call__(self, fn):
        try:
            img_value_str = self.client.get(fn)
            img = pil_loader(img_value_str)
        except:
            try:
                img = cv2_loader(img_value_str)
            except:
                print('Read image failed ({})'.format(fn))
                return None
            else:
                return img
        else:
            return img
