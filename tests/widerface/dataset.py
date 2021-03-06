import os, json, random
from pathlib import Path
from collections import UserDict

from detectron2.structures import BoxMode
from PIL import Image

from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog,
)

# category = ['face']
# category_id = dict(zip(category, range(len(category))))


# Prepare Dataset
class WiderFace():
    def __init__(self, work_dir):
        super().__init__()
        self.category = ['face']
        self.category_id = dict(zip(self.category, range(len(self.category))))
        self.work_dir = Path(work_dir)
        self.image_dir = {}
        self.anno = {}
        for t in ('train', 'val'):
            self.image_dir[t] = self.work_dir/f'WIDER_{t}/images'
            self.anno[t] = self.work_dir/f'wider_face_split/wider_face_{t}_bbx_gt.txt'

    def load_instances(self, typ):
        assert typ in ('train', 'val'), "Only 'train' or 'val' is supported"

        dataset_dicts = []
        idx = 0

        with self.anno[typ].open() as f:
            line = f.readline()
            while line:
                im_path = self.image_dir[typ]/line.strip()
                im = Image.open(im_path)

                record = {}
                record['file_name'] = str(im_path)
                record['image_id'] = idx
                record['width'], record['height'] = im.size
                idx += 1
                im.close()

                objs = []
                num_objs = int(f.readline())
                if num_objs == 0:
                    f.readline()
                for _ in range(num_objs):
                    anno = f.readline().split()
                    x, y, w, h = [int(x) for x in anno[:4]]
                    obj = {
                        'bbox': [x, y, w, h],
                        'bbox_mode': BoxMode.XYWH_ABS,
                        'category_id': 0,
                    }
                    objs.append(obj)
                record['annotations'] = objs
                dataset_dicts.append(record)

                line = f.readline()

        return dataset_dicts                                                                            



def register(root):
    widerface = WiderFace(root) 

    for d in ['train', 'val']:
        DatasetCatalog.register(
            'widerface_' + d, lambda d=d: widerface.load_instances(d))
        MetadataCatalog.get('widerface_' + d).set(thing_classes=widerface.category)


