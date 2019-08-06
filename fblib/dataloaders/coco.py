import os
import os.path

from pycocotools.coco import COCO
import torch.utils.data as data
from PIL import Image
import numpy as np
from fblib.util.mypath import Path


class CocoCaptions(data.Dataset):
    """`MS Coco Captions <http://mscoco.org/dataset/#captions-challenge2015>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    Example:
        .. code:: python
            import torchvision.datasets as dset
            import torchvision.transforms as transforms
            cap = dset.CocoCaptions(root = 'dir where images are',
                                    annFile = 'json annotation file',
                                    transform=transforms.ToTensor())
            print('Number of samples: ', len(cap))
            img, target = cap[3] # load 4th sample
            print("Image Size: ", img.size())
            print(target)
        Output: ::
            Number of samples: 82783
            Image Size: (3L, 427L, 640L)
            [u'A plane emitting smoke stream flying over a mountain.',
            u'A plane darts across a bright blue sky behind a mountain covered in snow',
            u'A plane leaves a contrail above the snowy mountain top.',
            u'A mountain that has a plane flying overheard in the distance.',
            u'A mountain view with a plume of smoke in the background']
    """
    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        target = [ann['caption'] for ann in anns]

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)


class CocoDetection(data.Dataset):
    """`MS Coco Captions <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)


class COCOSegmentation(data.Dataset):
    """`MS Coco Captions <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        split (string): Select split of the dataset, eg 'val2014' or 'train2014'
        area_range (list): Select min and max size of the objects eg [500, float("inf")]
        pascal_categories (boolean): Select only the categories of pascal
        db_root (string): Root folder where the coco dataset is stored, folder containing annotation and images folders.
        transform (callable, optional): A function/transform that  takes in a sample
            and returns a transformed version. E.g, ``transforms.ToTensor``
        retname (boolean): Return metadata about the sample
    """

    PASCAL_CAT_DICT = {'airplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5,
                       'bus': 6, 'car': 7, 'cat': 8, 'chair': 9, 'cow': 10,
                       'dining table': 11, 'dog': 12, 'horse': 13, 'motorcycle': 14, 'person': 15,
                       'potted plant': 16, 'sheep': 17, 'couch': 18, 'train': 19, 'tv': 20}

    def __init__(self,
                 split,
                 area_range=[],
                 only_pascal_categories=False,
                 mask_per_class=True,
                 db_root=Path.db_root_dir('COCO'),
                 n_samples=-1,
                 transform=None,
                 retname=True,
                 overfit=False
                 ):

        self.split = split
        self.root = os.path.join(db_root, 'images', split)
        annFile = os.path.join(db_root, 'annotations', 'instances_' + split + '.json')
        self.coco = COCO(annFile)
        self.pascal_cat_name = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'airplane',
                                'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'train', 'bottle', 'chair',
                                'dining table', 'potted plant', 'couch', 'tv']

        self.only_pascal_categories = only_pascal_categories
        if self.only_pascal_categories:
            cat_ids = self.coco.getCatIds(catNms=self.pascal_cat_name)
        else:
            cat_ids = self.coco.getCatIds()

        self.img_ids = list(self.coco.imgs.keys())

        self.ids = self.coco.getAnnIds(imgIds=self.img_ids, areaRng=area_range, catIds=cat_ids)
        self.transform = transform
        self.area_range = area_range
        self.cat_ids = cat_ids
        self.mask_per_class = mask_per_class
        self.retname = retname

        if self.mask_per_class:
            self._select_imgs()

        if n_samples > 0:
            if self.mask_per_class:
                self.img_ids = list(self.img_ids)[:n_samples]
            else:
                self.ids = self.ids[:n_samples]
        if overfit:
            n_of = 64
            self.img_ids = list(self.img_ids)[:n_of]

        # Display stats
        if self.mask_per_class:
            print("Number of images: {:d}".format(len(self.img_ids)))
        else:
            print('Number of images: {:d}\nNumber of objects: {:d}'.format(len(self.coco.imgs), len(self.ids)))

    def _select_imgs(self):
        lst = []
        for x in self.img_ids:
            ids_area = self.coco.getAnnIds(imgIds=x, areaRng=self.area_range, catIds=self.cat_ids)
            ids = self.coco.getAnnIds(imgIds=x, areaRng=[0, float('Inf')], catIds=self.cat_ids)
            if ids_area and len(ids) == len(ids_area):
                lst.append(x)
        self.img_ids = lst

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        if self.mask_per_class:
            img_id = self.img_ids[index]
            ann_meta = []
            for cat_id in self.cat_ids:
                ids = coco.getAnnIds(imgIds=img_id, catIds=cat_id)
                ann_meta.append(coco.loadAnns(ids))
            cat_id = self.cat_ids
        else:
            ann_meta = coco.loadAnns(self.ids[index])
            img_id = ann_meta[0]["image_id"]
            cat_id = ann_meta[0]['category_id']

        img_meta = coco.loadImgs(img_id)[0]
        path = img_meta['file_name']

        sample = {}
        if self.retname:
            sample['meta'] = {'image': str(path).split('.')[0],
                              'object': str(self.ids[index]),
                              'category': cat_id,
                              'im_size': (img_meta['height'], img_meta['width'])}

        try:
            img = np.array(Image.open(os.path.join(self.root, path)).convert('RGB')).astype(np.float32)
            if self.mask_per_class:
                target = np.zeros([img.shape[0], img.shape[1]])
                for ii in range(len(cat_id)):
                    ann_meta_class = ann_meta[ii]
                    target_tmp = np.zeros([img.shape[0], img.shape[1]])
                    for ann in ann_meta_class:
                        target_tmp = np.logical_or(target_tmp > 0, np.array(coco.annToMask(ann)) > 0)
                    if self.only_pascal_categories:
                        coco_cat_name = self.coco.cats[self.cat_ids[ii]]['name']
                        if coco_cat_name in self.pascal_cat_name:
                            target[target_tmp > 0] = self.PASCAL_CAT_DICT[coco_cat_name]
                    else:
                        target[target_tmp > 0] = ii + 1
            else:
                target = np.zeros([img.shape[0], img.shape[1], 1])
                for ann in ann_meta:
                    target = np.logical_or(target, np.array(coco.annToMask(ann).reshape([img.shape[0], img.shape[1], 1])))
            target = target.astype(np.float32)
        except ValueError:
            img = np.zeros((100, 100, 3))
            target = np.zeros((100, 100))
            print('Error reading image ' + str(path) + ' with object id ' + str(self.ids[index]))

        sample['image'] = img

        if self.mask_per_class:
            sample['semseg'] = target
        else:
            sample['gt'] = target

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        if self.mask_per_class:
            return len(self.img_ids)
        else:
            return len(self.ids)

    def __str__(self):
        return 'COCOSegmentation(split='+str(self.split)+', area_range='+str(self.area_range) + ')'


if __name__ == "__main__":
    from matplotlib.pyplot import imshow, show
    import torchvision.transforms as transforms
    import fblib.dataloaders.custom_transforms as tr
    transform = transforms.Compose([tr.ToTensor()])
    dataset = COCOSegmentation(split='val2017', transform=None, retname=True,
                               area_range=[1000, float("inf")], only_pascal_categories=True, overfit=True)
    for i in range(len(dataset)):
        a = dataset[i]
        print(a['semseg'].max(), a['semseg'].min())
        imshow(a['semseg']); show()