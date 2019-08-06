from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
from fblib.util.mypath import Path


class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root=Path.db_root_dir('CIFAR100'), train=True,
                 transform=None, target_transform=None,
                 download=False, n_tasks=None, n_outputs=10, mix_labels=False, use_task_labels=False, use_orig=True,
                 overfit=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.n_tasks = n_tasks
        self.n_outputs = n_outputs
        self.mix_labels = mix_labels
        self.overfit = overfit
        self.use_task_labels = use_task_labels
        self.use_orig = use_orig

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        self.data = []
        self.labels = []
        if self.train:
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.labels += entry['labels']
                else:
                    self.labels += entry['fine_labels']
                fo.close()

            self.data = np.concatenate(self.data)
            self.data = self.data.reshape((50000, 3, 32, 32))
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.data = entry['data']
            if 'labels' in entry:
                self.labels = entry['labels']
            else:
                self.labels = entry['fine_labels']
            fo.close()
            self.data = self.data.reshape((10000, 3, 32, 32))

        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.labels_multitask = [[x]*2 for x in self.labels]

        if self.n_tasks:
            if not self.mix_labels:
                self._process_labels()
            else:
                self._process_label_mix()

        if self.overfit:
            n_of = 1000
            print('Overfitting to {} first samples'.format(n_of))
            self.data = self.data[:n_of]
            self.labels = self.labels[:n_of]
            if self.n_tasks:
                self.labels_multitask = self.labels_multitask[:n_of]

    def _process_labels(self):

        cl_per_task = self.n_outputs - 1
        temp = [[0] * self.n_tasks for i in range(len(self.labels))]
        for i in range(len(self.labels)):
            for j in range(self.n_tasks):
                if not (j * cl_per_task <= self.labels[i] < j * cl_per_task + cl_per_task):
                    temp[i][j] = cl_per_task
                else:
                    temp[i][j] = self.labels[i] - j * cl_per_task
            self.labels_multitask = temp

    def _process_label_mix(self):
        cl_per_bin = 100 // self.n_outputs
        print("cl_per_bin: {}".format(cl_per_bin))
        temp = [[0] * self.n_tasks for i in range(len(self.labels))]
        for i in range(len(self.labels)):
            for j in range(self.n_tasks):
                temp[i][j] = ((self.labels[i] + 2 * j) // cl_per_bin) % self.n_outputs
        self.labels_multitask = temp

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.n_tasks is None:
            img, target = self.data[index], self.labels[index]
        else:
            img, target = self.data[index], self.labels_multitask[index]
            if self.use_orig:
                target = [self.labels[index]] + target
            if self.use_task_labels:
                task_labels = list(range(len(target)))

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.use_task_labels:
            return img, target, task_labels
        else:
            return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]


if __name__ == '__main__':
    import torch
    import torchvision.transforms as transforms

    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]

    n_tasks = 3
    n_outputs = 10

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    db_train = CIFAR100(download=True, transform=train_transform, n_tasks=n_tasks, n_outputs=n_outputs, mix_labels=True)
    print('bla')

    trainloader = torch.utils.data.DataLoader(db_train, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

    for ii, sample in enumerate(trainloader):
        print(sample)
