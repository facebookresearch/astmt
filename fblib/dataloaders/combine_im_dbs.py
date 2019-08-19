# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch.utils.data as data


class CombineIMDBs(data.Dataset):
    """
    Combine two datasets, for example to create VOC and SBD training set
    """
    def __init__(self, dataloaders, excluded=None, repeat=None):
        self.dataloaders = dataloaders
        self.excluded = excluded
        self.im_ids = []

        # Combine object lists
        for dl in dataloaders:
            for elem in dl.im_ids:
                if elem not in self.im_ids:
                    self.im_ids.append(elem)

        # Exclude
        if excluded:
            for dl in excluded:
                for elem in dl.im_ids:
                    if elem in self.im_ids:
                        self.im_ids.remove(elem)

        if repeat:
            self.repeat = repeat
            assert(len(repeat) == len(dataloaders))
        else:
            self.repeat = [1] * len(dataloaders)

        # Get object pointers
        self.im_list = []
        new_im_ids = []
        num_images = 0
        for ii, dl in enumerate(dataloaders):
            for jj, curr_im_id in enumerate(dl.im_ids):
                if (curr_im_id in self.im_ids) and (curr_im_id not in new_im_ids):
                    for r in range(self.repeat[ii]):
                        new_im_ids.append(curr_im_id)
                        self.im_list.append({'db_ii': ii, 'im_ii': jj})
                        num_images += 1

        self.im_ids = new_im_ids
        print('Combined number of images: {:d}\n'.format(num_images))

    def __getitem__(self, index):

        _db_ii = self.im_list[index]["db_ii"]
        _im_ii = self.im_list[index]['im_ii']
        # print("db_id: {}, im_id: {}".format(_db_ii, _im_ii))
        sample = self.dataloaders[_db_ii].__getitem__(_im_ii)

        if 'meta' in sample.keys():
            sample['meta']['db'] = str(self.dataloaders[_db_ii])

        return sample

    def __len__(self):
        return len(self.im_ids)

    def __str__(self):
        include_db = [str(db) for db in self.dataloaders]
        exclude_db = [str(db) for db in self.excluded]
        return 'Included datasets:'+str(include_db)+'\n'+'Excluded datasets:'+str(exclude_db)


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    import fblib.dataloaders as dataloaders

    pascal_train = dataloaders.VOC12(split='train', do_semseg=True)
    sbd = dataloaders.SBD(split=['train', 'val'], do_semseg=True)
    pascal_val = dataloaders.VOC12(split='val', do_semseg=True)
    dataset = CombineIMDBs([pascal_train, sbd], excluded=[pascal_val])

    for i, sample in enumerate(dataset):
        plt.imshow(sample['image']/255.)
        plt.show()
        plt.imshow(sample['semseg'])
        plt.show()
