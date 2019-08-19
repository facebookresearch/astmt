# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from collections import OrderedDict
from torch.nn.modules.container import Sequential


class SequentialMultiTask(Sequential):
    """A sequential container for multiple tasks.
    Forward pass re-written to incorporate multiple tasks
    """

    def __init__(self, *args):
        super(SequentialMultiTask, self).__init__(*args)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return SequentialMultiTask(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def forward(self, input, task=None):
        for module in self._modules.values():
            if task is None:
                input = module(input)
            else:
                input = module(input, task)
        return input
