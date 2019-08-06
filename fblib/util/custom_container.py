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

    def forward(self, input, task=None, p=[]):
        for module in self._modules.values():
            if task is None:
                input = module(input)
            else:
                if p:
                    input = module(input, task, p)
                else:
                    input = module(input, task)
        return input
