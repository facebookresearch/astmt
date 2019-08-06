import torch

from fblib.layers.attention import AttentionModuleFree


class BinaryAttention:

    def __init__(self, model, ratio=2, to_binarize=None):

        if to_binarize is None:
            to_binarize = [AttentionModuleFree]
        if not isinstance(to_binarize, list):
            to_binarize = [to_binarize]

        # count the number of to_binarize layers
        count_targets = 0
        for m in model.modules():
            if type(m) in to_binarize:
                count_targets = count_targets + 1

        self.ratio = ratio

        start_range = 0
        self.bin_range = list(range(start_range, count_targets))
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        index = -1
        for m in model.modules():
            if type(m) in to_binarize:
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

        self.frozen = False
        print('Initialized binary attention module with {} layers'.format(len(self.target_modules)))

    def binarize(self, deterministic=False, nonzero=False, freeze=False, debug=False):
        self.save_params()

        for index in range(self.num_of_params):
            K = int(self.target_modules[index].view(-1).shape[0] / self.ratio)
            if deterministic:
                if nonzero:
                    bin_idx = (self.target_modules[index].view(-1) != 0).nonzero().squeeze()
                    print('Number of nonzero values: {} out of {} '
                          .format(bin_idx.shape[0], self.target_modules[index].view(-1).shape[0]))
                else:
                    _, bin_idx = torch.topk(self.target_modules[index].view(-1), k=K)
            else:
                bin_idx = torch.multinomial(input=self.target_modules[index].view(-1), num_samples=K, replacement=True)

            if debug:
                print('Should be high values')
                print(self.target_modules[index].data.view(-1)[bin_idx])

            self.target_modules[index].data[:] = 0
            self.target_modules[index].data.view(-1)[bin_idx] = 1

            if freeze:
                self.target_modules[index].requires_grad = False
                self.frozen = True

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].data.copy_(self.target_modules[index].data)

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def clip(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = self.target_modules[index].clamp(0, 1).data


if __name__ == "__main__":
    from fblib.networks.resnext_cifar_multitask import resnext20
    net = resnext20(cardinality=8, base_width=64, num_classes=10, use_orig=True,
                    n_tasks=3, bn_per_task=False, adapters=False, attention=True, squeeze=False, binary_attention=True)

    bin_op = BinaryAttention(net, ratio=16, to_binarize=AttentionModuleFree)

    print('Original Values')
    print(bin_op.target_modules[0].data.view(-1))

    print('Binarization')
    bin_op.binarize(debug=True)
    print(bin_op.target_modules[0].data.view(-1))

    print('Restore original')
    bin_op.restore()
    print(bin_op.target_modules[0].data.view(-1))

    print('Binarization')
    bin_op.binarize(deterministic=True, debug=True)
    print(bin_op.target_modules[0].data.view(-1))

    print('Add 0.5 and clamp')
    bin_op.target_modules[0].data += 0.5
    bin_op.clip()
    print(bin_op.target_modules[0].data.view(-1))
