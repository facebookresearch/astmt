# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torchvision import utils as vutils

import fblib.util.pdf_visualizer as viz
from fblib.util.mypath import Path


def visualize_network(net, p):
    net.eval()
    x = torch.randn(1, 3, 512, 512)
    x.requires_grad_()

    # pdf visualizer
    y = {}
    for task in p.TASKS.NAMES:
        y[task], _ = net.forward(x, task)
    g = viz.make_dot(y, net.state_dict())
    g.view(directory=Path.save_root_dir())


class TBVisualizer(object):

    def __init__(self, tasks, min_ranges, max_ranges, batch_size):
        # Visualization settings
        self.grid_input = {
            'image': {
                'range': (0, 255),
                'normalize': True,
                'scale_each': True,
                'nrow': batch_size
            }}

        self.grid_output = {}

        for task in tasks:
            min_range = min_ranges[task]
            max_range = max_ranges[task]

            self.grid_input[task] = {
                'range': (min_range, max_range),
                'normalize': True,
                'scale_each': True,
                'nrow': batch_size
            }
            self.grid_output[task+'_pred'] = {
                'range': (min_range, max_range),
                'normalize': True,
                'scale_each': True,
                'nrow': batch_size
            }

    def visualize_images_tb(self, writer, sample, outputs, global_step, tag, phase='train'):
        """Vizualize images into Tensorboard

        writer:         Tensorboardx summary writer
        sample:         dataloader sample that contains a dict of tensors, aka images and groundtruths
        grid_input:     see function get_visualizer()
        grid_output:    see function get_visualizer()
        global_step:    global iteration num
        tag: current    iteration num to tag on tensorboard
        phase:          'train' or 'test
        """

        for k in list(self.grid_input.keys()):
            if k in sample.keys():
                elem = sample[k].detach()
                if k in {'normals', 'depth'}:
                    elem[elem == 255] = 0

                img_grid = vutils.make_grid(elem, **self.grid_input[k])
                writer.add_image(f'{k}_gt/{phase}_{tag}', img_grid, global_step)

        for k in list(outputs.keys()):
            if (k + '_pred') in self.grid_output.keys():
                output = outputs[k].detach()
                if k == 'normals':
                    elem = self._normalize(output)
                elif k in {'depth', 'albedo'}:
                    elem = output
                elif output.size()[1] == 1:
                    elem = 1 / (1 + torch.exp(-output))
                else:
                    _, argmax_pred = torch.max(output, dim=1)
                    argmax_pred = argmax_pred.type(torch.FloatTensor)
                    elem = torch.unsqueeze(argmax_pred, 1)
                img_grid = vutils.make_grid(elem, **self.grid_output[k + '_pred'])
                writer.add_image(f'{k}_pred/{phase}_{tag}', img_grid, global_step)

    @staticmethod
    def _normalize(bottom):
        qn = torch.norm(bottom, p=2, dim=1).unsqueeze(dim=1) + 1e-12

        return bottom.div(qn)
