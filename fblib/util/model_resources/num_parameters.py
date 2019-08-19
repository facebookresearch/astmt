# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
