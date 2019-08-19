# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn


def traverse_graph(var):
    """
    Args:
        var: output Variable
    """

    seen = set()
    var_lst = []

    def add_nodes(var):
        if var not in seen:
            if hasattr(var, 'variable'):
                u = var.variable
                if isinstance(u, nn.Parameter):
                    var_lst.append(u)
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        add_nodes(u[0])

    # handle multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_nodes(v.grad_fn)
    else:
        add_nodes(var.grad_fn)

    return var_lst


def make_closure(loss, net):
    def closure():
        used_vars = traverse_graph(loss)
        loss.backward()

        for p in net.parameters():
            exists = False
            for v in used_vars:
                exists = (p is v)
                if exists:
                    break
            if not exists:
                p.grad = None

        return loss

    return closure


def make_closure_fast(loss, net):
    def closure():
        used_vars = set(traverse_graph(loss))
        loss.backward()

        for p in net.parameters():
            if p not in used_vars:
                p.grad = None

        return loss

    return closure


class MWENet(nn.Module):
    def __init__(self):
        super(MWENet, self).__init__()

        self.a = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.rand(1))
        self.c = nn.Parameter(torch.rand(1))

    def forward_b(self, x):
        x = self.a * x
        x = x ** self.b
        return x

    def forward_c(self, x):
        x = self.a * x
        x = x ** self.c
        return x

    def print_params(self, txt='Before'):
        print('{0}: a: {1:.7f}, b: {2:.7f}, c: {3:.7f}'.format(
            txt, self.a[0].detach().numpy(), self.b[0].detach().numpy(), self.c[0].detach().numpy()))


def perform_first_iter(net, optimizer, x):
    out_b = net.forward_b(x)
    out_c = net.forward_c(x)
    loss = (1 - out_b) + (2 - out_c)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_default_optimizer():
    print('\n Using default optimizer. All parameters should change')
    x = torch.rand(1, requires_grad=True)
    net = MWENet()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.99, weight_decay=0.001)

    # First backward to get some momentum going
    perform_first_iter(net, optimizer, x)

    # Without modified optimizer
    out_b = net.forward_b(x)
    loss = (1 - out_b)

    # Before
    net.print_params()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # After: c must get updated without being part of the graph
    net.print_params('After ')


def test_modified_optimizer():
    print('\n Using modified optimizer. parameter c should not change')
    x = torch.rand(1, requires_grad=True)
    net = MWENet()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.99, weight_decay=0.0001)

    # First backward to get some momentum going
    perform_first_iter(net, optimizer, x)

    # With modified optimizer
    out_b = net.forward_b(x)
    loss = (1 - out_b)

    # Before
    net.print_params()

    optimizer.zero_grad()
    optimizer.step(closure=make_closure(loss, net))

    # After: c SHOULD NOT get updated because it's not part of the graph
    net.print_params('After ')


if __name__ == '__main__':
    test_default_optimizer()
    test_modified_optimizer()
