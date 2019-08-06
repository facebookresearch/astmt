import torch
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False
import numpy as np


def qp_solver(grads):

    # Pre-compute Q (M in the paper)
    n_tasks = len(grads)
    hash_t = {i: task for i, task in enumerate(grads)}
    Q = matrix(0.0, (n_tasks, n_tasks))
    for i in range(n_tasks):
        for j in range(i, n_tasks):
            t = torch.dot(grads[hash_t[i]], grads[hash_t[j]]).cpu().data.numpy()
            Q[i, j] = t.item(0)
            Q[j, i] = Q[i, j]

    # Solve the quadratic programming problem
    p = matrix(np.zeros((n_tasks, 1), float))
    G = matrix(-np.eye(n_tasks))
    h = matrix(np.zeros((n_tasks, 1), float))

    A = matrix(np.ones((1, n_tasks), float), (1, n_tasks))
    b = matrix(1.0)
    sol = solvers.qp(Q, p, G, h, A, b)

    # Construct dictionary of a_t values
    result = {}
    for i in range(n_tasks):
        result[hash_t[i]] = sol['x'][i]

    return result


def return_optimal_gradients(grads, custom_alphas=None):

    # Vectorize gradients
    x = {}
    for task in grads:
        x[task] = torch.cat([a.view(-1) for a in grads[task]])

    # Compute alpha coefficients
    if not custom_alphas:
        alphas = qp_solver(x)
    else:
        alphas = custom_alphas

    # Compute optimal gradient
    optimal_grads = []
    for i, task in enumerate(grads):
        # print('Alpha: {0}, alpha: {1:.3f}'.format(task, alphas[task]))
        for j, param in enumerate(grads[task]):
            if i == 0:
                optimal_grads.append(alphas[task] * param)
            else:
                optimal_grads[j] += alphas[task] * param

    return optimal_grads


if __name__ == '__main__':
    grads = {}
    grads['edge'] = torch.rand(5) / 100
    grads['semseg'] = torch.rand(5) / 100
    grads['human_parts'] = torch.rand(5) / 100
    qp_solver(grads)
