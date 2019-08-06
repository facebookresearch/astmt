import os
import socket


def get_gpu_id():
    if socket.gethostname() == 'devfair049' or socket.gethostname() == 'devfair0132':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        gpu_id = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        if len(gpu_id) == 1:
            gpu_id = gpu_id[0]
        print('Using gpu_id: {}'. format(gpu_id))
    elif 'learnfair' in socket.gethostname():
        gpu_id = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        if len(gpu_id) == 1:
            gpu_id = gpu_id[0]

        print('Using gpu_id: {}'. format(gpu_id))
    elif socket.gethostname() == 'eec':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        gpu_id = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        if len(gpu_id) == 1:
            gpu_id = gpu_id[0]
        print('Using gpu_id: {}'.format(gpu_id))
    elif 'SGE_GPU' not in os.environ.keys() and socket.gethostname() != 'reinhold':
        gpu_id = -1
        print('Using CPU')
    else:
        my_gpus = os.environ['SGE_GPU'].split('\n')
        print(my_gpus)
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(my_gpus)
        print('Using gpu_id: {}'.format(' '.join(my_gpus)))
        gpu_id = [i for i, x in enumerate(my_gpus)]
        if len(gpu_id) == 1:
            gpu_id = gpu_id[0]

    return gpu_id
