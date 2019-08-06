import os
import math
import torch, cv2
from scipy.ndimage import distance_transform_edt
import random
import numpy as np

from torchvision import utils as th_utils

# set random seed in each worker
worker_seed = lambda x: np.random.seed((torch.initial_seed()) % 2**32)


def tens2image(im):
    if im.size()[0] == 1:
        tmp = np.squeeze(im.numpy(), axis=0)
    else:
        tmp = im.numpy()
    if tmp.ndim == 2:
        return tmp
    else:
        return tmp.transpose((1, 2, 0))


def gt_from_scribble(scr, dilation=11, nocare_area=21):

    # Compute foreground
    if scr.max() == 1:
        kernel_fg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
        fg = cv2.dilate(scr.astype(np.uint8), kernel=kernel_fg).astype(scr.dtype)
    else:
        fg = scr

    # Compute nocare area
    if nocare_area is None:
        nocare = None
    else:
        kernel_nc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nocare_area, nocare_area))
        nocare = cv2.dilate(fg, kernel=kernel_nc) - fg

    return fg, nocare


def crop2fullmask(crop_mask, bbox, im=None, im_size=None, zero_pad=False, relax=0, mask_relax=True,
                  interpolation=cv2.INTER_CUBIC, scikit=False, proportional_bbox=False):
    """
    Places crop into full mask, given the bounding box of the crop
    crop_mask: the cropped element, MUST BE np.float
    """

    if scikit:
        from skimage.transform import resize as sk_resize
    assert(not(im is None and im_size is None)), 'You have to provide an image or the image size'
    if im is None:
        im_si = im_size
    else:
        im_si = im.shape
    # Borers of image
    bounds = (0, 0, im_si[1] - 1, im_si[0] - 1)

    # Valid bounding box locations as (x_min, y_min, x_max, y_max)
    bbox_valid = (max(bbox[0], bounds[0]),
                  max(bbox[1], bounds[1]),
                  min(bbox[2], bounds[2]),
                  min(bbox[3], bounds[3]))

    # Bounding box of initial mask
    if proportional_bbox:
        # bbox equals relax + bbox_init + relax. Alternative: bbox_init = get_bbox(im)
        pad_x = int(relax / (100 + 2 * relax) * (bbox[2] - bbox[0]))
        pad_y = int(relax / (100 + 2 * relax) * (bbox[3] - bbox[1]))
    else:
        pad_x = relax
        pad_y = relax

    bbox_init = (bbox[0] + pad_x,
                 bbox[1] + pad_y,
                 bbox[2] - pad_x,
                 bbox[3] - pad_y)

    if zero_pad:
        # Offsets for x and y
        offsets = (-bbox[0], -bbox[1])
    else:
        assert((bbox == bbox_valid).all())
        offsets = (-bbox_valid[0], -bbox_valid[1])

    # Simple per element addition in the tuple
    inds = tuple(map(sum, zip(bbox_valid, offsets + offsets)))

    if scikit:
        order = {cv2.INTER_CUBIC: 3, cv2.INTER_NEAREST: 0, cv2.INTER_LINEAR: 1}
        crop_mask = (255 * sk_resize(crop_mask.astype(np.uint8), (bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1),
                                    order=order[interpolation], mode='constant')).astype(crop_mask.dtype)
    else:
        crop_mask = cv2.resize(crop_mask, (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1), interpolation=interpolation)

    if crop_mask.ndim == 3:
        result_ = np.zeros((im_si[0], im_si[1], crop_mask.shape[2]), dtype=crop_mask.dtype)
        result_[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1, :] = \
            crop_mask[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1, :]

        result = np.zeros((im_si[0], im_si[1], crop_mask.shape[2]), dtype=crop_mask.dtype)
        if mask_relax:
            result[bbox_init[1]:bbox_init[3] + 1, bbox_init[0]:bbox_init[2] + 1, :] = \
                result_[bbox_init[1]:bbox_init[3] + 1, bbox_init[0]:bbox_init[2] + 1, :]
        else:
            result = result_
    else:
        result_ = np.zeros(im_si)
        result_[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1] = \
            crop_mask[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1]

        result = np.zeros(im_si)
        if mask_relax:
            result[bbox_init[1]:bbox_init[3]+1, bbox_init[0]:bbox_init[2]+1] = \
                result_[bbox_init[1]:bbox_init[3]+1, bbox_init[0]:bbox_init[2]+1]
        else:
            result = result_

    return result


def _pascal_color_map(N=256, normalized=False):
    """
    Python implementation of the color map function for the PASCAL VOC data set.
    Official Matlab version can be found in the PASCAL VOC devkit
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def overlay_semantic_mask(im, ann, alpha=0.5, colors=None, contour_thickness=None):
    im, ann = np.asarray(im, dtype=np.uint8), np.asarray(ann, dtype=np.int)
    if im.shape[:-1] != ann.shape:
        raise ValueError('First two dimensions of `im` and `ann` must match')
    if im.shape[-1] != 3:
        raise ValueError('im must have three channels at the 3 dimension')

    colors = colors or _pascal_color_map()
    colors = np.asarray(colors, dtype=np.uint8)

    mask = colors[ann]
    fg = im * alpha + (1 - alpha) * mask

    img = im.copy()
    img[ann > 0] = fg[ann > 0]

    if contour_thickness:  # pragma: no cover
        import cv2
        for obj_id in np.unique(ann[ann > 0]):
            contours = cv2.findContours((ann == obj_id).astype(
                np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            cv2.drawContours(img, contours[0], -1, colors[obj_id].tolist(),
                             contour_thickness)
    return img


def overlay_mask(im, ma, colors=None, alpha=0.5):
    assert np.max(im) <= 1.0
    if not colors:
        colors = np.load(os.path.join(os.path.dirname(__file__), 'pascal_map.npy'))/255.
    if ma.ndim == 3:
        assert len(colors) >= ma.shape[0], 'Not enough colors'
    ma = ma.astype(np.bool)
    im = im.astype(np.float32)

    # fg    = im*alpha + np.ones(im.shape)*(1-alpha) * np.array([23,23,197])/255.0

    if ma.ndim == 2:
        fg = im * alpha+np.ones(im.shape) * (1 - alpha) * colors[1, :3]   # np.array([0,0,255])/255.0
    else:
        fg = []
        for n in range(ma.ndim):
            fg.append(im * alpha + np.ones(im.shape) * (1 - alpha) * colors[1+n, :3])
    # Whiten background
    bg = im.copy()
    if ma.ndim == 2:
        bg[ma == 0] = im[ma == 0]
        bg[ma == 1] = fg[ma == 1]
        total_ma = ma
    else:
        total_ma = np.zeros([ma.shape[1], ma.shape[2]])
        for n in range(ma.shape[0]):
            tmp_ma = ma[n, :, :]
            total_ma = np.logical_or(tmp_ma, total_ma)
            tmp_fg = fg[n]
            bg[tmp_ma == 1] = tmp_fg[tmp_ma == 1]
        bg[total_ma == 0] = im[total_ma == 0]

    # [-2:] is s trick to be compatible both with opencv 2 and 3
    contours = cv2.findContours(total_ma.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cv2.drawContours(bg, contours[0], -1, (0.0, 0.0, 0.0), 1)

    return bg


def extreme_points(mask, pert):
    def find_point(id_x, id_y, ids):
        sel_id = ids[0][random.randint(0, len(ids[0]) - 1)]
        return [id_x[sel_id], id_y[sel_id]]

    # List of coordinates of the mask
    inds_y, inds_x = np.where(mask > 0.5)

    # Find extreme points
    return np.array([find_point(inds_x, inds_y, np.where(inds_x <= np.min(inds_x)+pert)),  # left
                     find_point(inds_x, inds_y, np.where(inds_x >= np.max(inds_x)-pert)),  # right
                     find_point(inds_x, inds_y, np.where(inds_y <= np.min(inds_y)+pert)),  # top
                     find_point(inds_x, inds_y, np.where(inds_y >= np.max(inds_y)-pert))  # bottom
                     ])


def bbox_points(mask, pert, ptype):

    # List of coordinates of the mask
    inds_y, inds_x = np.where(mask > 0.5)

    min_x = np.min(inds_x) + pert
    min_y = np.min(inds_y) + pert
    max_x = np.max(inds_x) - pert
    max_y = np.max(inds_y) - pert

    if ptype == 'corners':
        # Find bbox corner
        return np.array([[min_x, min_y],  # left-upper
                        [min_x, max_y],  # left-bottom
                        [max_x, min_y],  # right-upper
                        [max_x, max_y]])  # right-bottom
    else:
        print('point type {} is not implemented'.format(ptype))
        raise NotImplementedError


def distance_map(img, point):
    """
    Distance transform map to point, of size img.shape[:2],
    """
    mask = np.ones(img.shape[:2], dtype=img.dtype)
    mask[point[1], point[0]] = 0
    dt = distance_transform_edt(mask).astype(np.float32)
    dt = 1 - dt / dt.max()
    return dt


def points_in_segmentation(seg, thres=.5, n_points=1):
    """
    Return random representative point inside segmentation mask, selected in
    the region where the distance transform dt is larger than thres * max(dt)
    seg: binary segmentation
    return: point in format (x, y)
    """
    seg = np.pad(seg, pad_width=1, mode='constant')
    dt = distance_transform_edt(seg)[1:-1, 1:-1]
    dt = dt > thres * dt.max()

    inds_y, inds_x = np.where(dt > 0)

    points = []
    for i in range(n_points):
        pix_id = random.randint(0, len(inds_y) - 1)
        points.append([inds_x[pix_id], inds_y[pix_id]])

    return points


def get_bbox(mask, points=None, pad=0, zero_pad=False, proportional_bbox=False):
    if points is not None:
        inds = np.flip(points.transpose(), axis=0)
    else:
        inds = np.where(mask > 0)

    if inds[0].shape[0] == 0:
        return None

    if zero_pad:
        x_min_bound = -np.inf
        y_min_bound = -np.inf
        x_max_bound = np.inf
        y_max_bound = np.inf
    else:
        x_min_bound = 0
        y_min_bound = 0
        x_max_bound = mask.shape[1] - 1
        y_max_bound = mask.shape[0] - 1

    if proportional_bbox:
        pad_x = int(pad / 100 * (inds[1].max() - inds[1].min()))
        pad_y = int(pad / 100 * (inds[0].max() - inds[0].min()))
    else:
        pad_x = pad
        pad_y = pad

    x_min = max(inds[1].min() - pad_x, x_min_bound)
    y_min = max(inds[0].min() - pad_y, y_min_bound)
    x_max = min(inds[1].max() + pad_x, x_max_bound)
    y_max = min(inds[0].max() + pad_y, y_max_bound)

    return x_min, y_min, x_max, y_max


def farthest_boundary_point(mask, ground_truth):
    # Distance transform to the mask
    dt = distance_transform_edt(mask).astype(np.float32)

    # Get ground truth boundary pixels
    gt_shifted = np.zeros_like(ground_truth)
    gt_shifted[:-1, :-1] = ground_truth[1:, 1:]
    gt_b = gt_shifted != ground_truth

    # Get the farthest boundary pixel
    dists = np.multiply(gt_b, dt)

    return np.argwhere(dists == dists.max())


def crop_from_bbox(img, bbox, zero_pad=False):
    # Borers of image
    bounds = (0, 0, img.shape[1] - 1, img.shape[0] - 1)

    # Valid bounding box locations as (x_min, y_min, x_max, y_max)
    bbox_valid = (max(bbox[0], bounds[0]),
                  max(bbox[1], bounds[1]),
                  min(bbox[2], bounds[2]),
                  min(bbox[3], bounds[3]))

    if zero_pad:
        # Initialize crop size (first 2 dimensions)
        crop = np.zeros((bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1), dtype=img.dtype)

        # Offsets for x and y
        offsets = (-bbox[0], -bbox[1])

    else:
        assert(bbox == bbox_valid)
        crop = np.zeros((bbox_valid[3] - bbox_valid[1] + 1, bbox_valid[2] - bbox_valid[0] + 1), dtype=img.dtype)
        offsets = (-bbox_valid[0], -bbox_valid[1])

    # Simple per element addition in the tuple
    inds = tuple(map(sum, zip(bbox_valid, offsets + offsets)))

    img = np.squeeze(img)
    if img.ndim == 2:
        crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1] = \
            img[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1]
    else:
        crop = np.tile(crop[:, :, np.newaxis], [1, 1, 3])  # Add 3 RGB Channels
        crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1, :] = \
            img[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1, :]

    return crop


def fixed_resize(sample, resolution, flagval=None, scikit=False):

    if flagval is None:
        if ((sample == 0) | (sample == 1)).all():
            flagval = cv2.INTER_NEAREST
        else:
            flagval = cv2.INTER_CUBIC

    if isinstance(resolution, int):
        tmp = [resolution, resolution]
        tmp[np.argmax(sample.shape[:2])] = int(round(float(resolution)/np.min(sample.shape[:2])*np.max(sample.shape[:2])))
        resolution = tuple(tmp)

    if sample.ndim == 2 or (sample.ndim == 3 and sample.shape[2] == 3):
        if scikit:
            sample = sk_resize(sample, resolution,  order=0, mode='constant').astype(sample.dtype)
        else:
            sample = cv2.resize(sample, resolution[::-1], interpolation=flagval)
    else:
        tmp = sample
        sample = np.zeros(np.append(resolution, tmp.shape[2]), dtype=np.float32)
        for ii in range(sample.shape[2]):
            sample[:, :, ii] = cv2.resize(tmp[:, :, ii], resolution[::-1], interpolation=flagval)
    return sample


def crop_from_mask(img, mask, relax=0, zero_pad=False, proportional_bbox=False):
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, dsize=tuple(reversed(img.shape[:2])), interpolation=cv2.INTER_NEAREST)

    assert(mask.shape[:2] == img.shape[:2])

    bbox = get_bbox(mask, pad=relax, zero_pad=zero_pad, proportional_bbox=proportional_bbox)

    if bbox is None:
        return None

    crop = crop_from_bbox(img, bbox, zero_pad)

    return crop


def im_normalize(im):
    """
    Normalize image
    """
    imn = cstm_normalize(im, max_value=1)
    return imn


def cstm_normalize(im, max_value):
    """
    Normalize image to range 0 - max_value
    """
    imn = max_value * (im - im.min()) / max((im.max() - im.min()), 1e-8)
    return imn


def construct_name(p, prefix):
    """
    Construct the name of the model
    p: dictionary of parameters
    prefix: the prefix
    name: the name of the model - manually add ".pth" to follow the convention
    """
    name = prefix
    for key in p.keys():
        if (type(p[key]) != tuple) and (type(p[key]) != list):
            name = name + '_' + str(key) + '-' + str(p[key])
        else:
            name = name + '_' + str(key) + '-' + str(p[key][0])
    return name


def make_gaussian(size, sigma=10, center=None, d_type=np.float64):
    """ Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    y = y[:, np.newaxis]

    if center is None:
        x0 = y0 = size[0] // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2).astype(d_type)


def make_gt(img, labels, sigma=10, one_mask_per_point=False):
    """ Make the ground-truth for  landmark.
    img: the original color image
    labels: label with the Gaussian center(s) [[x0, y0],[x1, y1],...]
    sigma: sigma of the Gaussian.
    one_mask_per_point: masks for each point in different channels?
    """
    h, w = img.shape[:2]
    if labels is None:
        gt = make_gaussian((h, w), center=(h//2, w//2), sigma=sigma)
    else:
        labels = np.array(labels)
        if labels.ndim == 1:
            labels = labels[np.newaxis]
        if one_mask_per_point:
            gt = np.zeros(shape=(h, w, labels.shape[0]))
            for ii in range(labels.shape[0]):
                gt[:, :, ii] = make_gaussian((h, w), center=labels[ii, :], sigma=sigma)
        else:
            gt = np.zeros(shape=(h, w))
            for ii in range(labels.shape[0]):
                gt = np.maximum(gt, make_gaussian((h, w), center=labels[ii, :], sigma=sigma))

    gt = gt.astype(dtype=img.dtype)

    return gt

def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    for key, val in param.items():
        log_file.write(key+':'+str(val)+'\n')
    log_file.close()


# Converts a Tensor into an image array (numpy)
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def ind2sub(array_shape, inds):
    rows, cols = [], []
    for k in range(len(inds)):
        if inds[k] == 0:
            continue
        cols.append((inds[k].astype('int') // array_shape[1]))
        rows.append((inds[k].astype('int') % array_shape[1]))
    return rows, cols


def conform_to_max(im_shape, target_size, max_size, max_area=None, cnst=None):
    """
    Return image scale for resizing an image, so that its smallest dimension is target_size
    target_size: the target smallest side
    max_size: the maximum allowed size of any side
    max_area: the maximum allowed area of the image
    cnst: make the image downsample-able by a fixed resolution
    """
    im_size_min, i_min = np.min(im_shape[0:2]), np.argmin(im_shape[0:2])
    im_size_max, i_max = np.max(im_shape[0:2]), np.argmax(im_shape[0:2])

    if i_min == i_max:
        i_min = 0
        i_max = 1

    im_scale = [0] * 2
    im_scale[i_min] = float(target_size) / float(im_size_min)
    im_scale[i_max] = int(im_scale[i_min] * im_shape[i_max]) / float(im_size_max)

    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale[i_max] * im_size_max) > max_size:
        im_scale[i_max] = (float(max_size) / float(im_size_max))
        im_scale[i_min] = int(im_scale[i_max] * im_shape[i_min]) / float(im_size_min)

    im_size_y = im_shape[0]
    im_size_x = im_shape[1]

    # Check area of resized image
    if max_area and (im_size_y * im_scale[0] * im_size_x * im_scale[1]) > max_area:
        im_scale[i_max] = min(im_scale[i_max], math.sqrt(float(max_area) / float(im_shape[0] * im_shape[1])))
        im_scale[i_min] = min(im_scale[i_min], int(im_scale[i_max] * im_shape[i_min]) / float(im_size_min))

    # Check down-scaling
    if cnst:
        im_size_x_ = cnst * (math.ceil((im_size_x * im_scale[1] - 1.0) // cnst)) + 1
        im_size_y_ = cnst * (math.ceil((im_size_y * im_scale[0] - 1.0) // cnst)) + 1
        im_scale = [float(im_size_y_) / float(im_size_y), float(im_size_x_) / float(im_size_x)]

    return im_scale


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from fblib.dataloaders.bsds import BSDS500
    db = BSDS500()

    for i in range(len(db)):
        im = db[i]['image']
        im_scale = conform_to_max(im_shape=im.shape, target_size=600, max_size=1000, max_area=1e6)
        print(im_scale)
        im_resized = cv2.resize(im, dsize=None, fx=im_scale[1], fy=im_scale[0])
        plt.imshow(im_resized/255.)
        plt.show()
