from __future__ import division

import numpy.random as random
import numpy as np
import torch
import cv2
import math
import fblib.util.helpers as helpers
import torchvision.transforms.functional as F_vision


class ScaleNRotate(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        two possibilities:
        1.  rots (tuple): (minimum, maximum) rotation angle
            scales (tuple): (minimum, maximum) scale
        2.  rots [list]: list of fixed possible rotation angles
            scales [list]: list of fixed possible scales
    """
    def __init__(self, rots=(-30, 30), scales=(.75, 1.25), semseg=False, flagvals=None):
        assert (isinstance(rots, type(scales)))
        self.rots = rots
        self.scales = scales
        self.semseg = semseg
        self.flagvals = flagvals

    def __call__(self, sample):

        if type(self.rots) == tuple:
            # Continuous range of scales and rotations
            rot = (self.rots[1] - self.rots[0]) * random.random() - \
                  (self.rots[1] - self.rots[0])/2

            sc = (self.scales[1] - self.scales[0]) * random.random() - \
                 (self.scales[1] - self.scales[0]) / 2 + 1
        elif type(self.rots) == list:
            # Fixed range of scales and rotations
            rot = self.rots[random.randint(0, len(self.rots))]
            sc = self.scales[random.randint(0, len(self.scales))]

        for elem in sample.keys():
            if 'meta' in elem:
                continue

            tmp = sample[elem]

            h, w = tmp.shape[:2]
            center = (w / 2, h / 2)
            assert(center != 0)  # Strange behaviour warpAffine
            M = cv2.getRotationMatrix2D(center, rot, sc)
            if self.flagvals is None:
                if ((tmp == 0) | (tmp == 1)).all():
                    flagval = cv2.INTER_NEAREST
                elif 'gt' in elem and self.semseg:
                    flagval = cv2.INTER_NEAREST
                else:
                    flagval = cv2.INTER_CUBIC
            else:
                flagval = self.flagvals[elem]

            if elem == 'normals':
                # Rotate Normals properly
                in_plane = np.arctan2(tmp[:, :, 0], tmp[:, :, 1])
                nrm_0 = np.sqrt(tmp[:, :, 0] ** 2 + tmp[:, :, 1] ** 2)
                rot_rad= rot * 2 * math.pi / 360
                tmp[:, :, 0] = np.sin(in_plane + rot_rad) * nrm_0
                tmp[:, :, 1] = np.cos(in_plane + rot_rad) * nrm_0

            tmp = cv2.warpAffine(tmp, M, (w, h), flags=flagval)

            sample[elem] = tmp

        return sample

    def __str__(self):
        return 'ScaleNRotate:(rot='+str(self.rots)+',scale='+str(self.scales)+')'


class ScaleNRotateRandom(object):
    """Scale (zoom-in, zoom-out) in a random place and Rotate the image and the ground truth.
    Args:
        two possibilities:
        1.  rots (tuple): (minimum, maximum) rotation angle
            scales (tuple): (minimum, maximum) scale
        2.  rots [list]: list of fixed possible rotation angles
            scales [list]: list of fixed possible scales
    """
    def __init__(self, rots=(-30, 30), scales=(.75, 1.25), semseg=False, margin_per=(0, 0), consider_void_pixels=False):

        self.rots = rots
        self.scales = scales
        self.semseg = semseg
        self.consider_void_pixels = consider_void_pixels
        self.margin = margin_per

    def __call__(self, sample):

        if type(self.rots) == tuple:
            rot = (self.rots[1] - self.rots[0]) * random.random() + self.rots[0]

        elif type(self.rots) == list:
            rot = self.rots[random.randint(0, len(self.rots))]

        if isinstance(self.scales, tuple):
            sc = (self.scales[1] - self.scales[0]) * random.random() + self.scales[0]
        elif isinstance(self.scales, list):
            sc = self.scales[random.randint(0, len(self.scales))]

        center = None
        for elem in sample.keys():
            if 'meta' in elem:
                continue

            tmp = sample[elem]

            h, w = tmp.shape[:2]
            if center is None:
                if sc == 1:
                    center = (h/2, w/2)
                else:
                    mask = np.ones((h, w))
                    if self.margin != (0, 0):
                        mask[0:int(h*self.margin[0]/100.), :] = 0
                        mask[h-int(h * self.margin[0]/100.):, :] = 0
                        mask[:, 0:int(w * self.margin[1]/100.)] = 0
                        mask[:, w-int(w * self.margin[1]/100.):] = 0
                    ind_raw = np.arange(1, w * h + 1)
                    ind_raw = np.reshape(ind_raw, (h, w))
                    val_ind_raw = np.multiply(ind_raw, mask)
                    if self.consider_void_pixels:
                        val_ind_raw = np.multiply(val_ind_raw, np.logical_not(sample['void_pixels']))
                    val_ind_raw = np.reshape(val_ind_raw, (1, -1))
                    val_ind_raw = val_ind_raw[val_ind_raw != 0]
                    ind = random.randint(0, len(val_ind_raw)+1)
                    x_c, y_c = np.unravel_index(int(val_ind_raw[ind] - 1), (h, w))
                    center = (x_c, y_c)

            assert(center != 0)  # Strange behaviour warpAffine
            M = cv2.getRotationMatrix2D(center, rot, sc)

            if ((tmp == 0) | (tmp == 1)).all():
                flagval = cv2.INTER_NEAREST
            elif 'gt' in elem and self.semseg:
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC
            tmp = cv2.warpAffine(tmp, M, (w, h), flags=flagval)

            sample[elem] = tmp

        return sample

    def __str__(self):
        return 'ScaleNRotate:(rot='+str(self.rots)+',scale='+str(self.scales)+')'


class FixedResize(object):
    """Resize the image and the ground truth to specified resolution.
    Args:
        resolutions (dict): the list of resolutions
    """
    def __init__(self, resolutions=None, flagvals=None, scikit=False, use_mil=False, mildil=False):
        self.resolutions = resolutions
        self.flagvals = flagvals
        self.scikit = scikit
        self.use_mil = use_mil
        if self.flagvals is not None:
            assert(len(self.resolutions) == len(self.flagvals))
        if self.use_mil:
            self.process_mil = MILtoIdxs(d_size=self.resolutions[list(self.resolutions.keys())[0]], mildil=mildil)

    def __call__(self, sample):

        # Fixed range of scales
        if self.resolutions is None:
            return sample

        elems = list(sample.keys())
        for elem in elems:
            if elem == 'idxh' or elem == 'idxv' or (elem == 'edge' and 'idxh' in sample.keys()):
                continue
            if 'meta' in elem or 'bbox' in elem or ('extreme_points_coord' in elem and elem not in self.resolutions):
                continue
            if 'extreme_points_coord' in elem and elem in self.resolutions:
                bbox = sample['bbox']
                crop_size = np.array([bbox[3]-bbox[1]+1, bbox[4]-bbox[2]+1])
                res = np.array(self.resolutions[elem]).astype(np.float32)
                sample[elem] = np.round(sample[elem]*res/crop_size).astype(np.int)
                continue
            if elem in self.resolutions:
                if self.resolutions[elem] is None:
                    continue
                if isinstance(sample[elem], list):
                    if sample[elem][0].ndim == 3:
                        output_size = np.append(self.resolutions[elem], [3, len(sample[elem])])
                    else:
                        output_size = np.append(self.resolutions[elem], len(sample[elem]))
                    tmp = sample[elem]
                    sample[elem] = np.zeros(output_size, dtype=np.float32)
                    for ii, crop in enumerate(tmp):
                        if self.flagvals is None:
                            sample[elem][..., ii] = helpers.fixed_resize(crop, self.resolutions[elem], scikit=self.scikit)
                        else:
                            sample[elem][..., ii] = helpers.fixed_resize(crop, self.resolutions[elem], flagval=self.flagvals[elem], scikit=self.scikit)
                else:
                    if self.flagvals is None:
                        sample[elem] = helpers.fixed_resize(sample[elem], self.resolutions[elem], scikit=self.scikit)
                    else:
                        sample[elem] = helpers.fixed_resize(sample[elem], self.resolutions[elem], flagval=self.flagvals[elem], scikit=self.scikit)

                    if elem == 'normals':
                        N1, N2, N3 = sample[elem][:, :, 0], sample[elem][:, :, 1], sample[elem][:, :, 2]
                        Nn = np.sqrt(N1 ** 2 + N2 ** 2 + N3 ** 2) + np.finfo(np.float32).eps
                        sample[elem][:, :, 0], sample[elem][:, :, 1], sample[elem][:, :, 2] = N1/Nn, N2/Nn, N3/Nn
            else:
                del sample[elem]

        if self.use_mil:
            sample = self.process_mil(sample)

        return sample

    def __str__(self):
        return 'FixedResize:'+str(self.resolutions)


class RandomResize(object):
    """Randomly resize the image and the ground truth to specified scales.
    Args:
        scales (list): the list of scales
    """
    def __init__(self, scales=[0.5, 0.8, 1]):
        self.scales = scales

    def __call__(self, sample):

        # Fixed range of scales
        sc = self.scales[random.randint(0, len(self.scales))]

        for elem in sample.keys():
            if 'meta' in elem or 'bbox' in elem:
                continue

            tmp = sample[elem]

            if ((tmp == 0) | (tmp == 1)).all():
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC

            tmp = cv2.resize(tmp, None, fx=sc, fy=sc, interpolation=flagval)

            sample[elem] = tmp

        return sample

    def __str__(self):
        return 'RandomResize:'+str(self.scales)


class FixedResizeRatio(object):
    """Fixed resize for the image and the ground truth to specified scale.
    Args:
        scales (float): the scale
    """
    def __init__(self, scale=None, flagvals=None, use_mil=False):
        self.scale = scale
        self.flagvals = flagvals
        self.use_mil = use_mil
        if use_mil:
            self.create_mil = MILtoIdxs(sc=scale)

    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem:
                continue

            if elem in self.flagvals:
                if self.flagvals[elem] is None:
                    continue

                tmp = sample[elem]
                tmp = cv2.resize(tmp, None, fx=self.scale, fy=self.scale, interpolation=self.flagvals[elem])

                sample[elem] = tmp

        if self.use_mil:
             sample = self.process_mil(sample)

        return sample

    def __str__(self):
        return 'FixedResizeRatio: '+str(self.scale)


class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        if random.random() < 0.5:
            for elem in sample.keys():
                if 'meta' in elem or elem == 'idxv':
                    continue
                elif elem == 'idxh' and sample[elem] is not None:
                    #  Flip the horizontal indices of MIL bags: must be 1-indexed
                    _, w = sample['edge'].shape
                    sample[elem] = w - (sample[elem] - 1)
                else:
                    tmp = sample[elem]
                    tmp = cv2.flip(tmp, flipCode=1)
                    sample[elem] = tmp

                if elem == 'normals':
                    sample[elem][:, :, 0] *= -1

        return sample

    def __str__(self):
        return 'RandomHorizontalFlip'


class RandomPointsInMask(object):
    """
    Returns n_points random point(s) (Gaussian) in a given binary mask
    sigma: sigma of Gaussian
    thres: threshold of distance transform. For larger thres, point closer to the center of the object
    elem: which element of the sample to choose as the binary mask
    """
    def __init__(self, sigma=10, thres=.5, elem='gt', n_points=1):
        self.sigma = sigma
        self.thres = thres
        self.elem = elem
        self.n_points = n_points

    def __call__(self, sample):
        if sample[self.elem].ndim == 3:
            raise ValueError('RandomPointsInMask not implemented for multiple object per image.')
        _target = sample[self.elem]

        if _target.max() == 0:
            _heat_points = np.zeros(_target.shape, dtype=_target.dtype)
        else:
            _points = helpers.points_in_segmentation(_target, thres=self.thres, n_points=self.n_points)
            _heat_points = helpers.make_gt(_target, _points, sigma=self.sigma)
        sample['points'] = _heat_points

        return sample

    def __str__(self):
        return 'RandomPointsInMask:(sigma='+str(self.sigma)+', thres='+str(self.thres)+', elem='+str(self.elem)+')'


class BboxPoints(object):
    """
    Returns four points on the bounding box (with some random perturbation) in a given binary mask
    sigma: sigma of Gaussian to create a heatmap from a point
    pert: number of pixels fo the maximum perturbation
    elem: which element of the sample to choose as the binary mask

    ptype: 'corners' for the corners of the bbox
    """
    def __init__(self, sigma=10, pert=0, elem='gt', ptype='corners'):
        self.sigma = sigma
        self.pert = pert
        self.elem = elem
        self.ptype = ptype

    def __call__(self, sample):
        if sample[self.elem].ndim == 3:
            raise ValueError('ExtremePoints not implemented for multiple object per image.')
        _target = sample[self.elem]
        if np.max(_target) == 0:
            sample['bbox_points'] = np.zeros(_target.shape, dtype=_target.dtype) #  TODO: handle one_mask_per_point case
        else:
            _points = helpers.bbox_points(_target, self.pert, self.ptype)
            sample['bbox_points'] = helpers.make_gt(_target, _points, sigma=self.sigma, one_mask_per_point=False)

        return sample

    def __str__(self):
        return 'BboxPoints:(sigma=' + str(self.sigma) + ', pert=' + str(self.pert) + ', elem='+str(self.elem)\
               + ', ptype=' + str(self.ptype) + ')'


class ExtremePoints(object):
    """
    Returns the four extreme points (left, right, top, bottom) (with some random perturbation) in a given binary mask
    sigma: sigma of Gaussian to create a heatmap from a point
    pert: number of pixels fo the maximum perturbation
    elem: which element of the sample to choose as the binary mask
    """
    def __init__(self, sigma=10, pert=0, elem='gt'):
        self.sigma = sigma
        self.pert = pert
        self.elem = elem

    def __call__(self, sample):
        if sample[self.elem].ndim == 3:
            raise ValueError('ExtremePoints not implemented for multiple object per image.')
        _target = sample[self.elem]
        if np.max(_target) == 0:
            sample['extreme_points'] = np.zeros(_target.shape, dtype=_target.dtype) #  TODO: handle one_mask_per_point case
        else:
            _points = helpers.extreme_points(_target, self.pert)
            sample['extreme_points'] = helpers.make_gt(_target, _points, sigma=self.sigma, one_mask_per_point=False)

        return sample

    def __str__(self):
        return 'ExtremePoints:(sigma='+str(self.sigma)+', pert='+str(self.pert)+', elem='+str(self.elem)+')'


class DistanceMap(object):
    """
    Distance Transform map for a Point or a list of points
    """
    def __init__(self, elem='gt', p_type='ExtremePoints', max_val=1, pert=0):
        self.elem = elem
        self.max_val = max_val
        self.p_type = p_type
        self.pert = pert

    def __call__(self, sample):

        tmp = sample[self.elem]
        distmap = np.zeros(tmp.shape, dtype=tmp.dtype)

        if tmp.max() != 0:  # Handle empty mask case
            if self.p_type == 'ExtremePoints':
                points = helpers.extreme_points(tmp, pert=self.pert)
            elif self.p_type == 'BboxCenter':
                b = helpers.get_bbox(tmp)
                points = [[int((b[0]+b[2])/2), int((b[1]+b[3])/2)], ]

            for point in points:
                distmap = np.maximum(distmap, helpers.distance_map(tmp, point))

            distmap = distmap - distmap.min()
            distmap = self.max_val * distmap / distmap.max()

        sample['distmap'] = distmap

        return sample

    def __str__(self):
        return 'DistanceMap: ' + str(self.elem)


class ConcatInputs(object):

    def __init__(self, elems=('image', 'point')):
        self.elems = elems

    def __call__(self, sample):

        res = sample[self.elems[0]]

        for elem in self.elems[1:]:
            assert(sample[self.elems[0]].shape[:2] == sample[elem].shape[:2])

            # Check if third dimension is missing
            tmp = sample[elem]
            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            res = np.concatenate((res, tmp), axis=2)

        sample['concat'] = res

        return sample

    def __str__(self):
        return 'ExtremePoints:'+str(self.elems)


class FillWithOnes(object):
    """
    Helper transform that returns all-ones ground-truth, in the size of the element
    """
    def __init__(self, elem='gt'):
        self.elem = elem

    def __call__(self, sample):
        elem = sample[self.elem]
        sample['ones'] = np.ones(elem.shape[:2], dtype=np.float32)

        return sample


class CropFromMask(object):
    """
    Returns image cropped in bounding box from a given mask
    """
    def __init__(self, crop_elems=('image', 'gt'),
                 mask_elem='gt',
                 relax=0,
                 zero_pad=False,
                 proportional_bbox=False):

        self.crop_elems = crop_elems
        self.mask_elem = mask_elem
        self.relax = relax
        self.relax = relax
        self.zero_pad = zero_pad
        self.proportional_bbox=proportional_bbox

    def __call__(self, sample):
        _target = sample[self.mask_elem]
        if _target.ndim == 2:
            _target = np.expand_dims(_target, axis=-1)
        for elem in self.crop_elems:
            _img = sample[elem]
            _crop = []
            if self.mask_elem == elem:
                if _img.ndim == 2:
                    _img = np.expand_dims(_img, axis=-1)
                for k in range(0, _target.shape[-1]):
                    _tmp_img = _img[..., k]
                    _tmp_target = _target[..., k]
                    if np.max(_target[..., k]) == 0:
                        _crop.append(np.zeros(_tmp_img.shape, dtype=_img.dtype))
                    else:
                        _crop.append(helpers.crop_from_mask(_tmp_img, _tmp_target, relax=self.relax,
                                                            zero_pad=self.zero_pad,
                                                            proportional_bbox=self.proportional_bbox))
            else:
                for k in range(0, _target.shape[-1]):
                    if np.max(_target[..., k]) == 0:
                        _crop.append(np.zeros(_img.shape, dtype=_img.dtype))
                    else:
                        _tmp_target = _target[..., k]
                        _crop.append(helpers.crop_from_mask(_img, _tmp_target, relax=self.relax,
                                                            zero_pad=self.zero_pad,
                                                            proportional_bbox=self.proportional_bbox))
            if len(_crop) == 1:
                sample['crop_' + elem] = _crop[0]
            else:
                sample['crop_' + elem] = _crop
        return sample

    def __str__(self):
        return 'CropFromMask:(crop_elems='+str(self.crop_elems)+', mask_elem='+str(self.mask_elem)+\
               ', relax='+str(self.relax)+',zero_pad='+str(self.zero_pad)+')'


class CropFromExtremePoints(object):
    def __init__(self, crop_elems=('image', 'gt'),
                 relax=0,
                 zero_pad=False,
                 proportional_bbox=False):
        self.crop_elems = crop_elems
        self.relax = relax
        self.zero_pad = zero_pad
        self.proportional_bbox=proportional_bbox

    def __call__(self, sample):
        for elem in self.crop_elems:
            if 'extreme_points_coord' in elem:
                x_min, y_min, _, _ = helpers.get_bbox(sample['gt'], points=sample[elem],
                                                      pad=self.relax, zero_pad=self.zero_pad,
                                                      proportional_bbox=self.proportional_bbox)
                sample['crop_' + elem] = sample[elem] - np.array([x_min, y_min])
                continue
            bbox = helpers.get_bbox(sample[elem], points=sample['extreme_points_coord'], pad=self.relax,
                                    zero_pad=self.zero_pad, proportional_bbox=self.proportional_bbox)
            if bbox is None:
                crop = np.zeros(sample[elem].shape, dtype=sample[elem].dtype)
            else:
                crop = helpers.crop_from_bbox(sample[elem], bbox, self.zero_pad)
            sample['crop_' + elem] = crop

        return sample


class NormalizeImage(object):
    """
    Return the given elements between 0 and 1
    """
    def __init__(self, norm_elem='image', clip=False):
        self.norm_elem = norm_elem
        self.clip = clip

    def __call__(self, sample):
        if isinstance(self.norm_elem, tuple):
            for elem in self.norm_elem:
                if np.max(sample[elem]) > 1:
                    sample[elem] /= 255.0
        else:
            if self.clip:
                sample[self.norm_elem] = np.clip(sample[self.norm_elem], 0, 255)
            if np.max(sample[self.norm_elem]) > 1:
                sample[self.norm_elem] /= 255.0
        return sample

    def __str__(self):
        return 'NormalizeImage'


class ToImage(object):
    """
    Return the given elements between 0 and 255
    """
    def __init__(self, norm_elem='image', custom_max=255.):
        self.norm_elem = norm_elem
        self.custom_max = custom_max

    def __call__(self, sample):
        if isinstance(self.norm_elem, tuple):
            for elem in self.norm_elem:
                tmp = sample[elem]
                sample[elem] = self.custom_max * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)
        else:
            tmp = sample[self.norm_elem]
            sample[self.norm_elem] = self.custom_max * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)
        return sample

    def __str__(self):
        return 'NormalizeImage'


class BboxFromMask(object):
    """
    Return the coordinates of bbox of the object in the mask
    (0, y_top_left, x_top_left, y_bottom_right, x_bottom_right)
    """
    def __init__(self, mask_elem='gt', relax=0, zero_pad=False, proportional_bbox=False):
        self.mask_elem = mask_elem
        self.relax = relax
        self.zero_pad = zero_pad
        self.proportional_bbox=proportional_bbox

    def __call__(self, sample):
        if sample[self.mask_elem].ndim == 2:
            tmp = np.expand_dims(sample[self.mask_elem], axis=-1)
        else:
            tmp = sample[self.mask_elem]
        res_box = np.ndarray((tmp.shape[-1], 5))
        for k in range(0, tmp.shape[-1]):
            bbox = helpers.get_bbox(tmp[..., k], pad=self.relax, zero_pad=self.zero_pad,
                                    proportional_bbox=self.proportional_bbox)
            if bbox is None:
                res_box[k, :] = np.array((0, 0, 1, 0, 1))
            else:
                res_box[k, :] = np.array((0, bbox[0], bbox[1], bbox[2], bbox[3]))
        sample['bbox'] = np.squeeze(res_box)
        return sample

    def __str__(self):
        return 'BboxFromMask(mask_elem='+str(self.mask_elem)+', relax='+str(self.relax)+', zero_pad='+str(self.zero_pad)+')'


class BboxFromExtremePoints(object):
    def __init__(self, relax=0, zero_pad=False, proportional_bbox=False):
        self.relax = relax
        self.zero_pad = zero_pad
        self.proportional_bbox=proportional_bbox

    def __call__(self, sample):
        bbox = helpers.get_bbox(sample['image'], points=sample['extreme_points_coord'], pad=self.relax,
                                zero_pad=self.zero_pad, proportional_bbox=self.proportional_bbox)
        if bbox is None:
            res_box = np.array((0, 0, 1, 0, 1))
        else:
            res_box = np.array((0, bbox[0], bbox[1], bbox[2], bbox[3]))
        sample['bbox'] = res_box
        return sample


class DuplicateElem(object):
    def __init__(self, dup_elem=['gt'], prefix_name='ori_'):
        self.prefix_name = prefix_name
        self.dup_elem = dup_elem

    def __call__(self, sample):
        for x in self.dup_elem:
            sample[self.prefix_name+x] = sample[x]
        return sample

    def __str__(self):
        return 'DuplicateElem:'+str(self.dup_elem)


class SubtractMeanImage(object):
    def __init__(self, mean, change_channels=False):
        self.mean = mean
        self.change_channels = change_channels

    def __call__(self, sample):
        for elem in sample.keys():
            if 'image' in elem:
                if self.change_channels:
                    sample[elem] = sample[elem][:, :, [2, 1, 0]]
                sample[elem] = np.subtract(sample[elem], np.array(self.mean, dtype=np.float32))
        return sample

    def __str__(self):
        return 'SubtractMeanImage'+str(self.mean)


class DistanceTransform(object):
    """ Compute the distance transform for the given input image (in_key) and
    return it (out_key). If specified, also the maximum pixel location of the
    distance transform can be return (point_key).
    """

    PAD = 1
    OFFSET = np.asarray([1, 1])

    def __init__(self,
                 key=None,
                 out_key=None,
                 point_key=None,
                 invert_coordinates=False):
        if isinstance(key, str):
            key = [key]
        if isinstance(out_key, str):
            out_key = [out_key]
        self.keys = key
        self.out_keys = out_key
        self.point_keys = point_key or {}
        self.invert_coordinates = invert_coordinates

    def __call__(self, sample):
        if self.keys is None:
            return self._transform(sample)
        for in_key, out_key in zip(self.keys, self.out_keys):
            img = sample[in_key]
            img = np.pad(
                img, ((self.PAD, self.PAD), (self.PAD, self.PAD)),
                mode='constant',
                constant_values=0)
            img_t = self._transform(img)
            img_t = img_t[self.PAD:-self.PAD, self.PAD:-self.PAD]
            img_t = np.ascontiguousarray(img_t, dtype=img_t.dtype)
            sample[out_key] = img_t
            point_key = self.point_keys.get(in_key)
            if point_key:
                max_point = np.unravel_index(img_t.argmax(), img_t.shape)
                # max_point -= self.OFFSET
                if self.invert_coordinates:
                    max_point = np.ascontiguousarray(max_point[::-1])
                sample[point_key] = np.asarray(max_point)

        return sample

    def _transform(self, img):
        im = (img > 0).astype(np.uint8)
        img_t = cv2.distanceTransform(im, cv2.DIST_L2, 3)
        return img_t


class DefineInput(object):
    def __init__(self, input_key):
        self.input_key = input_key

    def __call__(self, sample):
        sample['input'] = sample[self.input_key]
        return sample


class Normalize(object):
    """Normalize a pytorch tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, keys_normalize, mean, std):
        if isinstance(keys_normalize, str):
            keys_normalize = [keys_normalize]
        self.keys_normalize = keys_normalize
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        for elem in sample.keys():
            if elem in self.keys_normalize:
                sample[elem] = F_vision.normalize(sample[elem], self.mean, self.std)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RemoveBG(object):
    def __init__(self, mask_key, keys_to_remove_bg, bg_value=0):
        if isinstance(keys_to_remove_bg, str):
            keys_to_remove_bg = [keys_to_remove_bg]
        self.keys_to_remove_bg = keys_to_remove_bg
        self.mask_key = mask_key
        self.bg_value = bg_value

    def __call__(self, sample):
        for elem in sample.keys():
            if elem in self.keys_to_remove_bg:
                tmp = np.array(sample[elem])
                if sample[elem].ndim == 3:
                    assert sample[self.mask_key].shape == sample[elem].shape[:2]
                    tmp[sample[self.mask_key] == 0, :] = self.bg_value
                elif sample[elem].ndim == 2:
                    assert sample[self.mask_key].shape == sample[elem].shape
                    tmp[sample[self.mask_key] == 0] = self.bg_value
                sample[elem] = tmp
        return sample


class RemapMaskValues(object):
    def __init__(self, mask_key, remap_dict):
        self.mask_key = mask_key
        self.remap_dict = remap_dict

    def __call__(self, sample):
        _tmp = np.array(sample[self.mask_key])
        for (k, v) in self.remap_dict.items():
            _tmp[sample[self.mask_key] == k] = v
        sample[self.mask_key] = _tmp.astype(np.float32)
        return sample


class MILtoIdxs(object):
    def __init__(self, d_size=None, sc=None, mildil=False):
        self.d_size = d_size
        self.sc = sc
        self.mildil = mildil
        if self.d_size is not None and self.sc is not None:
            raise ValueError("Choose either d_size or sc")

    def __call__(self, sample, curr_sc=None):

        # Possible to call the object with different d_size in each iteration
        if curr_sc is not None:
            self.sc = curr_sc

        if 'idxh' in sample:
            # ATTENTION: python-like indexing
            pos, idxh_, idxv_ = sample['edge'], sample['idxh'].astype(np.float) - 1, sample['idxv'].astype(np.float) - 1

            if (self.d_size is not None and self.d_size[0] > 0) or self.sc is not None:
                gt_shape = pos.shape

                if isinstance(self.sc, float):
                    # d_size = (int(self.d_size * gt_shape[0]), int(self.d_size * gt_shape[1]))
                    pos = cv2.resize(pos, None, fx=self.sc, fy=self.sc, interpolation=cv2.INTER_NEAREST)
                    d_size = pos.shape
                elif isinstance(self.sc, list):
                    pos = cv2.resize(pos, None, fx=self.sc[1], fy=self.sc[0], interpolation=cv2.INTER_NEAREST)
                    d_size = pos.shape
                else:
                    d_size = self.d_size
                    pos = cv2.resize(pos, dsize=d_size[::-1], interpolation=cv2.INTER_NEAREST)

                if self.sc is not None:
                    el_size = int(2 * (np.round((self.sc[0] ** 2 + self.sc[1] ** 2) ** 0.5)) + 1)
                    # print(self.sc, el_size)
                    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (el_size, el_size))
                else:
                    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

                # Question: double loss for boundary pixels vs thick no-care zone
                if self.mildil:
                    pos = cv2.dilate(pos.astype(np.uint8), kernel=strel).astype(np.float)

                # In the case of no boundaries in the image
                if len(idxh_) == 0:
                    print('No boundaries in this image. Return dummy edgeidx')
                    sample['edge'] = pos
                    sample['edgeidx'] = - np.ones((1, 2))
                    sample['idxh'] = - np.ones((1, 2))
                    sample['idxv'] = - np.ones((1, 2))

                    return sample

                scaling = (float(d_size[0]) / float(gt_shape[0]), float(d_size[1]) / float(gt_shape[1]))
                idxh = np.clip(np.around(idxh_ * scaling[1]), 0, d_size[1] - 1).astype(np.float)
                idxv = np.clip(np.around(idxv_ * scaling[0]), 0, d_size[0] - 1).astype(np.float)
            else:
                idxh = np.clip(idxh_, 0, pos.shape[1] - 1).astype(np.float)
                idxv = np.clip(idxv_, 0, pos.shape[0] - 1).astype(np.float)
                d_size = pos.shape

            if idxh.shape[0] > 0:
                abovz = (idxv_ >= 0) & (idxh_ >= 0) & (idxh >= 0) & (idxv >= 0)
                idxh = idxh * abovz.astype(np.float)
                idxv = idxv * abovz.astype(np.float)

                idxs = idxv * pos.shape[1] + idxh
                idxs = idxs * abovz.astype(np.float)
                assert (idxs.shape[0] > 0)

            else:
                pos = np.zeros((d_size[1], d_size[0]))
                idxs = np.zeros((1, 1))
                # clss = np.zeros((1, 1))

            sample['edge'] = pos
            sample['edgeidx'] = idxs.astype(np.float32)

        return sample


class FixedResizeWithMIL(object):
    """Randomly resize the image and the ground truth to specified scales.
    Args:
        resolutions (list): the list of scales
    """
    def __init__(self, reference='image', resolutions=600, mildil=False):
        self.reference = reference
        self.create_mil = MILtoIdxs(d_size=None, sc=None, mildil=mildil)
        self.resolutions = resolutions

        if isinstance(self.resolutions, int):
            self.resolutions = [self.resolutions]

    def __call__(self, sample):

        # Fixed range of scales
        resolution = self.resolutions[random.randint(0, len(self.resolutions))]
        sc = helpers.conform_to_max(im_shape=sample[self.reference].shape[:2], target_size=resolution,
                                    max_size=1000)
        for elem in sample.keys():
            if 'meta' in elem or 'bbox' in elem or elem == 'idxh' \
                    or elem == 'idxv' or (elem == 'edge' and 'idxh' in sample.keys()):
                continue

            tmp = sample[elem]

            if ((tmp == 0) | (tmp == 1)).all() or 'semseg' in elem or 'human_parts' in elem:
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC

            tmp = cv2.resize(tmp, None, fx=sc[1], fy=sc[0], interpolation=flagval)

            sample[elem] = tmp

        # Apply the same resizing to MIL
        sample = self.create_mil(sample, curr_sc=sc)

        return sample

    def __str__(self):
        return 'RandomResizeWithMIL:'+str(self.resolutions)


class AddIgnoreRegions(object):
    """Add Ignore Regions"""

    def __call__(self, sample):

        for elem in sample.keys():
            tmp = sample[elem]

            if elem == 'normals':
                # Check areas with norm 0
                Nn = np.sqrt(tmp[:, :, 0] ** 2 + tmp[:, :, 1] ** 2 + tmp[:, :, 2] ** 2)

                tmp[Nn == 0, :] = 255.
                sample[elem] = tmp
            elif elem == 'human_parts':
                # Check for images without human part annotations
                if (tmp == 0).all():
                    tmp = 255 * np.ones(tmp.shape, dtype=tmp.dtype)
                    sample[elem] = tmp
            elif elem == 'depth':
                tmp[tmp == 0] = 255.
                sample[elem] = tmp

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem:
                continue
            elif 'bbox' in elem:
                tmp = sample[elem]
                sample[elem] = torch.from_numpy(tmp)
                continue

            tmp = sample[elem]

            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            tmp = tmp.transpose((2, 0, 1))
            sample[elem] = torch.from_numpy(tmp.astype(np.float32))

        return sample

    def __str__(self):
        return 'ToTensor'
