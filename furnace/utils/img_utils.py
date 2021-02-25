import cv2
import math
import numpy as np
import numbers
import random
import collections

def random_crop_w_bbox(img, mask, bbox, bbox_ori):
    valid = np.zeros(mask.shape)
    bbox_ori = [bbox_ori[0], bbox_ori[1], bbox_ori[0]+bbox_ori[2], bbox_ori[1]+bbox_ori[3]] # xywh -> xyxy
    bbox_ori = [int(i) for i in bbox_ori]
    valid[bbox_ori[1]:bbox_ori[3], bbox_ori[0]:bbox_ori[2]] = 1

    bbox = [int(i) for i in bbox]
    img = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    mask = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    valid = valid[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    return img, mask, valid

def random_bbox_jitter(bbox, height, width):
    h, w = bbox[3], bbox[2]
    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
    g = np.random.normal(0, 1, size=(4))
    g = np.clip(g, -2.5, 2.5)
    x0 += 0.05 * g[0] * w
    x1 += 0.05 * g[1] * w
    y0 += 0.05 * g[2] * h
    y1 += 0.05 * g[3] * h

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(x1, width)
    y1 = min(y1, height)

    return [x0, y0, x1, y1]

def random_gamma(img):
    gamma = np.random.uniform(-0.05, 0.05)
    gamma = np.log(0.5 + 1 / math.sqrt(2) * gamma) / np.log(0.5 - 1 / math.sqrt(2) * gamma)
    table = (np.arange(0, 256.0 / 255, 1.0 / 255) ** gamma) * 255
    img = cv2.LUT(img, table.astype(np.uint8))
    return img

def generate_bbox_by_mask(mask):
    if mask.sum() == 0:
        return None
    xv, yv = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))
    x_bottom_left = xv[mask > 0].min()
    x_top_right = xv[mask > 0].max()
    y_bottom_left = yv[mask > 0].min()
    y_top_right = yv[mask > 0].max()
    bbox = [float(x_bottom_left), float(y_bottom_left), float(x_top_right), float(y_top_right)]
    return bbox

def generate_random_common_bbox(mask0, mask1):
    bbox0 = generate_bbox_by_mask(mask0)
    bbox1 = generate_bbox_by_mask(mask1)
    if bbox1 is None:
        bbox1 = bbox0
    h,w = mask0.shape

    xmin = min(bbox0[0], bbox1[0])
    ymin = min(bbox0[1], bbox1[1])
    xmax = max(bbox0[2], bbox1[2])
    ymax = max(bbox0[3], bbox1[3])

    xmin -= 0.05 * (xmax - xmin)
    ymin -= 0.05 * (ymax - ymin)
    xmax += 0.05 * (xmax - xmin)
    ymax += 0.05 * (ymax - ymin)

    xmin = int(max(0, xmin))
    ymin = int(max(0, ymin))
    xmax = int(min(w, xmax))
    ymax = int(min(h, ymax))

    bbox = [random.randint(0, xmin), random.randint(0, ymin), random.randint(xmax, w), random.randint(ymax, h)]
    return bbox

def random_scale_crop(img0, img1, mask0, mask1):

    scale = random.uniform(0.7, 1.3)
    h,w,_ = img0.shape
    h_ = int(h*scale)
    w_ = int(w*scale)

    img0 = cv2.resize(img0, (w_,h_))
    img1 = cv2.resize(img1, (w_,h_))
    mask0 = cv2.resize(mask0, (w_,h_), interpolation=cv2.INTER_NEAREST)
    mask1 = cv2.resize(mask1, (w_,h_), interpolation=cv2.INTER_NEAREST)

    H = int(0.9 * h_)
    W = int(0.9 * w_)
    H_offset = random.choice(range(h_ - H))
    W_offset = random.choice(range(w_ - W))
    H_slice = slice(H_offset, H_offset + H)
    W_slice = slice(W_offset, W_offset + W)

    img0 = img0[H_slice, W_slice, :]
    img1 = img1[H_slice, W_slice, :]
    mask0 = mask0[H_slice, W_slice]
    mask1 = mask1[H_slice, W_slice]

    return img0, img1, mask0, mask1

def random_hflip(img, mask, valid):
    if random.random() >= 0.5:
        img = img[:,::-1,:]
        mask = mask[:,::-1]
        valid = valid[:,::-1]
    return img, mask, valid

def random_hflip_adnet(img0, img1, mask):
    if random.random() >= 0.5:
        img0 = img0[:,::-1,:]
        img1 = img1[:,::-1,:]
        mask = mask[:,::-1]
    return img0, img1, mask

def random_vflip(img, mask, valid):
    if random.random() >= 0.5:
        img = img[::-1,:,:]
        mask = mask[::-1,:]
        valid = valid[::-1,:]
    return img, mask, valid

def get_2dshape(shape, *, zero=True):
    if not isinstance(shape, collections.Iterable):
        shape = int(shape)
        shape = (shape, shape)
    else:
        h, w = map(int, shape)
        shape = (h, w)
    if zero:
        minv = 0
    else:
        minv = 1

    assert min(shape) >= minv, 'invalid shape: {}'.format(shape)
    return shape

'''
def random_hflip(img0, img1, gt0, gt1):
    if random.random() >= 0.5:
        img0 = cv2.flip(img0, 1)
        img1 = cv2.flip(img1, 1)
        gt0 = cv2.flip(gt0, 1)
        gt1 = cv2.flip(gt1, 1)
    return img0, img1, gt0, gt1

def random_vflip(img0, img1, gt0, gt1):
    if random.random() >= 0.5:
        img0 = cv2.flip(img0, 0)
        img1 = cv2.flip(img1, 0)
        gt0 = cv2.flip(gt0, 0)
        gt1 = cv2.flip(gt1, 0)
    return img0, img1, gt0, gt1
'''

def random_crop_pad_to_shape(img, crop_pos, crop_size, pad_label_value):
    h, w = img.shape[:2]
    start_crop_h, start_crop_w = crop_pos
    assert ((start_crop_h < h) and (start_crop_h >= 0))
    assert ((start_crop_w < w) and (start_crop_w >= 0))

    crop_size = get_2dshape(crop_size)
    crop_h, crop_w = crop_size

    img_crop = img[start_crop_h:start_crop_h + crop_h,
               start_crop_w:start_crop_w + crop_w, ...]

    img_, margin = pad_image_to_shape(img_crop, crop_size, cv2.BORDER_CONSTANT,
                                      pad_label_value)

    return img_, margin

def random_crop_pad_to_shape_flow(flow, crop_pos, crop_size, pad_label_value):
    h, w = flow.shape[:2]
    start_crop_h, start_crop_w = crop_pos
    assert ((start_crop_h < h) and (start_crop_h >= 0))
    assert ((start_crop_w < w) and (start_crop_w >= 0))

    crop_size = get_2dshape(crop_size)
    crop_h, crop_w = crop_size

    flow_crop = flow[start_crop_h:start_crop_h + crop_h,
               start_crop_w:start_crop_w + crop_w, ...]

    flow_, margin = pad_image_to_shape(flow_crop, crop_size, cv2.BORDER_CONSTANT,
                                      pad_label_value)
    flow_[:,:,0] = flow_[:,:,0]*w/crop_w
    flow_[:,:,1] = flow_[:,:,1]*h/crop_h

    return flow_, margin


def generate_random_crop_pos(ori_size, crop_size):
    ori_size = get_2dshape(ori_size)
    h, w = ori_size

    crop_size = get_2dshape(crop_size)
    crop_h, crop_w = crop_size

    pos_h, pos_w = 0, 0

    if h > crop_h:
        pos_h = random.randint(0, h - crop_h + 1)

    if w > crop_w:
        pos_w = random.randint(0, w - crop_w + 1)

    return pos_h, pos_w

def get_mask_center(mask):
    h,w = mask.shape
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)
    hor, ver = np.meshgrid(x, y)
    hor *= mask
    ver *= mask
    center_x = int(hor.sum() / (mask.sum() + 1e-6))
    center_y = int(ver.sum() / (mask.sum() + 1e-6))

    min_x = int(hor.min())
    min_y = int(ver.min())
    max_x = int(hor.max())
    max_y = int(ver.max())

    return (center_x, center_y), (min_x, min_y), (max_x, max_y)

def generate_random_crop_pos_center(ori_size, crop_size, mask):
    ori_size = get_2dshape(ori_size)
    h, w = ori_size

    crop_size = get_2dshape(crop_size)
    crop_h, crop_w = crop_size

    pos_h, pos_w = 0, 0
    center, tl, br = get_mask_center(mask)

    maskw = br[0] - tl[0]
    maskh = br[1] - tl[1]

    maskcenter_x = random.randint(center[0] - maskw // 4, center[0] + maskw // 4)
    maskcenter_y = random.randint(center[1] - maskh // 4, center[1] + maskh // 4)

    pos_w = max(0, maskcenter_x - maskw // 2)
    pos_h = max(0, maskcenter_y - maskh // 2)

    #if br[0] - crop_w // 2 > 0:
    #    pos_w = random.randint(max(0, tl[0] - (br[0] - tl[0]) // 2 + 1), tl[0] + (br[0] - tl[0]) // 2 + 1)

    #if br[1] - crop_h // 2 > 0:
    #    pos_h = random.randint(max(0, tl[1] - (br[1] - tl[1]) // 2 + 1), tl[1] + (br[1] - tl[1]) // 2 + 1)

    return pos_h, pos_w


def pad_image_to_shape(img, shape, border_mode, value):
    margin = np.zeros(4, np.uint32)
    shape = get_2dshape(shape)
    pad_height = shape[0] - img.shape[0] if shape[0] - img.shape[0] > 0 else 0
    pad_width = shape[1] - img.shape[1] if shape[1] - img.shape[1] > 0 else 0

    margin[0] = pad_height // 2
    margin[1] = pad_height // 2 + pad_height % 2
    margin[2] = pad_width // 2
    margin[3] = pad_width // 2 + pad_width % 2

    img = cv2.copyMakeBorder(img, margin[0], margin[1], margin[2], margin[3],
                             border_mode, value=value)

    return img, margin


def pad_image_size_to_multiples_of(img, multiple, pad_value):
    h, w = img.shape[:2]
    d = multiple

    def canonicalize(s):
        v = s // d
        return (v + (v * d != s)) * d

    th, tw = map(canonicalize, (h, w))

    return pad_image_to_shape(img, (th, tw), cv2.BORDER_CONSTANT, pad_value)


def resize_ensure_shortest_edge(img, edge_length,
                                interpolation_mode=cv2.INTER_LINEAR):
    assert isinstance(edge_length, int) and edge_length > 0, edge_length
    h, w = img.shape[:2]
    if h < w:
        ratio = float(edge_length) / h
        th, tw = edge_length, max(1, int(ratio * w))
    else:
        ratio = float(edge_length) / w
        th, tw = max(1, int(ratio * h)), edge_length
    img = cv2.resize(img, (tw, th), interpolation_mode)

    return img


def random_scale(img, gt, scales):
    scale = random.choice(scales)
    sh = int(img.shape[0] * scale)
    sw = int(img.shape[1] * scale)
    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)

    return img, gt, scale

def random_scale_pair(img0, img1, gt0, gt1, flow0, flow1, mask, low, high):
    scale = np.random.uniform(low, high)
    sh = int(img0.shape[0] * scale)
    sw = int(img0.shape[1] * scale)
    img0 = cv2.resize(img0, (sw, sh), interpolation=cv2.INTER_LINEAR)
    img1 = cv2.resize(img1, (sw, sh), interpolation=cv2.INTER_LINEAR)
    flow0 = cv2.resize(flow0, (sw, sh), interpolation=cv2.INTER_LINEAR)
    flow1 = cv2.resize(flow1, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt0 = cv2.resize(gt0, (sw, sh), interpolation=cv2.INTER_NEAREST)
    gt1 = cv2.resize(gt1, (sw, sh), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)

    return img0, img1, gt0, gt1, flow0, flow1, mask, scale


def random_scale_new(img, pred, gt, fgprob, scales):
    scale = random.choice(scales)
    sh = int(img.shape[0] * scale)
    sw = int(img.shape[1] * scale)
    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    pred = cv2.resize(pred, (sw, sh), interpolation=cv2.INTER_NEAREST)
    fgprob = cv2.resize(fgprob, (sw, sh), interpolation=cv2.INTER_LINEAR)
    return img, pred, gt, fgprob, scale


def random_scale_with_length(img, gt, length):
    size = random.choice(length)
    sh = size
    sw = size
    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)

    return img, gt, size


def random_mirror(img, gt):
    if random.random() >= 0.5:
        img = cv2.flip(img, 1)
        gt = cv2.flip(gt, 1)

    return img, gt,

def random_mirror_new(img, pred, gt, fgprob):
    if random.random() >= 0.5:
        img = cv2.flip(img, 1)
        gt = cv2.flip(gt, 1)
        pred = cv2.flip(pred, 1)
        fgprob = cv2.flip(fgprob, 1)

    return img, pred, gt, fgprob


def random_rotation(img, gt):
    angle = random.random() * 20 - 10
    h, w = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    gt = cv2.warpAffine(gt, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)

    return img, gt

def random_rotation_pair(img0, img1, gt0, gt1):
    angle = random.random() * 80 - 40
    h, w = img0.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    mask = np.ones((h,w))
    img0 = cv2.warpAffine(img0, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    img1 = cv2.warpAffine(img1, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    gt0 = cv2.warpAffine(gt0, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
    gt1 = cv2.warpAffine(gt1, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
    mask = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
    gt0[mask == 0] = 255
    gt1[mask == 0] = 255

    return img0, img1, gt0, gt1, mask

def random_rotation_adnet(img0, img1, gt0):
    if random.random() >= 0.51:
        angle = random.random() * 90 - 45
        h, w = img0.shape[:2]
        #mask = np.ones((h,w))
        rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        img0 = cv2.warpAffine(img0, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
        img1 = cv2.warpAffine(img1, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
        gt0 = cv2.warpAffine(gt0, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
        #mask = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
        #gt0[mask == 0] = 255

    return img0, img1, gt0


def random_gaussian_blur(img):
    gauss_size = random.choice([1, 3, 5, 7])
    if gauss_size > 1:
        # do the gaussian blur
        img = cv2.GaussianBlur(img, (gauss_size, gauss_size), 0)

    return img


def center_crop(img, shape):
    h, w = shape[0], shape[1]
    y = (img.shape[0] - h) // 2
    x = (img.shape[1] - w) // 2
    return img[y:y + h, x:x + w]


def random_crop(img, gt, size):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        size = size

    h, w = img.shape[:2]
    crop_h, crop_w = size[0], size[1]

    if h > crop_h:
        x = random.randint(0, h - crop_h + 1)
        img = img[x:x + crop_h, :, :]
        gt = gt[x:x + crop_h, :]

    if w > crop_w:
        x = random.randint(0, w - crop_w + 1)
        img = img[:, x:x + crop_w, :]
        gt = gt[:, x:x + crop_w]

    return img, gt


def normalize(img, mean, std):
    # pytorch pretrained model need the input range: 0-1
    img = img.astype(np.float32) / 255.0
    img = img - mean
    img = img / std

    return img


def findContours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions 3 and 4
    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith('4'):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith('3'):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            'cv2 must be either version 3 or 4 to call this method')

    return contours, hierarchy
