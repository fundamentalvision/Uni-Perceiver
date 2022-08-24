import itertools
import numpy as np
import torch
# from torch.nn.utils.rnn import pad_sequence

def pad_sequence(sequences, batch_first=False, padding_value=0.0, padding_length=None):
    """
    modified from  torch.nn.utils.rnn.pad_sequence

    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences]) if padding_length is None else padding_length
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


def pad_tensor(tensor, padding_value, use_mask, padding_length=None):
    if isinstance(tensor[0], list):
        tensor = list(itertools.chain.from_iterable(tensor))

    out = pad_sequence(tensor, batch_first=True, padding_value=padding_value, padding_length=padding_length)
    if use_mask:
        lengths = [t.size(0) for t in tensor]
        max_lengths = max(lengths) if padding_length is None else padding_length
        mask = torch.zeros((out.size(0), max_lengths), dtype=torch.uint8)
        for i, length in enumerate(lengths):
            mask[i, 0:length] = 1
        return out, mask
    else:
        return out

def dict_to_cuda(input_dict):
    for key in input_dict:
        if isinstance(input_dict[key], torch.Tensor):
            input_dict[key] = input_dict[key].cuda(non_blocking=True)
        elif isinstance(input_dict[key], dict):
            input_dict[key] = dict_to_cuda(input_dict[key])



def dict_as_tensor(input_data):
    if isinstance(input_data, str) or isinstance(input_data, tuple) or  isinstance(input_data, int) or input_data is None:
        pass 
    elif isinstance(input_data, dict):
        for key in input_data:
            input_data[key] = dict_as_tensor(input_data[key])
    elif isinstance(input_data, list):
        input_data = [dict_as_tensor(item) for item in input_data]
    else:
        input_data = torch.as_tensor(input_data)
    return input_data 


def boxes_to_locfeats(boxes, image_w, image_h):
    image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
    image_location[:, :4] = boxes
    image_location[:, 4] = (
        (image_location[:, 3] - image_location[:, 1])
        * (image_location[:, 2] - image_location[:, 0])
        / (float(image_w) * float(image_h))
    )

    image_location[:, 0] = image_location[:, 0] / float(image_w)
    image_location[:, 1] = image_location[:, 1] / float(image_h)
    image_location[:, 2] = image_location[:, 2] / float(image_w)
    image_location[:, 3] = image_location[:, 3] / float(image_h)
    return image_location

def expand_tensor(tensor, size, dim=1):
    if size == 1 or tensor is None:
        return tensor
    tensor = tensor.unsqueeze(dim)
    if dim == 0:
        tensor = tensor.expand([size] + [-1] + list(tensor.shape[2:]))
        tensor = tensor.reshape([-1] + list(tensor.shape[2:]))
    else:
        tensor = tensor.expand(list(tensor.shape[:dim]) + [size] + list(tensor.shape[dim+1:]))
        tensor = tensor.reshape(list(tensor.shape[:dim-1]) + [-1] + list(tensor.shape[dim+1:]))
    return tensor

def iou(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.shape[0]
    K = gt_boxes.shape[0]

    gt_boxes_area = (
        (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
    ).reshape(1, K)

    anchors_area = (
        (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)
    ).reshape(N, 1)

    boxes = np.repeat(anchors.reshape(N, 1, 4), K, axis=1)
    query_boxes = np.repeat(gt_boxes.reshape(1, K, 4), N, axis=0)

    iw = (
        np.minimum(boxes[:, :, 2], query_boxes[:, :, 2])
        - np.maximum(boxes[:, :, 0], query_boxes[:, :, 0])
        + 1
    )
    iw[iw < 0] = 0

    ih = (
        np.minimum(boxes[:, :, 3], query_boxes[:, :, 3])
        - np.maximum(boxes[:, :, 1], query_boxes[:, :, 1])
        + 1
    )
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps


def get_max_len_from_mask(mask):
    return int(mask.sum(1).max().item())


def clip_v_inputs(v_feats, spatials, image_mask):
    max_len = get_max_len_from_mask(image_mask)
    v_feats = v_feats[:, :max_len]
    spatials = spatials[:, :max_len]
    image_mask = image_mask[:, :max_len]
    return v_feats, spatials, image_mask


def clip_t_inputs(input_txt, segment_ids, input_mask):
    max_len = get_max_len_from_mask(input_mask)
    input_txt = input_txt[:, :max_len]
    segment_ids = segment_ids[:, :max_len]
    input_mask = input_mask[:, :max_len]
    return input_txt, segment_ids, input_mask