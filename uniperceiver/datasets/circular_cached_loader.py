import queue
import random
from threading import Thread
import time

import pyarrow as pa
import torch.multiprocessing as multiprocessing

import torch
from copy import deepcopy

string_classes = (str, bytes)
import collections.abc as container_abcs
import re

def pin_memory(data):
    if isinstance(data, torch.Tensor):
        return data.pin_memory()
    elif isinstance(data, string_classes):
        return data
    elif isinstance(data, container_abcs.Mapping):
        return {k: pin_memory(sample) for k, sample in data.items()}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return type(data)(*(pin_memory(sample) for sample in data))
    elif isinstance(data, container_abcs.Sequence):
        return [pin_memory(sample) for sample in data]
    elif hasattr(data, "pin_memory"):
        return data.pin_memory()
    else:
        return data


np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


class CircularCachedInputIterator(object):
    """
    chunk: a serialized List[Dict] in the apache arrow format,
        could be sequentially loaded into memory with minimum deserialization cost(<1ms)
    shard: a part of dataset which is allocated to a specific rank(process) in the world,
        generally contains multiple chunks

    main thread:
        - populate chunk_index_queue
        - swap new chunk and old chunk
    prefetch threads:
        - fetch chunk_index_queue
        - populate loaded_chunk_queue

    chunk_index_queue: main -> prefetch, used for shuffling chunk order per epoch
    loaded_chunk_queue: preftch -> main, a limited-size channel for prefetching worker to send back result
    """
    def __init__(self,
                 input_map,
                 batch_size,
                 chunk_path_list,
                 num_data_point,
                 num_shards,
                 shard_id,
                 random_shuffle,
                 num_prefetch_chunk=4,
                 num_worker=4):
        self.input_map = input_map
        self.batch_size = batch_size
        self.num_shareds = num_shards
        self.shard_id = shard_id
        self.random_shuffle = random_shuffle
        self.num_data_point = num_data_point
        self.chunk_filename_list = chunk_path_list
        self.chunk = None
        self.next_chunk_queue = queue.Queue(num_prefetch_chunk)
        self.index_queue = queue.Queue()
        self.chunk_index_queue = queue.Queue()
        self.num_chunk_in_shard = None
        self.chunk_indexes = None
        self.worker = None
        self.num_worker = num_worker
        self.setup_shard()
        self.warmup_cache()

    def setup_shard(self):
        # ensure each shard has the same of of chunks per epoch
        # this might not be necessary
        self.num_chunk_in_shard = len(self.chunk_filename_list) // self.num_shareds
        # [start, end)
        shard_start = self.num_chunk_in_shard * self.shard_id
        shard_end = len(self.chunk_filename_list) if self.shard_id == self.num_shareds - 1 else self.num_chunk_in_shard * (self.shard_id + 1)
        self.chunk_indexes = list(range(shard_start, shard_end))

    def _chunk_prefetch_worker(self):
        while True:
            chunk_index = self.get_chunk_index()
            chunk_filename = self.chunk_filename_list[chunk_index]
            with open(chunk_filename, "rb") as fin:
                chunk = pa.deserialize_from(fin, None)
            self.next_chunk_queue.put(chunk)

    def warmup_cache(self):
        self.worker = [Thread(target=self._chunk_prefetch_worker, args=[]) for _ in range(self.num_worker)]
        for worker in self.worker:
            worker.daemon = True
            worker.start()

    def get_chunk_index(self):
        if self.chunk_index_queue.empty():
            if self.random_shuffle:
                random.shuffle(self.chunk_indexes)
            for ind in self.chunk_indexes[:self.num_chunk_in_shard]:
                self.chunk_index_queue.put(ind)
        return self.chunk_index_queue.get()

    def get_index(self):
        if self.index_queue.empty():
            if self.chunk is not None:
                del self.chunk  # release memory
            self.chunk = self.next_chunk_queue.get()
            self.indexes = list(range(len(self.chunk)))
            if self.random_shuffle:
                random.shuffle(self.indexes)
            # keep all shards of the same size
            for ind in self.indexes:
                self.index_queue.put(ind)
        return self.index_queue.get()

    def epoch_size(self):
        return self.num_data_point // self.num_shareds

    def __iter__(self):
        return self

    def __next__(self):
        datas = tuple([] for _ in self.input_map)
        for _ in range(self.batch_size):
            ind = self.get_index()
            data = self.chunk[ind]
            # value = data['jpeg']
            # label = data['label']
            # # DO NOT reuse the buffer
            # jpegs.append(value)
            # labels.append(np.array([label], dtype=np.int32))
            # datas.append(data)
            for i, k in enumerate(self.input_map):
                datas[i].append(deepcopy(data[k]))
        return datas

    next = __next__

