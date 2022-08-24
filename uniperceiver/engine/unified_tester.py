from torch.functional import Tensor
import tqdm
import os
import pickle
import sys
import numpy as np
import itertools
import random
import torch
from torch.cuda.amp import autocast
import shutil
import uniperceiver.utils.comm as comm
from timm.utils import accuracy
from collections import defaultdict, deque
import torch.distributed as dist


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.

    # borrowed from diet and mae
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not comm.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value)


def tester(task_cfg, model, test_data_loader, evaluator, epoch, amp_fp16, apex_fp16):
    results = dict()
    for task in test_data_loader.keys():
        comm._LOCAL_CURRENT_TASK = task  # used for other script
        if test_data_loader[task] is None:
            continue
        if comm.is_main_process():
            print('val/test task {}'.format(task))
        if 'to_task' in dir(model):
            model.to_task(task)
        else:
            model.module.to_task(task)
        task_type = task_cfg[task]['DATASETS']['TASK_TYPE']
        if task_type in ["image_retrieval", 'video_retrieval']:
            results[task] = test_retrieval(task_cfg[task], model, test_data_loader[task], evaluator[task], epoch, amp_fp16, task)
        else:
            results[task] = test_cls(task_cfg[task], model, test_data_loader[task], evaluator[task], epoch, amp_fp16, task)

        if 'reset_attr' in dir(model):
            model.reset_attr()
        else:
            model.module.reset_attr()
    return results


# TODO write eval func for each task_type
def test_cls(cfg, model, test_data_loader, evaluator, epoch, amp_fp16, task=None):
    # only one works
    # if not comm.is_main_process():
    #     return None
    model.eval()
    results = []

    if not os.path.exists(comm.temp_dir):
        os.mkdir(comm.temp_dir)

    # shared_seed = comm.shared_random_seed() this simply does not work!
    shared_seed = random.randint(0, sys.maxsize)
    shared_seed = torch.tensor(shared_seed, device=next(model.parameters()).device)
    torch.distributed.broadcast(shared_seed, src=0)
    shared_seed = shared_seed.item()
    if comm.is_main_process():
        os.makedirs(os.path.join(comm.temp_dir, str(shared_seed)))
    comm.synchronize()

    # remove the cached  embedding for word vocab
    if isinstance(getattr(comm.unwrap_model(model), 'beam_searcher', None), torch.nn.Module):
        if hasattr(getattr(comm.unwrap_model(model), 'beam_searcher', None), 'pre_computed_word_embeds'):
            del comm.unwrap_model(model).beam_searcher.pre_computed_word_embeds
            comm.unwrap_model(model).beam_searcher.pre_computed_word_embeds = None

    meters = defaultdict(SmoothedValue)
    with torch.no_grad():

        for i, data in tqdm.tqdm(enumerate(test_data_loader)) if comm.is_main_process() else enumerate(test_data_loader):
            # data = comm.unwrap_model(model).preprocess_batch(data)
            # if i > 10:
            #     break
            #     model.train()
            #     return {}
            if task is not None:
                data["task_info"]['task_name'] = task
            data = move_to_cuda(data)
            task_type = data['task_info']['task_type']

            sample_infos = data['input_sample_list'][0].get('sample_info', None)
            with autocast(amp_fp16):
                if cfg.INFERENCE.GENERATION_MODE:
                    res = model(data, use_beam_search=True, output_sents=True)
                else:
                    res = model(data)

            if isinstance(res["output"], torch.Tensor) and res["output"].dtype != torch.float32:
                res["output"] = res["output"].float()

            outputs = res["output"]

            if task_type == 'vqa':
                u_logits = res["output"]
                outputs = torch.softmax(u_logits, dim=-1)
                outputs = torch.max(outputs, 1)[1].data

                if isinstance(data['input_sample_list'][0]['sample_info'], dict):
                    # single gpu; changes for data['input_sample_list'][0]['sample_info']
                    sample_infos = data['input_sample_list'][0]['sample_info']['sample_info_per_sample'][1]
                elif  isinstance(data['input_sample_list'][0]['sample_info'], list):
                    # multi gpu;  original data
                    sample_infos = data['input_sample_list'][1]['sample_info']

                for sample_info_pers_ample, output in zip(sample_infos, outputs):
                    if isinstance(output, torch.Tensor):
                        output = output.cpu()
                    # results.append({ "task_name": task, "answer": output, "question_id": int(sample_info_pers_ample['question_id'])})
                    results.append({ "answer": output, "question_id": int(sample_info_pers_ample['question_id'])})

            elif task_type in ['image_classification']:
                # targets in the input data
                targets = data['target_idx_list'][0]
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                bs = targets.shape[0]
                meters['acc1'].update(acc1.item(), n=bs)
                meters['acc5'].update(acc5.item(), n=bs)

                pass
                "an early version for evaluating Imagenet-1K "
                """
                # rely on ids to retrive label
                outputs = outputs.cpu()
                if isinstance(data['input_sample_list'][0]['sample_info'], dict):
                    # single gpu; changes for data['input_sample_list'][0]['sample_info']
                    sample_infos = data['input_sample_list'][0]['sample_info']['sample_info_per_sample'][0]
                elif  isinstance(data['input_sample_list'][0]['sample_info'], list):
                    # multi gpu;  original data
                    sample_infos = data['input_sample_list'][0]['sample_info']
                else:
                    raise NotImplementedError('please check')

                for idx, si in enumerate(sample_infos):
                    results.append({cfg.INFERENCE.ID_KEY: si['id'], cfg.INFERENCE.VALUE: outputs[idx]})
                """
            elif task_type in ['image_caption', 'video_caption']:
                ids = res["IDS"]
                for id, output in zip(ids, outputs):
                    results.append({"image_id": int(id.item()), "caption": output})
            elif task_type in ['text_classification']:
                for label, output in zip(data['target_idx_list'][0], outputs):
                    results.append({"label": int(label), "pred": output})

            elif task_type in ['video_classification']:
                # targets in the input data
                targets = data['target_idx_list'][0]
                outputs = torch.softmax(outputs, -1).view(-1, sample_infos[0]['num_views'], outputs.size(-1)).mean(1)
                acc1 = accuracy(outputs, targets, topk=(1,))[0]
                bs = targets.shape[0]
                meters['acc1'].update(acc1.item(), n=bs)

            else:
                raise NotImplementedError


    if task_type in ['image_classification']:
        for meter in meters.values():
            meter.synchronize_between_processes()
        eval_res = {'Acc@1': meters['acc1'].global_avg, 'Acc@5': meters['acc5'].global_avg}
    elif task_type in ['video_classification']:
        for meter in meters.values():
            meter.synchronize_between_processes()
        eval_res = {'Acc@1': meters['acc1'].global_avg}
    else:
        with open(os.path.join(comm.temp_dir, str(shared_seed), "rank_{}.pkl".format(comm.get_rank())), 'wb') as f:
            # json.dump(results, f)
            pickle.dump(results, f)
        comm.synchronize()
        if comm.is_main_process():
            results_all = list()
            for i in range(comm.get_world_size()):
                with open(os.path.join(comm.temp_dir, str(shared_seed), "rank_{}.pkl".format(i)), 'rb') as f:
                    # results_all += json.load(f)
                    results_all += pickle.load(f)

            results = results_all

            if evaluator is not None:
                eval_res = evaluator.eval(results, epoch)
            else:
                eval_res = ''

            # remove cached files
            shutil.rmtree(os.path.join(comm.temp_dir, str(shared_seed)))

    model.train()
    comm.synchronize()
    if comm.is_main_process():
        return eval_res
    else:
        return None


def test_retrieval(cfg, model, test_data_loader, evaluator, epoch, amp_fp16, task=None):

    if evaluator is not None:
        if not comm.is_main_process():
            comm.synchronize()
            return None
        ret = {}
        model.eval()
        ids = []
        vfeats = []
        tfeats = []
        with torch.no_grad():
            for data in tqdm.tqdm(test_data_loader):
                if task is not None:
                    data["task_info"]['task_name'] = task
                data = move_to_cuda(data)
                # task_type = data['task_info']['task_type']

                ids_local = [si['id'] for si in data['input_sample_list'][0]['sample_info']]
                with autocast(amp_fp16):
                    outputs = model(data)
                ids += ids_local
                vfeats.append(outputs["input_feats"])
                tfeats.append(outputs["tgt_feats"])

        iids = [i[0] for i in ids]
        cids = [i[1] for i in ids]
        cids = list(itertools.chain.from_iterable(cids))
        labels = np.expand_dims(cids, axis=1) == np.expand_dims(iids, axis=0)
        labels = labels.astype(int)
        vfeats = torch.cat(vfeats, dim=0)
        tfeats = torch.cat(tfeats, dim=0)

        ret.update(evaluator.eval(vfeats, tfeats, labels, 't2i'))
        ret.update(evaluator.eval(tfeats, vfeats, labels.T, 'i2t'))
        model.train()
        comm.synchronize()
        return ret

    else:
        raise NotImplementedError('please use \'RetrievalEvaler\'.')


def move_to_cuda(data):
    if isinstance(data, dict):
        for key in data:
            data[key] = move_to_cuda(data[key])
        return data
    elif isinstance(data, list):
        return [move_to_cuda(item) for item in data]
    elif isinstance(data, torch.Tensor):
        return data.cuda(non_blocking=True)
    else:
        # let alone variable with other type
        return data


def dict_to_cuda(input_dict):
    for key in input_dict:
        if isinstance(input_dict[key], torch.Tensor):
            input_dict[key] = input_dict[key].cuda(non_blocking=True)
        elif isinstance(input_dict[key], dict):
            input_dict[key] = dict_to_cuda(input_dict[key])
    return input_dict


def list_to_cuda(input_list):
    # e.g., shared_targets
    return [dict_to_cuda(item) if isinstance(item, dict) else item for item in input_list]


def data_to_cuda(data):
    data = dict_to_cuda(data)
    data['net_input']['shared_targets'] = list_to_cuda(data['net_input']['shared_targets'])


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
