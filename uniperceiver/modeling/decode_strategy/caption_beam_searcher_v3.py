# Copyright (c) 2019, AImageLab
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from uniperceiver.config import configurable
from uniperceiver.functional import expand_tensor
from .decode_strategy import DecodeStrategy
from .build import DECODE_STRATEGY_REGISTRY
from uniperceiver.utils import comm
import math
from torch.cuda.amp import autocast

@DECODE_STRATEGY_REGISTRY.register()
class CaptionBeamSearcherV3(DecodeStrategy):

    def data_half(self, data):
        if self.fp16:
            for k, v in data.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                    data[k] = v.half()
                    # print(k)
            return data
        else:
            return data




    def _select(self, batch_size, beam_size, t, candidate_logprob):
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(batch_size, -1), -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :beam_size], selected_idx[:, :beam_size]
        return selected_idx, selected_logprob

    def _expand_state(self, states, selected_beam, batch_size, beam_size, cur_beam_size):
        for i in range(len(states)):
            shape = list(states[i].shape)
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            states[i] = torch.gather(states[i].view(*([batch_size, cur_beam_size] + shape[1:])), 1,
                beam.expand(*([batch_size, beam_size] + shape[1:])))
            states[i] = states[i].view(*([-1, ] + shape[1:]))


    def _forward(self, batched_inputs, model):
        # only two  caption tasks are generative task now!
        # for caption tasks, the computations are:
        # 1. encode the image sequence; save for further use.
        # 2. if no cached encoded dictionary, encode the dictionary and save; otherwise reuse cache.
        # 3. compute attention. We use cross attention insted of self attention.

        # batched_inputs[kfg.IMAGE] = batched_inputs.pop(kfg.VIDEO).squeeze(1)

        inputs = batched_inputs
        inputs = self.data_half(inputs)


        out_size = batched_inputs.get('OUT_SIZE', 1)

        task_info = inputs['task_info']
        bs = task_info['batch_size']
        if isinstance(bs, torch.Tensor):
            bs = bs.item()

        image_input = inputs['input_sample_list']
        vocab_input = inputs['shared_target_sets'][self.vocab_name]


        # 1. encode the image/video sequence.
        moe_embedding = None
        for image_data in image_input:
            if 'moe_embedding' in image_data:
                moe_embedding = image_data['moe_embedding']
        image_encode = model._forward_data(image_input, task_info=task_info, return_all=True)[0]['data']


        # 2. encode the vocabulary - if no pre-computed, add that into input
        if getattr(self, 'pre_computed_word_embeds', None) is None:
            vocab_encode = model._forward_data(vocab_input, task_info=task_info, return_all=False)[0]
            self.pre_computed_word_embeds = vocab_encode
        else:
            vocab_encode = self.pre_computed_word_embeds

        # 3. compute attention

        comm._CAPTION_GEN_MODE = True
        task_info.update({"prefix_spe_before_fuse": False})

        beam_size = self.beam_size
        log_probs = []
        selected_words = None
        seq_logprob = torch.zeros((bs, 1, 1)).cuda() # bs, 1, 1
        seq_mask = torch.ones((bs, beam_size, 1)).cuda()
        wt = Variable(torch.zeros(bs, dtype=torch.long).cuda().unsqueeze(1)) + self.spe_token_id
        u_tokens_type = wt.new_zeros(wt.shape) # [Note] we assume the type tokens are 0.

        history_states = image_encode[:-1]
        len_prefix = history_states[0].shape[1]
        total_history_states = [ history_states[0].new_zeros(beam_size * bs, image_encode[0].shape[1] + self.max_seq_len, image_encode[0].shape[2]) for _ in history_states]
        for i, ths in enumerate(total_history_states):
            hs = history_states[i]
            ths[:hs.shape[0], :hs.shape[1], :] = hs

        outputs = []
        common_info =  {
            "modality": "text",
            'data_type': 'input',
            'moe_embedding': moe_embedding,

        }
        for t in range(self.max_seq_len):
            cur_beam_size = 1 if t == 0 else beam_size

            history_states = [ ths[ :(cur_beam_size*bs), :(len_prefix+t), :] for ths in total_history_states]

            step_data = {   "data": wt,
                            "time_step": t,
                            "sample_info":
                            {
                                "data_cum_length": [1, len_prefix, len_prefix+t+1]
                            }
                            }
            step_data.update(common_info)

            step_encode = model._forward_data([step_data], task_info=task_info, history_states=history_states, return_all=False)

            step_predictor_input = {
                "input_sample_list": step_encode,
                "target_sample_list": [],
                "shared_target_sets": {self.vocab_name: [vocab_encode]},
                "target_set_list": [self.vocab_name],
                "target_idx_list": [],
                "task_info": task_info
            }
            logit = model.loss_prepare(**step_predictor_input)['output']

            with autocast(enabled=not self.cfg.SOLVER.FORCE_SOFTMAX_FP16):
                word_logprob = F.log_softmax(logit, dim=-1)
            word_logprob = word_logprob.view(bs, cur_beam_size, -1)
            candidate_logprob = seq_logprob + word_logprob

            # # Mask sequence if it reaches EOS
            # if t > 0:
            #     mask = (selected_words.view(bs, cur_beam_size) != 0).float().unsqueeze(-1) # 为什么是不等于0
            #     seq_mask = seq_mask * mask
            #     word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
            #     old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
            #     old_seq_logprob[:, :, 1:] = -999
            #     candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            if t > 0:
                mask = (selected_words.view(bs, cur_beam_size) != self.eos_token_id).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, :self.eos_token_id] = -999
                old_seq_logprob[:, :, self.eos_token_id + 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            selected_idx, selected_logprob = self._select(bs, beam_size, t, candidate_logprob) # bs beam
            selected_beam = torch.div(selected_idx, candidate_logprob.shape[-1], rounding_mode='floor')
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

            self._expand_state(history_states, selected_beam, bs, beam_size, cur_beam_size)

            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1,
                selected_beam.unsqueeze(-1).expand(bs, beam_size, word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(bs, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            # wt = selected_words

            if t == 0:
                u_tokens_type = expand_tensor(u_tokens_type, beam_size)
                wt = expand_tensor(wt, beam_size)

            step_selected_data = {"data": selected_words, "time_step": t, "sample_info": {"data_cum_length":  [1, len_prefix, len_prefix+t+1]}}
            step_selected_data.update(common_info)

            step_selected_encode = model._forward_data([step_selected_data], task_info=task_info, history_states=history_states, return_all=True)

            for i, ths in enumerate(total_history_states):
                hs = history_states[i]
                ths[:hs.shape[0], :hs.shape[1], :] = hs
                ths[:hs.shape[0], hs.shape[1], :] = step_selected_encode[0]['data'][i].squeeze(1)

        outputs = torch.cat(outputs, -1)


        if self.len_penalty > 0:
            step = outputs.ne(self.eos_token_id).sum(-1, keepdim=True) + 1
            seq_logprob /= step ** self.len_penalty
        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)

        outputs = torch.gather(outputs, 1, sort_idxs.expand(bs, beam_size, self.max_seq_len))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(bs, beam_size, self.max_seq_len))

        outputs = outputs.contiguous()[:, :out_size]
        log_probs = log_probs.contiguous()[:, :out_size]
        if out_size == 1:
            outputs = outputs.squeeze(1)
            log_probs = log_probs.squeeze(1)

        comm._CAPTION_GEN_MODE = False

        ids = torch.stack([torch.tensor(v['id']) for v in inputs['input_sample_list'][0]['sample_info']])

        return {
            "IDS": ids,
            "G_SENTS_IDS": outputs,
            "G_LOGP": log_probs
        }
