
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


@DECODE_STRATEGY_REGISTRY.register()
class CaptionBeamSearcherV2(DecodeStrategy):

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


        # 0. token embedding
        if model.visual_embed is not None:
            # ve_out = model.visual_embed(batched_inputs)
            # inputs.update(ve_out)
            model.visual_embed(inputs)


        if model.video_embed is not None:
            # ve_out = model.video_embed(batched_inputs)
            # inputs.update(ve_out)
            model.video_embed(inputs)

        if model.token_embed is not None:
            # te_out = model.token_embed(batched_inputs)
            # inputs.update(te_out)
            model.token_embed(inputs)

        prompt_data = {}
        if model.prompt_embed is not None:
            prompt_data = model.prompt_embed(batched_inputs)
            prompt_data[kfg.DEEP_PROMPT] = model.prompt and model.deep_prompt
            inputs.update(prompt_data)

        # 1. encode the image/video sequence.
        # bs = inputs[kfg.ATT_FEATS].size(0)
        bs = inputs['images'].size(0)

        v_input = []
        # v_input.append(model._get_sep_embed(inputs, bs))
        v_input.append(model._get_sep_embed(inputs.get('spe_token_embed', None), bs))
        # v_input.append(inputs[kfg.ATT_FEATS])
        # comm._LOCAL_IMAGE_LENGTH = inputs[kfg.ATT_FEATS].shape[1]
        comm._LOCAL_IMAGE_LENGTH = inputs['images'].shape[-1]
        # add by zjg
        if kfg.PROMPT_EMBED in inputs and not model.deep_prompt:
            v_input.append(batched_inputs[kfg.PROMPT_EMBED])

        v_input = torch.cat(v_input, dim=1)

        # ext_u_tmasks = torch.ones((bs, v_input.shape[1], v_input.shape[1]), dtype=v_input.dtype, device=v_input.device)
        # ext_u_tmasks = ext_u_tmasks.unsqueeze(1)
        # ext_u_tmasks = (1.0 - ext_u_tmasks) * -10000.0
        # for img encoder, do not need mask
        v_input = {
            kfg.MM_EMBEDS: v_input,
            # kfg.ATT_FEATS: inputs[kfg.ATT_FEATS],
            kfg.TEXT_GEN_MODE: False,
            kfg.EXT_U_TOKENS_MASKS: None,
        }

        # for deep prompt tuning
        if prompt_data.get(kfg.DEEP_PROMPT, False):
            v_input.update(prompt_data)


        # masks = model.get_extended_attention_mask(v_input)
        # v_input.update(masks)

        # v_input.update( {kfg.EXT_U_TOKENS_MASKS: v_input[kfg.EXT_U_TOKENS_MASKS][:, :, :, 1:]} ) # remove the mask for special token
        # vfeats = model.encoder(v_input)[kfg.U_HIDDEN_STATES]

        # 2. encode the dictionary - if no pre-computed, add that into input
        if getattr(self, 'pre_computed_word_embeds', None) is None:
            w_input = []
            vocab_size = model.token_embed.embeddings.num_embeddings
            w_input.append(model._get_sep_embed(inputs.get('spe_token_embed', None), vocab_size))

            # range_slice = torch.arange(start=0, end=vocab_size).unsqueeze(1).to(inputs[kfg.ATT_FEATS].device)
            range_slice = torch.arange(start=0, end=vocab_size).unsqueeze(1).to(inputs['images'].device)
            # - [HACK] we hardcode the EOT token
            eot_to_append = range_slice.new_full(range_slice.shape, 49407)
            range_slice_concat_eot = torch.cat([range_slice, eot_to_append], dim=1)
            # temp = {
            #         kfg.U_TOKENS_IDS: range_slice_concat_eot,
            #         kfg.U_TOKENS_TYPE: torch.zeros_like(range_slice_concat_eot)
            # }
            temp = {
                    "shared_targets": [{
                       "shared_tgt_tokens":range_slice_concat_eot,
                    },
                    ]    
                    # kfg.U_TOKENS_TYPE: torch.zeros_like(range_slice_concat_eot)
            }
            
            # word_embeddings = model.token_embed(temp)['shared_tgt_token_embed']
            model.token_embed(temp)
            word_embeddings = temp["shared_targets"][0]['shared_tgt_token_embed']

            w_input.append(word_embeddings)
            w_input = torch.cat(w_input, dim=1)
            v_input.update({ kfg.WORD_EMBEDS: w_input })

            v_input = self.data_half(v_input)

            model.add_tag_embedding(v_input)

            enc_out = model.encoder(v_input, return_all=True)
            self.pre_computed_word_embeds = enc_out[kfg.WORD_HIDDEN_STATES]
            vfeats = enc_out[kfg.U_HIDDEN_STATES]
        else:
            v_input = self.data_half(v_input)
            vfeats = model.encoder(v_input, return_all=True)[kfg.U_HIDDEN_STATES]

        # 3. compute attention

        comm._CAPTION_GEN_MODE = True

        beam_size = self.beam_size
        log_probs = []
        selected_words = None
        seq_logprob = torch.zeros((bs, 1, 1)).cuda() # bs, 1, 1
        seq_mask = torch.ones((bs, beam_size, 1)).cuda()
        wt = Variable(torch.zeros(bs, dtype=torch.long).cuda().unsqueeze(1)) + self.spe_token_id
        u_tokens_type = wt.new_zeros(wt.shape) # [Note] we assume the type tokens are 0.

        history_states = vfeats[:-1]
        len_prefix = history_states[0].shape[1]
        total_history_states = [ history_states[0].new_zeros(beam_size * bs, vfeats[0].shape[1] + self.max_seq_len, vfeats[0].shape[2]) for _ in history_states]
        for i, ths in enumerate(total_history_states):
            hs = history_states[i]
            ths[:hs.shape[0], :hs.shape[1], :] = hs

        outputs = []
        for t in range(self.max_seq_len):
            cur_beam_size = 1 if t == 0 else beam_size

            history_states = [ ths[ :(cur_beam_size*bs), :(len_prefix+t), :] for ths in total_history_states]
            t_input = {
                kfg.U_TOKENS_IDS: wt,
                kfg.U_TOKENS_TYPE: u_tokens_type,
                kfg.EXT_U_TOKENS_MASKS: None,
                kfg.HISTORY_STATES: history_states,
                kfg.TIME_STEP: t
            }

            vt_out = model.token_embed(t_input)
            t_input.update(vt_out)

            t_input.update({ kfg.MM_EMBEDS: t_input[kfg.U_TOKEN_EMBED] })

            if prompt_data.get(kfg.DEEP_PROMPT, False) and prompt_data['PROMPT_EMBED'].shape[1] != t_input[
                        'MM_EMBEDS'].shape[0]:
                prompt_data['PROMPT_EMBED'] = prompt_data[
                    'PROMPT_EMBED'][:, :1].expand(
                        -1, t_input['MM_EMBEDS'].shape[0], -1, -1)
            t_input.update(prompt_data)

            t_input = self.data_half(t_input)
            encoder_out = model.encoder(t_input, return_all=True)

            pred_input = {
                kfg.TEXT_GEN_MODE: True,
                kfg.WORD_HIDDEN_STATES: self.pre_computed_word_embeds,
                kfg.U_HIDDEN_STATES: encoder_out[kfg.U_HIDDEN_STATES],
                kfg.TASK_NAME: batched_inputs[kfg.TASK_NAME]
            }

            logit = model.predictor(pred_input, force_spe_first=True)[kfg.OUTPUT]
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

            eos_id = 49407
            if t > 0:
                mask = (selected_words.view(bs, cur_beam_size) != eos_id).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, :eos_id] = -999
                old_seq_logprob[:, :, eos_id + 1:] = -999
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

            selected_t_input = {
                kfg.U_TOKENS_IDS: selected_words,
                kfg.U_TOKENS_TYPE: u_tokens_type,
                kfg.EXT_U_TOKENS_MASKS: None,
                kfg.HISTORY_STATES: history_states,
                kfg.TIME_STEP: t
            }
            selected_vt_out = model.token_embed(selected_t_input)
            selected_t_input.update(selected_vt_out)

            selected_t_input.update({ kfg.MM_EMBEDS: selected_t_input[kfg.U_TOKEN_EMBED] })

            selected_t_prompt_data = dict(prompt_data)
            if selected_t_prompt_data.get(kfg.DEEP_PROMPT, False) and  selected_t_prompt_data['PROMPT_EMBED'].shape[1] != selected_t_input['MM_EMBEDS'].shape[0]:
                selected_t_prompt_data['PROMPT_EMBED'] = selected_t_prompt_data['PROMPT_EMBED'][:, :1].expand(
                    -1, selected_t_input['MM_EMBEDS'].shape[0], -1, -1)
            selected_t_input.update(selected_t_prompt_data)

            selected_t_input = self.data_half(selected_t_input)
            selected_encoder_out = model.encoder(selected_t_input, return_all=True)

            for i, ths in enumerate(total_history_states):
                hs = history_states[i]
                ths[:hs.shape[0], :hs.shape[1], :] = hs
                ths[:hs.shape[0], hs.shape[1], :] = selected_encoder_out[kfg.U_HIDDEN_STATES][i].squeeze(1)

                # expand_keys = {
                #     kfg.ATT_FEATS,
                #     kfg.GLOBAL_FEATS,
                #     kfg.ATT_MASKS,
                #     kfg.EXT_ATT_MASKS,
                #     kfg.P_ATT_FEATS,
                #     kfg.EXT_G_TOKENS_MASKS,
                #     kfg.G_TOKENS_TYPE
                # }
                # for key in expand_keys:
                #     if key in inputs:
                #         if isinstance(inputs[key], list):
                #             inputs[key] = inputs[key][-1] # usually is ATT_FEATS in TDEN
                #         tensor = expand_tensor(inputs[key], beam_size)
                #         inputs.update({ key: tensor })

        outputs = torch.cat(outputs, -1)


        if self.len_penalty > 0:
            step = outputs.ne(49407).sum(-1, keepdim=True) + 1
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

        return {
            kfg.IDS: batched_inputs[kfg.IDS],
            kfg.G_SENTS_IDS: outputs,
            kfg.G_LOGP: log_probs
        }
