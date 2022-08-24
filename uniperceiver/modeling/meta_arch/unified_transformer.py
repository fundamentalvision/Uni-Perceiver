import os
import pickle
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import weakref

from uniperceiver.utils.transformer_util import data_half, preprocess, postprocess, null_loss_check
from uniperceiver.config import configurable
from uniperceiver.functional import pad_tensor, dict_to_cuda, dict_as_tensor
from ..predictor import build_v_predictor
from .build import META_ARCH_REGISTRY
from ..embedding import build_embeddings
from ..encoder import build_encoder, add_encoder_config, build_unfused_encoders
from ..predictor import build_predictor, add_predictor_config
from collections import defaultdict
from omegaconf import DictConfig
from ..decode_strategy import build_beam_searcher, build_greedy_decoder
from .base_enc_dec import BaseEncoderDecoder
from uniperceiver.modeling.predictor import EmbedClsAsRetrievalPredictor
from torch.nn import init
import math
from uniperceiver.utils import comm
import  torch.distributed.nn
from uniperceiver.tokenization import ClipTokenizer
import logging
from uniperceiver.losses import build_losses


__all__ = ["MultiTaskTransformerEncoder"]


@META_ARCH_REGISTRY.register()
class MultiTaskTransformerEncoder(BaseEncoderDecoder):

    @configurable
    def __init__(
        self,
        *,
        task_modules,
        fused_encoder,
        unfused_encoders,
        decoder,
        token_embed,
        video_embed,
        prompt_embed,
        loss_prepare,
        vocab_size,
        imagenet_tuning,
        cfg,
    ):
        super().__init__(fused_encoder=fused_encoder,
                         decoder=decoder,
                         vocab_size=vocab_size,
                         token_embed=token_embed,
                         **list(task_modules.values())[0])

        self.unfused_encoders = unfused_encoders
        for name, module in self.unfused_encoders.items():
            self.add_module(name, module)
        self.video_embed = video_embed
        self.prompt_embed = prompt_embed
        self.task_modules = dict()
        self.module_names = set()
        self.imagenet_tuning = imagenet_tuning
        self.cfg = cfg

        self.losses = self.build_losses(cfg)

        self.tokenizer = ClipTokenizer()

        self.loss_prepare = loss_prepare


        for task_name, task_module in task_modules.items():
            self.task_modules[task_name] = nn.Module()
            for module_name, sub_module in task_module.items():
                setattr(self.task_modules[task_name], module_name, sub_module)
                self.module_names.add(module_name)
                self.process_module(sub_module)
            self.add_module(task_name,self.task_modules[task_name])



        if self.cfg.MODEL.SHARE_LAYERNORM:
            from uniperceiver.utils.transformer_util import share_token_embed_ln
            share_token_embed_ln(self.video_embed, self.token_embed)

        self.prepare_prompt_embed(cfg)

        self.fp16 = self.cfg.SOLVER.AMP_FP16
        self.bf16 = self.cfg.SOLVER.BF16



        if self.token_embed is None:
            # used for standard classification head
            self.cls_token = nn.Embedding(1,cfg.MODEL.BERT.HIDDEN_SIZE)


        self.initialize(cfg)

        # init fc prompt layer
        if self.use_fc_prompt and self.prompt:
            nn.init.zeros_(self.fc_prompt.weight)
            nn.init.zeros_(self.fc_prompt.bias)


        self.logger = logging.getLogger(__name__)

        if not  self.cfg.MODEL.OLD_CHECKPONT:
            comm.old_checkpoint = False
            self.logger.info(f'please note that the <|spe|> is \'spe\' now!')

    def prepare_prompt_embed(self, cfg):

        self.prompt = cfg.MODEL.PROMPT
        self.deep_prompt = cfg.MODEL.PROMPT_EMBED.DEEP_PROMPT
        self.use_fc_prompt = cfg.MODEL.FC_PROMPT
        prompt_params = cfg.MODEL.PROMPT_PARAM
        fc_prompt_out = cfg.MODEL.FC_PROMPT_OUT
        fc_prompt_weights = cfg.MODEL.FC_PROMPT_WEIGHTS

        if self.prompt and 's_token_bias' in prompt_params:
            self.s_token_bias = nn.Parameter(torch.zeros((1, self.token_embed.embeddings.weight.size(1)), device=self.token_embed.embeddings.weight.device))
            self.token_embed.set_s_token_bias(self.s_token_bias)

        if self.use_fc_prompt:
            self.fc_prompt = nn.Linear(self.cfg.MODEL.BERT.HIDDEN_SIZE, fc_prompt_out)
            if fc_prompt_weights == 'learn':
                self.similarity_weight = nn.Parameter(torch.ones([]))
            elif fc_prompt_weights == 'zero':
                self.similarity_weight = 0.
            else:
                raise NotImplementedError

        if self.prompt:
            for name, param in self.named_parameters():
                if not any([p_param in name for p_param in prompt_params]):
                    param.requires_grad = False


    def initialize(self, cfg ):
        if cfg.MODEL.TimmParamsInit:
            global INIT_STD
            INIT_STD = cfg.MODEL.TimmParamsInitSTD
            global INIT_EMBEDDING_STD
            INIT_EMBEDDING_STD = cfg.MODEL.TimmParamsINIT_EMBEDDING_STD
            from uniperceiver.utils.transformer_util import init_timm_params
            self.apply(init_timm_params)
        elif cfg.MODEL.MAEParamsInit:
            from uniperceiver.utils.transformer_util import initialize_weights_as_mae
            initialize_weights_as_mae(self)
        elif cfg.MODEL.MOCOv3ParamsInit:
            from uniperceiver.utils.transformer_util import initialize_weights_as_mocov3
            initialize_weights_as_mocov3(self)
        elif cfg.MODEL.SwitchParamsInit:
            from uniperceiver.utils.transformer_util import init_switchtransformer_params
            self.apply(init_switchtransformer_params)
        elif cfg.MODEL.BertParamsInit:
            from uniperceiver.utils.transformer_util import init_bert_params
            self.apply(init_bert_params)
        elif cfg.MODEL.UniformTokenEmbed:
            init.kaiming_uniform_(self.token_embed.embeddings.weight, a=math.sqrt(5))
        else:
            print('please check your parameters initialization method!')

    @classmethod
    def build_losses(cls, cfg):
        losses = {}
        for task_config in cfg.TASKS:
            task_config = DictConfig(task_config)
            losses[task_config.NAME] = build_losses(task_config)

        return losses

    def process_module(self, submodule):
        '''
        process some submodule
        '''
        if isinstance(submodule, EmbedClsAsRetrievalPredictor):
            submodule.replace_weight(self.token_embed.embeddings.weight)


    def operatedweight(self, ):
        pass


    @classmethod
    def from_config(cls, cfg):
        task_names = [ a['NAME'] for a in cfg.TASKS]
        task_modules = defaultdict(dict)

        for idx, task_names in enumerate(task_names):
            cfg_task = DictConfig(cfg.TASKS[idx])
            this_task_modules = {

            "greedy_decoder": None,
            "beam_searcher": None if getattr(cfg_task, 'DECODE_STRATEGY', None) is None
            else build_beam_searcher(cfg_task),
            # "vocab_size": cfg_task.MODEL.VOCAB_SIZE,
            "max_seq_len": cfg_task.MODEL.MAX_SEQ_LEN,
            }

            task_modules[task_names].update(this_task_modules)

        if cfg.SOLVER.AUGLOSS:
            num_augloss = (cfg.MODEL.BERT.NUM_HIDDEN_LAYERS - max(
                0, cfg.SOLVER.AUGLOSS_START)) // cfg.SOLVER.AUGLOSS_INTERVAL
        ret = {
            "task_modules":
            task_modules,
            "fused_encoder":
            build_encoder(cfg),
            "unfused_encoders":
            build_unfused_encoders(cfg),
            "decoder":
            None,
            "loss_prepare":
            build_predictor(cfg) if not cfg.SOLVER.AUGLOSS else nn.ModuleList(build_predictor(cfg) for _ in range(num_augloss)),
            "vocab_size":
            cfg.MODEL.VOCAB_SIZE,
            "prompt_embed":
            None if getattr(cfg.MODEL, 'PROMPT_EMBED', None) is None or not cfg.MODEL.PROMPT else build_embeddings(
                cfg, cfg.MODEL.PROMPT_EMBED.NAME),
            "imagenet_tuning":
            cfg.MODEL.IN_TUNING,

            "token_embed":  None if not getattr(cfg.MODEL.TOKEN_EMBED, 'NAME', None)
            else build_embeddings(cfg, cfg.MODEL.TOKEN_EMBED.NAME),
            "video_embed": None if not getattr(cfg.MODEL.VIDEO_EMBED, 'NAME', None)
            else build_embeddings(cfg, cfg.MODEL.VIDEO_EMBED.NAME),
            "cfg": cfg,
            }


        return ret

    @classmethod
    def add_config(cls, cfg, tmp_cfg):
        add_encoder_config(cfg, tmp_cfg)
        # we do not have decoder anymore
        # add_decoder_config(cfg, tmp_cfg)
        cfg.MODEL.SharePredictor = False
        cfg.MODEL.UniformTokenEmbed = False
        cfg.MODEL.BertParamsInit = False

    def to_task(self, task_name):
        # in train_loop, you do not need to reset_atrr explictly
        self.reset_attr()
        for name in self.module_names:
            setattr(self, name, getattr(self.task_modules[task_name], name))

    def reset_attr(self):
        for name in self.module_names:
            # in case different task has different modules
            if  getattr(self, name, 'none') != 'none':
                delattr(self, name)


    def _forward(self, batched_inputs):


        batched_inputs = data_half(self.fp16, self.bf16, batched_inputs)

        #TODO: add imagenet classname and word in evaluation mode

        task_info = batched_inputs['task_info']



        batched_inputs['input_sample_list'] = self._forward_data(
            batched_inputs['input_sample_list'], task_info=task_info)

        if batched_inputs['target_sample_list'] is not None and len(batched_inputs['target_sample_list']) > 0:
            batched_inputs['target_sample_list'] = self._forward_data(batched_inputs['target_sample_list'], task_info=task_info)


        for target_set_name, data_list in batched_inputs['shared_target_sets'].items():
            if data_list is not None and len(data_list)>0:
                batched_inputs['shared_target_sets'][target_set_name] = self._forward_data(data_list, task_info=task_info)

        loss_inputs = self.loss_prepare(**batched_inputs)

        self.fc_prompt_process(loss_inputs)

        if self.training:
            # training mode
            loss_dict = {}
            for loss in self.losses[task_info['task_name']]:
                loss_dict.update(loss(loss_inputs))

        # if self.load_balance_losses is not None:
        #     loss_dict.update(self.load_balance_losses(batched_inputs))

            loss_dict.update(null_loss_check(outputs_dict=batched_inputs))
            return loss_dict
        else:
            # evaluation mode
            return loss_inputs

    def fc_prompt_process(self, outputs_dict):
        if self.prompt and self.use_fc_prompt:
            for idx, logit in enumerate(outputs_dict['logits']):
                assert 'feats' in outputs_dict
                feat = outputs_dict['feats'][idx]
                logit = self.similarity_weight * logit + self.fc_prompt(feat)
                outputs_dict['logits'][idx] = logit
                if 'output' in outputs_dict:
                    outputs_dict['output'] = logit



    def _forward_data(self, data_list:list, task_info:dict, history_states=None, return_all=False):

        # data is dict value
        for data in data_list:

            data = data_half(self.fp16, self.bf16, data)

            self._tokenize(data, task_info)

            self._forward_unfused_encoders(data, task_info)

        # fused encoders
        if self.prompt_embed is not None:
            # prefix_prompt, label prompt
            self.prompt_embed(data_list=data_list)
        fused_data_dict = preprocess(self.tokenizer, self.token_embed, data_list, task_info=task_info)

        fused_data_dict = data_half(self.fp16, self.bf16, fused_data_dict)
        fused_data_dict['data'] = self.fused_encoder(**fused_data_dict, task_info=task_info, history_states=history_states, return_all=return_all)

        postprocess(fused_data_dict, task_info=task_info)

        return [fused_data_dict]

    def _tokenize(self, data, task_info):
        # toknizer
        if data['modality'] in ['image', 'video']:
            data['data'] = self.video_embed(**data, task_info=task_info)
        elif data['modality'] == 'text':
            data['data'] = self.token_embed(**data, task_info=task_info)
        else:
            raise NotImplementedError


    def _forward_unfused_encoders(self, data, task_info):


        # specific encoders.
        # defaultly, modality-specific encoder
        if data['modality'] in ['image', 'video']:
            if "VisualEncoder" in self.unfused_encoders:
                data['data'] = self.unfused_encoders['VisualEncoder'](**data, task_info=task_info)
        elif data['modality'] == 'text':
            if "TextEncoder" in self.unfused_encoders:
                data['data'] = self.unfused_encoders['TextEncoder'](**data, task_info=task_info)
        else:
            raise NotImplementedError





    @torch.jit.ignore
    def no_weight_decay(self,):
        ret = [
            'logit_scale', 'logit_scale_img_cls', 'logit_scale_video_cls',
            'logit_scale_text_mlm', 'logit_scale_text_caption',
            'logit_scale_caption', 'logit_scale_mlm', 'logit_scale_retrieve',
            'logit_scale_text_retrieve', "logit_scale_downstream",
            "logit_scale_tqa_mlm", "logit_scale_tqa_caption",
            "logit_scale_tqa_retrieve", "similarity_weight", "gamma_1", "gamma_2",
        ]
        if self.cfg.SOLVER.OUTPUTPROJ_NOWD:
            ret.append("predictor.proj")
        return ret

    @torch.jit.ignore
    def expert_gate_group(self, ):
        return ['gate.wg', 'gate.tag_transform']



    def load_state_dict(self, state_dict, strict=True):
        out_dict = {}
        if self.cfg.MODEL.CHECKPOINT_FILETER:
            def resize_pos_embed(posemb, posemb_new, cls_token=False):
                # Rescale the grid of position embeddings when loading from state_dict. Adapted from
                # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
                self.logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
                ntok_new = posemb_new.shape[0]
                posemb_tok = posemb
                if not cls_token:
                    posemb_grid = posemb
                else:
                    raise NotImplementedError
                gs_old = int(math.sqrt(len(posemb_grid)))
                gs_new = int(math.sqrt(ntok_new))


                self.logger.info('Position embedding grid-size from %s to %s',
                                 gs_old, gs_new)
                posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
                posemb_grid = F.interpolate(posemb_grid.float(), size=(gs_new, gs_new), mode='bilinear')
                posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1).squeeze(0)
                if cls_token:
                    posemb_grid = torch.cat([posemb_tok, posemb_grid], dim=1)
                return posemb_grid.to(posemb_new.dtype)
            # 'convert patch embedding weight from manual patchify'

            for k, v in state_dict.items():
                if k.startswith('video_embed.embeddings_st_pos.spatial_pos_embed') or k.startswith('visual_embed.patch_embed.pos_embed'):
                    # To resize pos embedding when using model at different size from pretrained weights
                    if v.shape != self.state_dict()[k].shape:
                        v = resize_pos_embed(v, self.state_dict()[k])

                out_dict[k] = v
        else:

            for k, v in state_dict.items():
                if k.startswith('video_embed.embeddings_st_pos.spatial_pos_embed') or k.startswith('visual_embed.patch_embed.pos_embed'):
                    # To resize pos embedding when using model at different size from pretrained weights
                    if v.shape != self.state_dict()[k].shape:
                        # v = resize_pos_embed(v, self.state_dict()[k])
                        continue
                out_dict[k] = v

        if self.cfg.MODEL.CHECKPOINT_FILETER_VIDEO:

            def resize_temporal_pos_embed(posemb, posemb_new, cls_token=False):
                # Rescale the grid of position embeddings when loading from state_dict. Adapted from
                # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
                self.logger.info('Resized position embedding: %s to %s',
                                 posemb.shape, posemb_new.shape)
                ntok_new = posemb_new.shape[0]
                if not cls_token:
                    posemb_grid = posemb
                else:
                    raise NotImplementedError
                gs_old = len(posemb_grid)
                gs_new = ntok_new

                self.logger.info('temporal embedding grid-size from %s to %s',
                                 gs_old, gs_new)
                posemb_grid = posemb_grid.reshape(1, gs_old,
                                                  -1).permute(0, 2, 1)
                posemb_grid = F.interpolate(posemb_grid.float(),
                                            size=(gs_new),
                                            mode='linear')
                posemb_grid = posemb_grid.permute(0, 2, 1).squeeze(0)

                return posemb_grid.to(posemb_new.dtype)

            # 'convert patch embedding weight from manual patchify'
            for k, v in out_dict.items():
                if k.startswith(
                        'video_embed.embeddings_st_pos.temporal_pos_embed'
                ) :
                    # To resize pos embedding when using model at different size from pretrained weights
                    if v.shape != self.state_dict()[k].shape:
                        v = resize_temporal_pos_embed(v, self.state_dict()[k])

                out_dict[k] = v


        return super().load_state_dict(out_dict, strict=strict)