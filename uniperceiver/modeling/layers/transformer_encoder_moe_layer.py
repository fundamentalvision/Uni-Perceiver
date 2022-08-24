import torch
from torch import nn
import torch.nn.functional as F

from typing import Optional, Any, Union, Callable
from torch import Tensor

from .create_act import get_act_layer, get_activation
from timm.models.layers import DropPath
from .layer_norm import LayerNorm
from .pe_encoder import DeepPrompt
from uniperceiver.task_moe.layer import TaskMoE
from uniperceiver.utils import comm
from functools import partial
import math
from uniperceiver.modeling.layers import FP16LayerNorm
from torch.cuda.amp import autocast

class MoETransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    Fast path:
        forward() will use a special optimized implementation if all of the following
        conditions are met:
        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - norm_first is ``False`` (this restriction may be loosened in the future)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)
        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.
    """
    __constants__ = ['batch_first', 'norm_first'] # we inherit this variable from pytorch's code for jit

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, drop_path_ratio: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, layer_scale: bool = False, ls_init_values: float = 1e-3,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None, cfg: dict = None, ffn_moe: bool = False, attn_moe: bool = False) -> None:

        if batch_first and comm.is_main_process():
            print(f'set batch_first to \'False\' to support torch >= 1.12!')
            batch_first = False

        factory_kwargs = {}
        super(MoETransformerEncoderLayer, self).__init__()

        self.cfg = cfg

        # The interface of nn.MultiheadAttention changed since torch 1.9.0.
        _torch_version_main = torch.__version__.split('.')[:2]
        if (int(_torch_version_main[0]) >= 1) and (int(_torch_version_main[1])) >= 9:
            self._torch_nn_new_interface = True
        else:
            self._torch_nn_new_interface = False

        # for moe
        self.ffn_moe = ffn_moe and self.cfg.MOE.MOE
        self.attn_moe = attn_moe and self.cfg.MOE.MOE
        if self.cfg.MOE.MOE:
            # assert self.ffn_moe and self.attn_moe
            # data-independent moe
            if self.cfg.MOE.MOE_TYPE in ['attribute']:
                MoE_layer = partial(
                    TaskMoE,
                    num_experts=cfg.MOE.NUM_EXPERTS,
                    k=cfg.MOE.TOP_K,
                    capacity_factor=cfg.MOE.CAPACITY_FACTOR,
                    eval_capacity_factor=cfg.MOE.EVAL_MIN_CAPACITY,
                    min_capacity=cfg.MOE.MIN_CAPACITY,
                    noisy_gate_policy=cfg.MOE.NOISY_GATE_POLICY,
                    use_rts=cfg.MOE.USE_RTS,
                    use_tutel=cfg.MOE.USE_TUTEL,
                    cfg=cfg,
                )
            else:
                raise NotImplementedError(f'{self.cfg.MOE.MOE_TYPE}')



        self.self_attn = MoEAttentionBlock(d_model, nhead, attention_probs_dropout_prob=dropout, cfg=cfg, moe_layer=MoE_layer, attn_moe=attn_moe)


        self.batch_first = batch_first

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        if self.ffn_moe:
            self.linear1 = MoE_layer(hidden_size=d_model, expert=self.linear1)
            self.linear2 = MoE_layer(hidden_size=d_model, expert=self.linear2)

        self.norm_first = norm_first
        if self.cfg.SOLVER.FUSED_LAYERNORM:
            self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
            self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        elif self.cfg.SOLVER.FORCE_LN_FP16:
            self.norm1 = FP16LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = FP16LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.drop_path1 = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

        self.layer_scale = layer_scale
        if self.layer_scale:
            self.gamma_1 = nn.Parameter(ls_init_values * torch.ones((d_model)),requires_grad=True)
            self.gamma_2 = nn.Parameter(ls_init_values * torch.ones((d_model)),requires_grad=True)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = get_activation(activation)

        self.activation = activation

        # prompt embedding setup
        self.deep_prompt = self.cfg.MODEL.PROMPT_EMBED.DEEP_PROMPT
        if self.deep_prompt:
            self.deep_prompt_embedding = DeepPrompt(cfg)




    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(MoETransformerEncoderLayer, self).__setstate__(state)

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                history_states: Optional[Tensor] = None,
                **kwargs) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src

        if self.norm_first:
            history_states_norm = history_states if (history_states is None) else self.norm1(history_states)
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, history_states=history_states_norm, **kwargs)
            x = x + self._ff_block(self.norm2(x), **kwargs)
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, history_states=history_states, **kwargs))
            x = self.norm2(x + self._ff_block(x), **kwargs)


        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], history_states: Optional[Tensor],
                  **kwargs) -> Tensor:

        if history_states is not None:
            kv = torch.cat(
                [history_states, x],
                dim=1
            )
            # TODO: changes for attn_mask and key_padding_mask
        else:
            kv = None

        if self.deep_prompt:

            deep_prompt_embedding = self.deep_prompt_embedding(x, batch_first=True, **kwargs)
            if self.norm_first:
                deep_prompt_embedding = self.norm1(deep_prompt_embedding)
            kv = torch.cat([deep_prompt_embedding, x], dim=1) if kv is None else torch.cat([deep_prompt_embedding, kv], dim=1)
            if 'sample_info' in kwargs:
                pe_length = deep_prompt_embedding.shape[1]
                kwargs['sample_info']['pe_length'] = pe_length
            if attn_mask is not None:
                L, S = attn_mask.shape
                pe_length = deep_prompt_embedding.shape[1]  # length, bs, hidden_size
                attn_mask = torch.cat([torch.zeros((L, pe_length), dtype=attn_mask.dtype, device=attn_mask.device), attn_mask], dim=1)
            if key_padding_mask is not None:

                bs, pe_length = deep_prompt_embedding.shape[:2]

                key_padding_mask = torch.cat(
                    [torch.zeros((bs, pe_length), dtype=key_padding_mask.dtype, device=key_padding_mask.device), key_padding_mask], dim=1)


        x, _ = self.self_attn(x, history_states=kv, attn_mask=attn_mask, key_padding_mask=key_padding_mask, **kwargs)
        x = self.drop_path1(self.dropout1(x))
        if self.layer_scale:
            if self.cfg.MODEL.LAYER_SCALE_FP32:
                x = self.gamma_1 * x
            else:
                x = self.gamma_1.to(x.dtype) * x
        return x


    # feed forward block
    def _ff_block(self, x: Tensor, **kwargs) -> Tensor:
        if self.ffn_moe:
            x, gate_decision = self.linear1(x, **kwargs)
            if not self.cfg.MOE.FFN_SHARE_GATE_DECISION:
                gate_decision = None
            x, _ = self.linear2(self.dropout(self.activation(x)), gate_decision=gate_decision, **kwargs)
        else:
            x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.drop_path2(self.dropout2(x))
        if self.layer_scale:
            if self.cfg.MODEL.LAYER_SCALE_FP32:
                x = self.gamma_2 * x
            else:
                x = self.gamma_2.to(x.dtype) * x
        return x


class MoEAttentionBlock(nn.Module):

    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, cfg, moe_layer=None, attn_moe=False):
        super(MoEAttentionBlock, self).__init__()
        self.cfg = cfg
        if hidden_size % num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention " "heads (%d)" % (hidden_size, num_attention_heads))

        self.hidden_size = hidden_size

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.qkv_bias = cfg.MODEL.BERT.QKV_BIAS

        self.unify_qkv = cfg.MODEL.BERT.UNIFY_QKV

        if not cfg.MODEL.BERT.UNIFY_QKV:
            self.query = nn.Linear(hidden_size, self.all_head_size, bias=self.qkv_bias)
            self.key = nn.Linear(hidden_size, self.all_head_size, bias=self.qkv_bias)
            self.value = nn.Linear(hidden_size, self.all_head_size, bias=self.qkv_bias)
        else:
            self.qkv_proj = nn.Linear(hidden_size, self.all_head_size * 3, bias=self.qkv_bias)

        self.dense = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        self.attn_moe = attn_moe
        if self.attn_moe:
            if not cfg.MODEL.BERT.UNIFY_QKV:
                raise NotADirectoryError('use UNIFY_QKV=True please')
            else:
                self.qkv_proj = moe_layer(hidden_size=hidden_size, expert=self.qkv_proj)
                self.dense = moe_layer(hidden_size=hidden_size, expert=self.dense)

        self.scale_multi_before = cfg.MODEL.BERT.SCALE_MULTI_BEFORE

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)

        shape_list = list(range(len(new_x_shape)))
        shape_list[-2], shape_list[-3] = shape_list[-3], shape_list[-2]
        return x.permute(shape_list)
        #return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attn_mask, key_padding_mask, history_states=None,  **kwargs):
        if attn_mask is None and key_padding_mask is None:
            attention_mask = None
        else:
            # attn_mask [L, S]  key_padding_mask[N, S]
            if attn_mask is not None and key_padding_mask is not None:
                attention_mask = torch.logical_or(attn_mask.unsqueeze(0).bool(), key_padding_mask.unsqueeze(1).bool())
            elif attn_mask is not None:
                attention_mask = attn_mask.unsqueeze(0)
            else:
                attention_mask = key_padding_mask.unsqueeze(1)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1) * -10000.0

        if self.unify_qkv:
            if history_states is None:

                B, N, C = hidden_states.shape
                if self.attn_moe:
                    # qkv, _, _ = self.self.qkv_proj(hidden_states)
                    hidden_states, gate_decision = self.qkv_proj(hidden_states, **kwargs)
                    mixed_query_layer, mixed_key_layer, mixed_value_layer =hidden_states.chunk(3, dim=-1)
                else:
                    mixed_query_layer, mixed_key_layer, mixed_value_layer = self.qkv_proj(hidden_states).chunk(3, dim=-1)

            else:
                # usually inference with history embedding
                if self.attn_moe:

                    mixed_query_layer, gate_decision = self.qkv_proj(hidden_states, mode='q', **kwargs)

                    history_states = self.qkv_proj(history_states, mode='kv', gate_decision=gate_decision, **kwargs)[0]
                    mixed_key_layer, mixed_value_layer = history_states.chunk(2, dim=-1)

                else:
                    # query
                    _start = 0
                    _end = self.hidden_size
                    mixed_query_layer = F.linear(hidden_states,
                                                 self.qkv_proj.weight[_start:_end, :],
                                                 bias=None if self.qkv_proj.bias is None else self.qkv_proj.bias[_start:_end])

                    # key and value
                    # torch.equal(key, value)
                    _start = self.hidden_size
                    mixed_key_layer, mixed_value_layer = F.linear(history_states,
                                                                  self.qkv_proj.weight[_start:, :],
                                                                  bias=None if self.qkv_proj.bias is None else self.qkv_proj.bias[_start:]).chunk(
                                                                      2, dim=-1)


        else:
            raise NotImplementedError('please use unify qkv_proj')

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        if self.scale_multi_before:
            attention_scores = torch.matmul(query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
        else:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        if self.cfg.SOLVER.FORCE_SOFTMAX_FP16:
            with autocast(enabled=False):
                attention_probs = F.softmax(attention_scores.half(), dim=-1)
        else:
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        shape_list = list(range(len(context_layer.shape)))
        shape_list[-2], shape_list[-3] = shape_list[-3], shape_list[-2]
        context_layer = context_layer.permute(shape_list).contiguous()
        #context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)

        if self.attn_moe:
            context_layer, _ = self.dense(context_layer, gate_decision=gate_decision, **kwargs)
        else:
            context_layer = self.dense(context_layer)

        return context_layer, attention_probs
