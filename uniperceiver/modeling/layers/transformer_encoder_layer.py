import torch
from torch import nn
import torch.nn.functional as F

from typing import Optional, Any, Union, Callable
from torch import Tensor

from .create_act import get_act_layer, get_activation
from timm.models.layers import DropPath
from .layer_norm import LayerNorm
from .pe_encoder import DeepPrompt

class TransformerEncoderLayer(nn.Module):
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
                 device=None, dtype=None, cfg: dict = None) -> None:
        #
        factory_kwargs = {}
        super(TransformerEncoderLayer, self).__init__()

        self.cfg = cfg

        # The interface of nn.MultiheadAttention changed since torch 1.9.0.
        _torch_version_main = torch.__version__.split('.')[:2]
        if (int(_torch_version_main[0]) >= 1) and (int(_torch_version_main[1])) >= 9:
            self._torch_nn_new_interface = True
        else:
            self._torch_nn_new_interface = False

        if self._torch_nn_new_interface:
            factory_kwargs = {'device': device, 'dtype': dtype}
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                **factory_kwargs)
        else:
            factory_kwargs = {}
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                **factory_kwargs)

        self.batch_first = batch_first

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        if self.cfg.SOLVER.FUSED_LAYERNORM:
            self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
            self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
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
        super(TransformerEncoderLayer, self).__setstate__(state)

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

        if self.batch_first and not self._torch_nn_new_interface:
            x = src.transpose(0,1)
            if history_states is not None:
                history_states = history_states.transpose(0,1)
        else:
            x = src

        if self.norm_first:
            history_states_norm = history_states if (history_states is None) else self.norm1(history_states)
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, history_states=history_states_norm, **kwargs)
            x = x + self._ff_block(self.norm2(x), **kwargs)
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, history_states=history_states, **kwargs))
            x = self.norm2(x + self._ff_block(x), **kwargs)

        if self.batch_first and not self._torch_nn_new_interface:
            x = x.transpose(0, 1)

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], history_states: Optional[Tensor],
                  **kwargs) -> Tensor:

        if history_states is not None:
            kv = torch.cat(
                [history_states, x],
                dim=1 if (self.batch_first and self._torch_nn_new_interface) else 0
            )
            # TODO: changes for attn_mask and key_padding_mask
        else:
            kv = x

        if self.deep_prompt:

            deep_prompt_embedding = self.deep_prompt_embedding(x, batch_first=(self.batch_first and self._torch_nn_new_interface), **kwargs)
            if self.norm_first:
                deep_prompt_embedding = self.norm1(deep_prompt_embedding)
            kv = torch.cat([deep_prompt_embedding, kv], dim=1 if (self.batch_first and self._torch_nn_new_interface) else 0)
            if attn_mask is not None:
                L, S = attn_mask.shape
                pe_length = deep_prompt_embedding.shape[1 if
                                                        (self.batch_first and self._torch_nn_new_interface) else 0]  # length, bs, hidden_size
                attn_mask = torch.cat([torch.zeros((L, pe_length), dtype=attn_mask.dtype, device=attn_mask.device), attn_mask], dim=1)
            if key_padding_mask is not None:
                if self.batch_first and self._torch_nn_new_interface:
                    bs, pe_length = deep_prompt_embedding.shape[:2]
                else:
                    pe_length, bs = deep_prompt_embedding.shape[:2]
                key_padding_mask = torch.cat(
                    [torch.zeros((bs, pe_length), dtype=key_padding_mask.dtype, device=key_padding_mask.device), key_padding_mask], dim=1)


        x = self.self_attn(x, kv, kv,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        x = self.drop_path1(self.dropout1(x))
        if self.layer_scale:
            x = self.gamma_1 * x
        return x


    # feed forward block
    def _ff_block(self, x: Tensor, **kwargs) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.drop_path2(self.dropout2(x))
        if self.layer_scale:
            x = self.gamma_2 * x
        return x
