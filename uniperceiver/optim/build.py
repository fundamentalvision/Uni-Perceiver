import copy
import torch
import itertools
from enum import Enum
from uniperceiver.config import CfgNode
from uniperceiver.utils.registry import Registry
from uniperceiver.utils import comm

from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union

SOLVER_REGISTRY = Registry("SOLVER")
SOLVER_REGISTRY.__doc__ = """
Registry for SOLVER.
"""

_GradientClipperInput = Union[torch.Tensor, Iterable[torch.Tensor]]
_GradientClipper = Callable[[_GradientClipperInput], None]

def _create_gradient_clipper(cfg: CfgNode) -> _GradientClipper:
    def clip_grad_norm(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_norm_(p, cfg.SOLVER.GRAD_CLIP, cfg.SOLVER.NORM_TYPE)

    def clip_grad_value(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_value_(p, cfg.SOLVER.GRAD_CLIP)

    _GRADIENT_CLIP_TYPE_TO_CLIPPER = {
        'value': clip_grad_value,
        'norm': clip_grad_norm,
    }
    clipper = _GRADIENT_CLIP_TYPE_TO_CLIPPER[cfg.SOLVER.GRAD_CLIP_TYPE]
    if cfg.SOLVER.GRAD_CLIP_TYPE == 'value':
        return clipper, None
    else:
        return None, clipper


def get_default_optimizer_params(
    model: torch.nn.Module,
    base_lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    weight_decay_norm: Optional[float] = None,
    bias_lr_factor: Optional[float] = 1.0,
    weight_decay_bias: Optional[float] = None,
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
):
    if weight_decay_bias is None:
        weight_decay_bias = weight_decay
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()

    no_decay_list = {}
    if hasattr(model, 'no_weight_decay'):
        no_decay_list = model.no_weight_decay()

    for module_name, module in model.named_modules():
        no_decay = False
        if module_name in no_decay_list:
            no_decay = True
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            schedule_params = {
                "lr": base_lr,
                "weight_decay": weight_decay,
            }


            if isinstance(module, norm_module_types):
                schedule_params["weight_decay"] = weight_decay_norm
            elif module_param_name == "bias":
                # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                # hyperparameters are by default exactly the same as for regular
                # weights.
                schedule_params["lr"] = base_lr * bias_lr_factor
                schedule_params["weight_decay"] = weight_decay_bias

            if no_decay or (module_param_name in no_decay_list):
                schedule_params["weight_decay"] = 0.


            if overrides is not None and module_param_name in overrides:
                schedule_params.update(overrides[module_param_name])
            params += [
                {
                    "params": [value],
                    "lr": schedule_params["lr"],
                    "weight_decay": schedule_params["weight_decay"],
                }
            ]

    return params

def get_layer_id(module_name, num_layers):
    """
    Assign a parameter with its layer id
    modified from BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if module_name.split('.')[0] in [
            'video_embed', 'token_embed', 'prompt_embed', 'visual_embed', 'cls_token' ''
    ]:
        return 0
    elif module_name.startswith('encoder'):
        return int(module_name.split('.')[2]) + 1
    elif module_name.startswith('predictor'):
        return num_layers
    else:
        raise NotImplementedError('please check this layer')

def create_seperate_moe_param_groups(
    model,
    base_lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    weight_decay_norm: Optional[float] = None,
    bias_lr_factor: Optional[float] = 1.0,
    wg_lr_facetor: Optional[float] = 1.0,
    weight_decay_bias: Optional[float] = None,
    weight_decay_embedding: Optional[float] = None,
    weight_decay_wg: Optional[float] = None,
    cfg: dict = None,
):
    try:
        from deepspeed.moe.utils import is_moe_param
    except:
        def is_moe_param(param: torch.Tensor) -> bool:
            if hasattr(param, "allreduce") and not param.allreduce:
                return True
            return False

    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()

    num_layers = cfg.MODEL.BERT.NUM_HIDDEN_LAYERS + 1
    layer_decay = cfg.SOLVER.LAYER_LR_DECAY
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))


    if weight_decay_bias is None:
        weight_decay_bias = weight_decay
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )




    no_decay_list = {}
    if hasattr(model, 'no_weight_decay'):
        no_decay_list = model.no_weight_decay()

    wg_list = {}
    if hasattr(model, 'expert_gate_group'):
        wg_list = model.expert_gate_group()



    for module_name, module in model.named_modules():
        no_decay = False
        if module_name in no_decay_list:
            no_decay = True
        is_wg_param = False
        for wg_name in wg_list:
            if wg_name in module_name:
                is_wg_param = True
                continue

        for module_param_name, value in module.named_parameters(recurse=False):
            # layer_id = get_layer_id(module_name, num_layers)
            this_scale = layer_scales[ get_layer_id(module_name, num_layers)] if layer_decay < 1.0 else 1.0
            # if isinstance(module, torch.nn.Embedding):
            #     print(module_name, module_param_name)
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            schedule_params = {
                "lr": base_lr,
                "weight_decay": weight_decay,
                "moe": False,
            }
            if is_moe_param(value):
                schedule_params['moe'] = True

            if no_decay or (module_param_name in no_decay_list):
                schedule_params["weight_decay"] = 0.
            elif is_wg_param and isinstance(
                    module,
                    torch.nn.Linear) and module_param_name != "bias":
                # only add linear weights in gate function
                schedule_params["lr"] = base_lr * wg_lr_facetor
                schedule_params["weight_decay"] = weight_decay_wg

            elif isinstance(module, torch.nn.Embedding):
                schedule_params['weight_decay'] = weight_decay_embedding

            elif isinstance(module, norm_module_types):
                if not cfg.SOLVER.WEIGHT_DECAY_NORMBIAS_WEIGHT and module_param_name == "bias":
                    # ln bias use the same params as linear bias
                    schedule_params["lr"] = base_lr * bias_lr_factor
                    schedule_params['weight_decay'] = weight_decay_bias
                else:
                    schedule_params['weight_decay'] = weight_decay_norm

            elif module_param_name == "bias" or value.ndim == 1:
                schedule_params["lr"] = base_lr * bias_lr_factor
                schedule_params['weight_decay'] = weight_decay_bias

            params += [{
                "params": [value],
                "lr": max(schedule_params["lr"] * this_scale, cfg.LR_SCHEDULER.get('MIN_LR', 1e-6)),
                "moe": schedule_params['moe'],
                "weight_decay": schedule_params["weight_decay"],
                "name": f'{module_name}.{module_param_name}'
            }]



    return params


def create_group_moe_param_groups(
    model,
    base_lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    weight_decay_norm: Optional[float] = None,
    bias_lr_factor: Optional[float] = 1.0,
    wg_lr_facetor: Optional[float] = 1.0,
    weight_decay_bias: Optional[float] = None,
    weight_decay_embedding: Optional[float] = None,
    weight_decay_wg: Optional[float] = None,
    cfg: dict = None,
):
    from deepspeed.moe.utils import is_moe_param

    # params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()

    if weight_decay_bias is None:
        weight_decay_bias = weight_decay
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )

    group_params_dict = {}

    no_decay_list = {}
    if hasattr(model, 'no_weight_decay'):
        no_decay_list = model.no_weight_decay()

    wg_list = {}
    if hasattr(model, 'expert_gate_group'):
        wg_list = model.expert_gate_group()

    for module_name, module in model.named_modules():
        no_decay = False
        if module_name in no_decay_list:
            no_decay = True
        is_wg_param = False
        for wg_name in wg_list:
            if wg_name in module_name:
                is_wg_param = True
                continue

        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            # default setting
            lr_of_this_param = base_lr
            wd_of_this_param = weight_decay
            moe_of_this_param = False
            if is_moe_param(value):
                moe_of_this_param = True

            if no_decay or (module_param_name in no_decay_list):

                wd_of_this_param = 0.
            elif is_wg_param and isinstance(
                    module, torch.nn.Linear) and module_param_name != "bias":
                # only add linear weights in gate function
                lr_of_this_param = base_lr * wg_lr_facetor
                wd_of_this_param = weight_decay_wg

            elif isinstance(module, torch.nn.Embedding):
                wd_of_this_param = weight_decay_embedding

            elif isinstance(module, norm_module_types):
                if not cfg.SOLVER.WEIGHT_DECAY_NORMBIAS_WEIGHT and module_param_name == "bias":
                    # ln bias uses the same params as linear bias
                    lr_of_this_param = base_lr * bias_lr_factor
                    wd_of_this_param = weight_decay_bias
                else:
                    wd_of_this_param = weight_decay_norm

            elif module_param_name == "bias":
                lr_of_this_param = base_lr * bias_lr_factor
                wd_of_this_param = weight_decay_bias

            param_group_name = f'lr_{lr_of_this_param}_wd_{wd_of_this_param}_moe_{moe_of_this_param}'
            if param_group_name not in group_params_dict:
                group_params_dict[param_group_name] = {
                    'params': [],
                    "lr": lr_of_this_param,
                    "weight_decay": wd_of_this_param,
                    'moe': moe_of_this_param,
                    'name': param_group_name,
                    'params_name': [],
                }
            group_params_dict[param_group_name]['params'].append(value)
            group_params_dict[param_group_name]['params_name'].append(
                f'{module_name}.{module_param_name}')


    valid_params_groups =  list(group_params_dict.values())
    return valid_params_groups




def create_moe_param_groups(
    model,
    base_lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    weight_decay_norm: Optional[float] = None,
    bias_lr_factor: Optional[float] = 1.0,
    wg_lr_facetor: Optional[float] = 1.0,
    weight_decay_bias: Optional[float] = None,
    weight_decay_embedding: Optional[float] = None,
    weight_decay_wg: Optional[float] = None,

):
    from deepspeed.moe.utils import is_moe_param

    '''
    name: 
    '''
    if weight_decay_bias is None:
        weight_decay_bias = weight_decay
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )

    if weight_decay_embedding == 0.0:
        norm_module_types = norm_module_types + (torch.nn.Embedding, )
    else:
        # if weight_decay_embedding is not 0.0, we set its weight_decay as normal weights
        # assert weight_decay_embedding == weight_decay
        pass



    params_with_weight_decay = {
        'params': [],
        'name': 'weight_decay_params',
        'params_name': [],
    }
    params_without_weight_decay = {
        'params': [],
        "weight_decay": 0.0,
        'name': 'without_weight_decay_params',
        'params_name': [],
    }
    bias_params = {
        'params': [],
        "lr": base_lr * bias_lr_factor,
        "weight_decay": weight_decay_bias,
        'name': 'bias_params',
        'params_name': [],
    }
    wg_params = {
        'params': [],
        "lr": base_lr * wg_lr_facetor,
        "weight_decay": weight_decay_wg,
        'name': 'wg_params',
        'params_name': [],
    }
    norm_params = {
        'params': [],
        "weight_decay": weight_decay_norm,
        'name': 'norm_params',
        'params_name': [],
    }
    moe_params_with_weight_decay = {
        'params': [],
        'moe': True,
        'name': 'weight_decay_moe_params',
        'params_name': [],
    }
    moe_params_without_weight_decay = {
        'params': [],
        "weight_decay": 0.0,
        'moe': True,
        'name': 'without_weight_decay_moe_params',
        'params_name': [],
    }
    moe_bias_params = {
        'params': [],
        "lr": base_lr * bias_lr_factor,
        "weight_decay": weight_decay_bias,
        'moe': True,
        'name': 'bias_moe_params',
        'params_name': [],
    }
    moe_norm_params = {
        'params': [],
        "weight_decay": weight_decay_norm,
        'moe': True,
        'name': 'norm_moe_params',
        'params_name': [],
    }

    params_groups = [
        params_with_weight_decay, params_without_weight_decay, norm_params, bias_params, wg_params, \
        moe_params_with_weight_decay, moe_params_without_weight_decay, moe_norm_params, moe_bias_params
    ]



    no_decay_list = {}
    if hasattr(model, 'no_weight_decay'):
        no_decay_list = model.no_weight_decay()

    wg_list = {}
    if hasattr(model, 'expert_gate_group'):
        wg_list = model.expert_gate_group()

    memo: Set[torch.nn.parameter.Parameter] = set()

    for module_name, module in model.named_modules():
        no_decay = False
        if module_name in no_decay_list:
            no_decay = True
        is_wg_param = False
        for wg_name in wg_list:
            if wg_name in module_name:
                is_wg_param = True
                continue

        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            if is_moe_param(value):
                if no_decay or (module_param_name in no_decay_list):
                    moe_params_without_weight_decay['params'].append(value)
                elif isinstance(module, norm_module_types):
                    moe_norm_params['params'].append(value)
                elif module_param_name == "bias":
                    moe_bias_params['params'].append(value)
                else:
                    moe_params_with_weight_decay['params'].append(value)
            else:
                if no_decay or (module_param_name in no_decay_list):
                    params_without_weight_decay['params'].append(value)
                    params_without_weight_decay['params_name'].append(f'{module_name}.{module_param_name}')
                elif is_wg_param and isinstance(module, torch.nn.Linear) and module_param_name != "bias":
                    # only add linear weights in gate function
                    wg_params['params'].append(value)
                    wg_params['params_name'].append(
                        f'{module_name}.{module_param_name}')
                elif isinstance(module, norm_module_types):
                    norm_params['params'].append(value)
                    norm_params['params_name'].append(
                        f'{module_name}.{module_param_name}')
                elif module_param_name == "bias":
                    bias_params['params'].append(value)
                    bias_params['params_name'].append(
                        f'{module_name}.{module_param_name}')
                else:
                    params_with_weight_decay['params'].append(value)
                    params_with_weight_decay['params_name'].append(
                        f'{module_name}.{module_param_name}')

    valid_params_groups = [
        group for group in params_groups if len(group['params']) > 0
    ]

    return  valid_params_groups






def _generate_optimizer_class_with_gradient_clipping(
    optimizer: Type[torch.optim.Optimizer],
    *,
    per_param_clipper: Optional[_GradientClipper] = None,
    global_clipper: Optional[_GradientClipper] = None,
) -> Type[torch.optim.Optimizer]:
    """
    Dynamically creates a new type that inherits the type of a given instance
    and overrides the `step` method to add gradient clipping
    """
    assert (
        per_param_clipper is None or global_clipper is None
    ), "Not allowed to use both per-parameter clipping and global clipping"

    def optimizer_wgc_step(self, closure=None):
        if per_param_clipper is not None:
            for group in self.param_groups:
                for p in group["params"]:
                    per_param_clipper(p)
        else:
            # global clipper for future use with detr
            # (https://github.com/facebookresearch/detr/pull/287)
            all_params = itertools.chain(*[g["params"] for g in self.param_groups])
            norm_before_clip = global_clipper(all_params)

        super(type(self), self).step(closure)

    OptimizerWithGradientClip = type(
        optimizer.__name__ + "WithGradientClip",
        (optimizer,),
        {"step": optimizer_wgc_step},
    )
    return OptimizerWithGradientClip

def maybe_add_gradient_clipping(
    cfg: CfgNode, optimizer: Type[torch.optim.Optimizer]
) -> Type[torch.optim.Optimizer]:
    """
    If gradient clipping is enabled through config options, wraps the existing
    optimizer type to become a new dynamically created class OptimizerWithGradientClip
    that inherits the given optimizer and overrides the `step` method to
    include gradient clipping.

    Args:
        cfg: CfgNode, configuration options
        optimizer: type. A subclass of torch.optim.Optimizer

    Return:
        type: either the input `optimizer` (if gradient clipping is disabled), or
            a subclass of it with gradient clipping included in the `step` method.
    """
    if cfg.SOLVER.GRAD_CLIP <= 0:
        return optimizer
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer_type = type(optimizer)
    else:
        assert issubclass(optimizer, torch.optim.Optimizer), optimizer
        optimizer_type = optimizer

    per_param_clipper, global_clipper = _create_gradient_clipper(cfg)
    OptimizerWithGradientClip = _generate_optimizer_class_with_gradient_clipping(
        optimizer_type, per_param_clipper=per_param_clipper, global_clipper=global_clipper
    )
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer.__class__ = OptimizerWithGradientClip  # a bit hacky, not recommended
        return optimizer
    else:
        return OptimizerWithGradientClip

def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    # params = get_default_optimizer_params(
    #     model,
    #     base_lr=cfg.SOLVER.BASE_LR,
    #     weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    #     weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
    #     bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
    #     weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
    # )
    params = create_seperate_moe_param_groups(
                    model,
                    base_lr=cfg.SOLVER.BASE_LR,
                    weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                    weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
                    bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
                    wg_lr_facetor=cfg.SOLVER.WG_LR_FACTOR,
                    weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
                    weight_decay_embedding=cfg.SOLVER.WEIGHT_DECAY_EMBEDDING,
                    weight_decay_wg=cfg.SOLVER.WEIGHT_DECAY_WG,
                    cfg=cfg,
                )
    if cfg.SOLVER.NAME == 'LAMB':
        from uniperceiver.optim import LAMB
        optimizer = LAMB(
            params,
            lr=cfg.SOLVER.BASE_LR,
            betas=cfg.SOLVER.BETAS,
            eps=cfg.SOLVER.EPS,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY, )

    else:
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.SOLVER.BASE_LR,
            betas=cfg.SOLVER.BETAS,
            eps=cfg.SOLVER.EPS,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY, 
        )
    # optimizer = SOLVER_REGISTRY.get(cfg.SOLVER.NAME)
    # return maybe_add_gradient_clipping(cfg, optimizer)(cfg, params)
    return optimizer
