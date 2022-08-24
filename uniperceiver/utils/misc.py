import torch
import torch.distributed as dist
from torch._six import inf
import io
from timm.utils import get_state_dict
try:
    from apex import amp
    APEX_INSTALLED = True
except:
    print('apex has not been installed.')
    APEX_INSTALLED = False


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, enabled=True, growth_interval=500, init_scale=2.**13):
        self.enabled = enabled
        self._scaler = torch.cuda.amp.GradScaler(init_scale=init_scale, growth_interval=growth_interval, enabled=self.enabled)

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True,
        fp16=False, iter=0, min_loss_scale= 2048.0, loss_scale_window=200):
        self._scaler.scale(loss).backward(create_graph=create_graph)

        if fp16:
            # used for stable training
            if iter > 5000 and self._scaler.get_scale() < min_loss_scale:
                min_growth_interval = 5
                if self._scaler.get_growth_interval() != min_growth_interval:
                    self._scaler.set_growth_interval(min_growth_interval)

            elif iter > 5000 and self._scaler.get_growth_interval() == 5:
                self._scaler.set_growth_interval(loss_scale_window)

        if update_grad:
            if clip_grad is not None and clip_grad > 0.0:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
        else:
            norm = None
        return norm

    def step(self, optimizer):
        self._scaler.step(optimizer)

    def update(self):
        self._scaler.update()

    def get_scale(self):
        return self._scaler.get_scale()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


class ApexScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._scaler = amp

    def __call__(self,
                 loss,
                 optimizer,
                 clip_grad=None,
                 parameters=None,
                 create_graph=False,
                 update_grad=True,
                 fp16=False,
                 iter=0,
                 min_loss_scale=2048.0,
                 loss_scale_window=200):

        with self._scaler.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        if update_grad:
            if clip_grad is not None and clip_grad > 0.0:
                norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), clip_grad)
            else:

                norm = get_grad_norm_(amp.master_params(optimizer))
        else:
            norm = None
        return norm

    def step(self, optimizer):
        optimizer.step()


    def update(self):
        pass

    def get_scale(self):
        return self._scaler.state_dict()['loss_scaler0']['loss_scale']

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm