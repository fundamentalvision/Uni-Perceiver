
import torch
from torch import nn

from uniperceiver.config import configurable
from uniperceiver.config import kfg
from .build import PREDICTOR_REGISTRY
import math
import torch.nn.functional as F

__all__ = ["BasePredictor", "RobertaLMHead","TwoLayerPredictor", "RobertaRegressionHead"]

@PREDICTOR_REGISTRY.register()
class BasePredictor(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,   # include <BOS>/<EOS>
        dropout: float
    ):
        super(BasePredictor, self).__init__()
        self.logits = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.DECODER_DIM,
            "vocab_size": cfg.MODEL.VOCAB_SIZE,
            "dropout": cfg.MODEL.PRED_DROPOUT
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        hidden_states = batched_inputs[kfg.G_HIDDEN_STATES]
        if isinstance(hidden_states, list):
            hidden_states = hidden_states[-1]        
        if self.dropout:  
            hidden_states = self.dropout(hidden_states)
        logits = self.logits(hidden_states)
        return { kfg.G_LOGITS: logits }
    
def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
        0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )


def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)

@PREDICTOR_REGISTRY.register()
class TwoLayerPredictor(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,   # include <BOS>/<EOS>
        dropout: float
    ):
        super(TwoLayerPredictor, self).__init__()
        
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation_fn = gelu
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.logits = nn.Linear(hidden_size, vocab_size)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
    
    def replace_logits(self, shared_weights):
        self.logits.weight = shared_weights

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.DECODER_DIM,
            "vocab_size": cfg.MODEL.VOCAB_SIZE,
            "dropout": cfg.MODEL.PRED_DROPOUT
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        hidden_states = batched_inputs[kfg.G_HIDDEN_STATES]
        if isinstance(hidden_states, list):
            hidden_states = hidden_states[-1]                 
        
        x = self.dense(hidden_states)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        
        logits = self.logits(x)
        return { kfg.G_LOGITS: logits }


@PREDICTOR_REGISTRY.register()
class RobertaLMHead(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,   # include <BOS>/<EOS>
        dropout: float,
        untie_weight_embedding: bool,
        use_bias: bool,
        share_hidden: bool,
    ):
        super(RobertaLMHead, self).__init__()
        
        
        self.activation_fn = gelu
        
        if untie_weight_embedding is True:
            self.weight = nn.Linear(hidden_size, vocab_size, bias=False).weight
        else:
            self.weight = None
            
        if share_hidden:
            self.dense = None
            self.layer_norm = None
        else:
            self.dense = nn.Linear(hidden_size, hidden_size)
            self.layer_norm = nn.LayerNorm(hidden_size)
        
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(vocab_size))
        else:
            self.bias = None
        
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        # print("dropout: {}".format(self.dropout))
    
    def replace_weight(self, weight):
        if self.weight is None:
            self.weight = weight
        else:
            print('already has weight, please set UNTIE_WEIGHT_EMBEDDING to False')
            
    def replace_module_hidden(self, dense, layer_norm):
        if (self.dense is None) and (self.layer_norm is None):
            self.dense = dense
            self.layer_norm = layer_norm
        else:
            print('already has hidden layers!')
            raise ValueError
        
            
    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.DECODER_DIM,
            "vocab_size": cfg.MODEL.VOCAB_SIZE,
            "dropout": cfg.MODEL.PRED_DROPOUT,
            "untie_weight_embedding": cfg.MODEL.UNTIE_WEIGHT_EMBEDDING,
            "use_bias": cfg.MODEL.USE_PREDICTOR_BIAS,
            "share_hidden": cfg.MODEL.SHARE_PREDICTOR_HIDDEN,
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        
        if kfg.G_HIDDEN_STATES in batched_inputs:
            hidden_states = batched_inputs[kfg.G_HIDDEN_STATES]
            if isinstance(hidden_states, list):
                hidden_states = hidden_states[-1]  
                               
            if kfg.G_TARGET_IDS in batched_inputs:
                mask_tokens = batched_inputs[kfg.G_TARGET_IDS].ne(-1) 
                hidden_states = hidden_states[mask_tokens]  
                batched_inputs[kfg.G_TARGET_IDS] = batched_inputs[kfg.G_TARGET_IDS][mask_tokens] 
            logits = self._forward(hidden_states)
            
            return { kfg.G_LOGITS: logits }

        elif kfg.U_HIDDEN_STATES in batched_inputs:
            hidden_states = batched_inputs[kfg.U_HIDDEN_STATES]
            if isinstance(hidden_states, list):
                hidden_states = hidden_states[-1]
            
            mask_tokens = batched_inputs[kfg.U_TARGET_IDS].ne(-1) 
            hidden_states = hidden_states[mask_tokens]  
            batched_inputs[kfg.U_TARGET_IDS] = batched_inputs[kfg.U_TARGET_IDS][mask_tokens]         
            logits = self._forward(hidden_states)
        
            return { kfg.U_LOGITS: logits }

    def _forward(self, x):
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        
        if self.dropout:  
            x = self.dropout(x)
        if self.bias is not None:
            logits = F.linear(x, self.weight) + self.bias
        else:
            logits = F.linear(x, self.weight)
        return logits

@PREDICTOR_REGISTRY.register()
class RobertaRegressionHead(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        hidden_size,
        feat_dim,
        transform,
        sigmoid
    ):
        super(RobertaRegressionHead, self).__init__()
        self.transform = transform
        self.decoder = nn.Linear(hidden_size, feat_dim)
        self.output_sigmoid = sigmoid
        
        
    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.DECODER_DIM,
            "feat_dim":  cfg.MODEL.LABELS_NUM,
            "sigmoid":  cfg.MODEL.SIGMOID,
            "transform": BertPooler(cfg)
        }

    @classmethod
    def add_config(cls, cfg):
        pass
    
    def test_forward(self, u_logits):
        # for Single stream similarity
        return { kfg.OUTPUT: u_logits }

    def forward(self, batched_inputs):
        ret = {}
        if kfg.G_HIDDEN_STATES in batched_inputs:
            hidden_states = batched_inputs[kfg.G_HIDDEN_STATES]
            if isinstance(hidden_states, list):
                hidden_states = hidden_states[-1]    
            hidden_states = self.transform(hidden_states)
            logits = self.decoder(hidden_states)   
            if self.output_sigmoid:
                logits = torch.sigmoid(logits)
            ret.update({ kfg.G_LOGITS: logits })          
            if not self.training:
                ret_test = self.test_forward(logits)
                ret.update(ret_test)
            return ret

        elif kfg.U_HIDDEN_STATES in batched_inputs:
            hidden_states = batched_inputs[kfg.U_HIDDEN_STATES]
            if isinstance(hidden_states, list):
                hidden_states = hidden_states[-1]    
            hidden_states = self.transform(hidden_states)
            logits = self.decoder(hidden_states)      
            if self.output_sigmoid:
                logits = torch.sigmoid(logits)       
            ret.update({ kfg.U_LOGITS: logits })          
            if not self.training:
                ret_test = self.test_forward(logits)
                ret.update(ret_test)
            return ret
