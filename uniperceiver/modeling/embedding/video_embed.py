import torch
from torch import nn

from uniperceiver.config import configurable
from ..layers.create_act import get_act_layer
from .build import EMBEDDING_REGISTRY
from .position_embedding import NNEmbeddingEncoding
from einops import rearrange, repeat
from uniperceiver.modeling.layers import FP16LayerNorm


__all__ = ["VideoBaseEmbedding"]

@EMBEDDING_REGISTRY.register()
class VideoBaseEmbedding(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        cfg: dict,
        in_dim: int,
        out_dim: int,
        patch_size: int,
        time_span: int,
        max_time_len: int,
        max_spatial_size = 196,
        **kwargs
    ):
        super(VideoBaseEmbedding, self).__init__()
        self.cfg = cfg
        self.embeddings = nn.Linear(in_dim, out_dim)
        self.embeddings_act = kwargs.pop("embeddings_act", None)
        self.embeddings_norm = kwargs.pop("embeddings_norm", None)
        self.embeddings_dropout = kwargs.pop("embeddings_dropout", None)
        self.embeddings_pos = kwargs.pop('embeddings_pos', None)
        self.embeddings_type = kwargs.pop("embeddings_token_type", None)
        self.random_temporal_pos = kwargs.pop("random_temporal_pos", True)
        self.patch_size = patch_size
        self.time_span = time_span
        self.pos_before = kwargs.pop('pos_before', True)
        self.add_type_embedding = cfg.MODEL.VIDEO_EMBED.ADD_TYPE_EMBED
        if self.add_type_embedding:
            assert self.embeddings_type is not None

        self.embeddings_st_pos = None
        self.max_spatial_size = max_spatial_size
        if isinstance(self.embeddings_pos, str):
            if self.embeddings_pos == 'divide_st_pos':
                self.embeddings_st_pos = Divide_ST_POS(
                    max_spatial_size, max_time_len, out_dim,
                    self.random_temporal_pos)
                self.embeddings_pos = None
                del self.embeddings
                self.embeddings = nn.Conv2d(in_dim//(self.patch_size**2), out_dim, kernel_size=self.patch_size, stride=self.patch_size)
        pass

    def replace_weight(self, visual_embed):
        if visual_embed is not None:
            del self.embeddings
            self.embeddings = visual_embed.patch_embed.proj

    def share_spatial_pos(self, visual_embed):
        if self.embeddings_st_pos is not None and visual_embed is not None:

            if self.embeddings_st_pos.spatial_pos_embed.weight.shape[0] == visual_embed.patch_embed.pos_embed.weight.shape[0]:
                self.embeddings_st_pos.spatial_pos_embed_index = 0
            else:
                # cls token in image patch tokenizer
                self.embeddings_st_pos.spatial_pos_embed_index = 1
            del self.embeddings_st_pos.spatial_pos_embed
            self.embeddings_st_pos.spatial_pos_embed = visual_embed.patch_embed.pos_embed
            pass

    @classmethod
    def from_config(cls, cfg):
        kwargs = {
            "in_dim": cfg.MODEL.VIDEO_EMBED.IN_DIM,
            "out_dim": cfg.MODEL.VIDEO_EMBED.OUT_DIM,
            "patch_size": cfg.MODEL.PATCH_SIZE,
            "time_span": cfg.MODEL.VIDEO_EMBED.PATCH_SIZE_T,
            "max_time_len": cfg.MODEL.VIDEO_EMBED.MAX_FRAMES,
        }
        max_spatial_size = int(cfg.MODEL.IMG_INPUT_SIZE/cfg.MODEL.PATCH_SIZE)**2
        kwargs['max_spatial_size'] = max_spatial_size
        activation_name = (cfg.MODEL.VIDEO_EMBED.ACTIVATION).lower()
        if activation_name != "none":
            activation = get_act_layer(activation_name)
            assert activation is not None

            act_kwargs = {}
            if activation_name in { "elu", "celu" }:
                act_kwargs["alpha"] = cfg.MODEL.VIDEO_EMBED.ELU_ALPHA
            embeddings_act = activation(**act_kwargs)
            kwargs['embeddings_act'] = embeddings_act

        if cfg.MODEL.VIDEO_EMBED.DROPOUT > 0:
            embeddings_dropout = nn.Dropout(cfg.MODEL.VIDEO_EMBED.DROPOUT)
            kwargs['embeddings_dropout'] = embeddings_dropout

        if cfg.MODEL.VIDEO_EMBED.USE_NORM:
            if cfg.SOLVER.FORCE_LN_FP16:
                embeddings_norm = FP16LayerNorm(cfg.MODEL.VIDEO_EMBED.OUT_DIM)
            else:
                embeddings_norm = nn.LayerNorm(cfg.MODEL.VIDEO_EMBED.OUT_DIM)
            kwargs['embeddings_norm'] = embeddings_norm

        if cfg.MODEL.VIDEO_EMBED.DIVIDE_ST_POS:
            kwargs['embeddings_pos'] = "divide_st_pos"

        elif cfg.MODEL.VIDEO_EMBED.POSITION.lower() != 'none':
            embeddings_pos = NNEmbeddingEncoding(cfg.MODEL.VIDEO_EMBED.OUT_DIM, cfg.MODEL.VIDEO_EMBED.MAX_LENGTH)
            kwargs['embeddings_pos'] = embeddings_pos

        if cfg.MODEL.VIDEO_EMBED.TYPE_SIZE > 0:
            embeddings_token_type = nn.Embedding(
            cfg.MODEL.VIDEO_EMBED.TYPE_SIZE, cfg.MODEL.VIDEO_EMBED.OUT_DIM)
            kwargs['embeddings_token_type'] = embeddings_token_type
        kwargs['random_temporal_pos'] = cfg.MODEL.VIDEO_EMBED.POS_RANDOM
        kwargs['pos_before'] = cfg.MODEL.POS_BEFORE
        kwargs['cfg'] = cfg
        return kwargs

    def forward(self, data, **kwargs):

        if data.dim() == 4:
            #images
            data = data.unsqueeze(1)  # BS, 3, 224, 224


        if self.embeddings_st_pos is not None:
            bs = data.size(0)
            x = self.embeddings(data.flatten(0, 1)) # b*t, dim, 14, 14
            x = x.flatten(2) # .flatten(2)
            embeddings = rearrange(x, '(b t s) c hw -> b t hw (s c)', b=bs,  s = self.time_span)
            embeddings_pos = self.embeddings_st_pos(embeddings).unsqueeze(
                0).flatten(1, 2)
            embeddings = embeddings.flatten(1, 2)
            if self.pos_before:
                embeddings = embeddings + embeddings_pos.to(embeddings.dtype)


        if self.embeddings_pos is not None:
            x = rearrange(data, 'b (t s) c (h p1) (w p2) -> b (t h w) (s c p1 p2)', s = self.time_span, p1 = self.patch_size, p2 = self.patch_size)
            embeddings = self.embeddings(x)
            embeddings_pos = self.embeddings_pos(x).unsqueeze(0)
            if self.pos_before:
                embeddings = embeddings + embeddings_pos.to(embeddings.dtype)

        if self.add_type_embedding:
            embeddings = embeddings + self.embeddings_type.weight[0].unsqueeze(0).unsqueeze(1).to(embeddings.dtype)

        if self.embeddings_act is not None:
            embeddings = self.embeddings_act(embeddings)

        if self.embeddings_norm is not None:
            embeddings = self.embeddings_norm(embeddings)

        if not self.pos_before:
            embeddings = embeddings + embeddings_pos

        if self.embeddings_dropout is not None:
            embeddings = self.embeddings_dropout(embeddings)

        return embeddings



class Divide_ST_POS(nn.Module):
    def __init__(self, num_patches, max_time_len, out_dim,
                 random_temporal_pos):
        super(Divide_ST_POS, self).__init__()
        self.spatial_pos_embed = nn.Embedding(num_patches, out_dim)
        self.temporal_pos_embed = nn.Embedding(max_time_len, out_dim)
        self.spatial_pos_embed_index = 0 # sometimes image has cls_token
        self.max_frames = max_time_len
        self.random_temporal_pos = random_temporal_pos

    def forward(self, x):
        dtype = x.dtype
        temp_len, spatial_size = x.size(1), x.size(2)

        if self.training and self.random_temporal_pos:
            temporal_pos_ids = torch.arange(temp_len, dtype=torch.long, device=x.device) + \
                torch.randint(0, self.max_frames - temp_len + 1, size=(1,), dtype=torch.long, device=x.device)
        else:
            temporal_pos_ids = torch.arange(temp_len, dtype=torch.long, device=x.device)

        pos_embed = self.temporal_pos_embed(temporal_pos_ids).unsqueeze(1).to(dtype=dtype) + \
            self.spatial_pos_embed( torch.arange(start= self.spatial_pos_embed_index, end=spatial_size +  self.spatial_pos_embed_index , dtype=torch.long, device=x.device)).unsqueeze(0).to(dtype=dtype)
        return pos_embed
