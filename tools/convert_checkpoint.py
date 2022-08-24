#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[2]:
# ckpt_path = '/mnt/cache/zhujinguo/codes/UniPerceiver/work_dirs/deepspeed_moe/BERT_L12_H768_experiments/16task_90k_bertbase_lr1e-3_wd0.2_gc0.1_prenorm_warm10k_layerscale1e-3_uniformdp0.1_maeinit_fixedpos_torchfp16_unifieddataset_changeweight_stage2_224size/bertbase_womoe_pretrain2/89999/mp_rank_00_model_states.pt'
# save_path = '/mnt/lustre/zhujinguo/codes/Uni-Perceiver/work_dirs/pretrained_models/uni-perceiver-base-L12-H768-224size-pretrained.pth'

ckpt_path = '/mnt/cache/zhujinguo/codes/UniPerceiver/work_dirs/deepspeed_moe/BERT_L24_H1024_experiments/16task_90k_bertlarge_lr2e-5_wd0.05_gc0.1_prenorm_warm5k_layerscale1e-3_uniformdp0.2_maeinit_fixedpos_torchfp16_unifieddataset_pretrain_stage2_224size_bw128_all0.5_accum2_bwv2_k700_8frames_yfccfixcap_womixup/all0.5_rmmixup_from430/89999/mp_rank_00_model_states.pt'
save_path = '/mnt/lustre/zhujinguo/codes/Uni-Perceiver/work_dirs/pretrained_models/uni-perceiver-large-L24-H1024-224size-pretrained.pth'
origin_checkpoint_path = ckpt_path 


# In[3]:


origin_checkpoint = torch.load(origin_checkpoint_path, 'cpu')
origin_checkpoint.keys()
# list(origin_checkpoint['module'].keys())


# In[4]:


len(list(origin_checkpoint['module'].keys()))


# In[8]:


# new_checkpoint_path = 'new_exp/model_Epoch_00160_Iter_0000159.pth'
# new_checkpoint = torch.load(new_checkpoint_path, 'cpu')
# new_checkpoint.keys()
# list(new_checkpoint['model'].keys())


# In[10]:


# len(list(new_checkpoint['model'].keys()))


# In[5]:


mapping_dict = {

    'encoder.': 'fused_encoder.',
    'attention.self.qkv_proj.weight': 'self_attn.in_proj_weight',
    'attention.self.qkv_proj.bias': 'self_attn.in_proj_bias',
    'attention.output.dense': 'self_attn.out_proj',
    'attention_output.residual_scale': 'gamma_1',
    'ffn.dense.': 'linear1.',
    'ffn.dense2.': 'linear2.',
    'ffn_output.residual_scale': 'gamma_2',
    'LayerNormModules.0.': 'norm1.',
    'LayerNormModules.1.': 'norm2.',
    'predictor.': 'loss_prepare.',
    
}


# In[6]:


new_checkpoint = { } 

module_checkpoint = origin_checkpoint['module']

for k, v in module_checkpoint.items():
    if k.endswith('residual_scale'):
        v.squeeze_(1).squeeze_(0)
    if k.startswith('visual_embed'):
        continue
    for origin_str, target_str in mapping_dict.items():
        if origin_str in k:
            k = k.replace(origin_str, target_str)
    
    new_checkpoint[k] = v.float()

# merge type embedding in video_embed 
new_checkpoint['video_embed.embeddings.bias'] = new_checkpoint['video_embed.embeddings.bias'] + new_checkpoint['video_embed.embeddings_type.weight'][0]

# In[7]:



torch.save({ 'model': new_checkpoint}, save_path)

