import torch


class DeepPrompt(torch.nn.Module):
    # naive implementation
    def __init__(self, cfg):
        super().__init__()

        embedding_hidden_size = cfg.MODEL.BERT.HIDDEN_SIZE
        self.target_prompt = cfg.MODEL.PROMPT_EMBED.TARGET_DEEP_PROMPT and not cfg.MODEL.PROMPT_EMBED.SHARE_DEEP_PROMPT
        self.embedding = torch.nn.Embedding(cfg.MODEL.PROMPT_EMBED.INPUT_DEEP_PROMPT_LENGTH, embedding_hidden_size)
        if self.target_prompt:
            self.target_embedding = torch.nn.Embedding(cfg.MODEL.PROMPT_EMBED.TARGET_DEEP_PROMPT_LENGTH, embedding_hidden_size)


    def forward(self, x, batch_first=False, data_type=None, **kwargs):
        # x: length, bs, hidden_size

        if data_type == 'target' and self.target_prompt:
            embddings = self.target_embedding.weight
        else:
            embddings = self.embedding.weight

        if batch_first:
            bs = x.shape[0]
            embddings = embddings.unsqueeze(0).expand(bs, -1, -1)
        else:
            bs = x.shape[1]
            embddings = embddings.unsqueeze(1).expand(-1,bs, -1)
        return embddings
