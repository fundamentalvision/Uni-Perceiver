
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from uniperceiver.config import configurable
from uniperceiver.functional import load_vocab, decode_sequence, decode_sequence_bert
# from uniperceiver.tokenization import BertTokenizer
from uniperceiver.tokenization import ClipTokenizer

class DecodeStrategy(nn.Module, metaclass=ABCMeta):
    @configurable
    def __init__(
        self,
        *,
        vocab_path,
        vocab_name,
        beam_size,
        max_seq_len,
        tokenizer,
        bos_token_id,
        eos_token_id,
        spe_token_id = None,
        fp16=False,
        cfg=None,
    ):
        super().__init__()
        self.beam_size = beam_size
        if tokenizer is None:
            self.vocab = load_vocab(vocab_path)
        else:
            self.vocab = None

        if len(vocab_name) > 1:
            raise NotImplementedError("Only support caption inference on a single vocabulary!")
        else:
            self.vocab_name = vocab_name[0]
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.spe_token_id = spe_token_id
        self.fp16 = fp16
        self.cfg = cfg
        self.len_penalty = self.cfg.DECODE_STRATEGY.get('LEN_PENALTY',  0.0) # do not normalize
        pass

    @classmethod
    def from_config(cls, cfg):
        tokenizer_map = {
            # 'BERT': BertTokenizer,
            'CLIP': ClipTokenizer,
            }

        tokenizer_cls = tokenizer_map.get(cfg.INFERENCE.VOCAB, None)
        spe_token_id = None
        if tokenizer_cls is None:
            tokenizer = None
            bos_token_id = 0
            eos_token_id = 0
        elif cfg.INFERENCE.VOCAB == 'CLIP':
            tokenizer = tokenizer_cls()
            bos_token_id = tokenizer.vocab['<|startoftext|>']
            eos_token_id = tokenizer.vocab['<|endoftext|>']
            spe_token_id = tokenizer.vocab['<|spe|>']
        elif cfg.INFERENCE.VOCAB == 'CLIP_CAPTION':
            tokenizer = tokenizer_cls()
            bos_token_id = tokenizer.vocab['<|gen|>']
            eos_token_id = tokenizer.vocab['<|endoftext|>']
        else:
            tokenizer = tokenizer_cls.from_pretrained(cfg.MODEL.PRETRAINING.MODEL_NAME, do_lower_case=cfg.MODEL.PRETRAINING.DO_LOWER_CASE)
            if cfg.INFERENCE.VOCAB == 'BERT':
                raise NotImplementedError
                bos_token_id = tokenizer.vocab["[CLS]"]
                eos_token_id = tokenizer.vocab["[SEP]"]

        return {
            "vocab_path": cfg.INFERENCE.VOCAB,
            "vocab_name": cfg.DATASETS.TARGET_SET,
            "beam_size": cfg.DECODE_STRATEGY.BEAM_SIZE,
            "max_seq_len": cfg.MODEL.EVAL_MAX_SEQ_LEN if 'EVAL_MAX_SEQ_LEN' in cfg.MODEL else cfg.MODEL.MAX_SEQ_LEN,
            'tokenizer': tokenizer,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "spe_token_id": spe_token_id,
            "cfg": cfg,
            # "fp16": cfg.SOLVER.AMP_FP16,
        }

    @abstractmethod
    def _forward(self, batched_inputs, model):
        pass

    def forward(self, batched_inputs, output_sents, model):
        ret = self._forward(batched_inputs, model)
        if output_sents:
            if self.vocab:
                sents = decode_sequence(self.vocab, ret["G_SENTS_IDS"])
            else:
                sents = decode_sequence_bert(self.tokenizer, ret["G_SENTS_IDS"], self.eos_token_id)
            ret.update({ "output": sents })
        return ret