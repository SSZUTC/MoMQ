from sentencepiece import SentencePieceProcessor
from logging import getLogger
from typing import List
import os

logger = getLogger()


class SentencePieceTokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.Encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.Decode(t)

    @property
    def vocab_size(self):
        return self.sp_model.vocab_size()

    @property
    def vocab(self):
        return NotImplementedError

    @property
    def inv_vocab(self):
        raise NotImplementedError

    def tokenize(self, text, bos=False, eos=False):
        ids = self.encode(text, bos, eos)
        # ids = self.encode(text, bos=False, eos=False)
        return ids

    def detokenize(self, token_ids):
        return self.sp_model.Decode(token_ids)

    @property
    def eod(self):
        return self.eos_id

    def tokenize_with_special_tokens(self, text: str):
        ids = self.encode(text, bos=False, eos=False)
        return ids
