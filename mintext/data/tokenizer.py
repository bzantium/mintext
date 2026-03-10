"""Minimal tokenizer wrapper for conversion tools.

Training data is pre-tokenized (mmap or arecord), so this is only
used by tools/text_to_arecord.py.
"""

from __future__ import annotations


class Tokenizer:
    """Wraps HuggingFace or SentencePiece tokenizer."""

    def __init__(self, path: str, type: str = "huggingface") -> None:
        if type == "huggingface":
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            self._vocab_size = self._tok.vocab_size
            self._eos_id = self._tok.eos_token_id or 0
            self._pad_id = self._tok.pad_token_id if self._tok.pad_token_id is not None else 0
        elif type == "sentencepiece":
            import sentencepiece as spm

            self._tok = spm.SentencePieceProcessor(model_file=path)
            self._vocab_size = self._tok.get_piece_size()
            self._eos_id = self._tok.eos_id()
            self._pad_id = self._tok.pad_id() if self._tok.pad_id() >= 0 else 0
        else:
            raise ValueError(f"Unknown tokenizer type: {type}")

        self._type = type

    def encode(self, text: str) -> list[int]:
        if self._type == "huggingface":
            return self._tok.encode(text, add_special_tokens=False)
        else:
            return self._tok.encode(text)

    def decode(self, ids: list[int]) -> str:
        if self._type == "huggingface":
            return self._tok.decode(ids, skip_special_tokens=False)
        else:
            return self._tok.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def pad_id(self) -> int:
        return self._pad_id
