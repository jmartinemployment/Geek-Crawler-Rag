"""How the chunker measures text, and why it never decodes to produce it.

The chunker used tiktoken's ``cl100k_base`` while the embedder counts WordPiece. That is not a
rounding difference: WordPiece runs far longer on code and technical identifiers, so a parent
inside its configured 480-token budget arrived at inference over the model's 512-token ceiling and
was silently truncated. Measured on the live corpus 2026-09-30: 136 of 1,984 parent points (6.9%)
crossed the ceiling, worst case 638 tokens, and the discarded tail was Python source.

**The fix is not to swap the tokenizer under the existing code.** The old ``chunk_text`` produced
each chunk by decoding a token slice, which is lossless for BPE and destructive for WordPiece --
bge-small's tokenizer is uncased and splits on punctuation, so::

    invoice = Invoice.model_validate_json(raw_json)

round-trips through decode as::

    invoice = invoice. model _ validate _ json ( raw _ json )

Lowercased, spaces injected. Chunking that way would corrupt every chunk in the corpus and, because
chunk text is also the citation-verification target, would break quote verification wholesale.

So this module exposes **character spans, not tokens**. The chunker measures in the model's own
tokens and slices the *original string* at token boundaries, which is exact in both directions.

**Truncation must be off here and on at inference.** Encoding a 2,000-token section with the
embedder's tokenizer returns 512 tokens and 512 spans, so the section would look short enough to
need no windowing and would be silently cut instead. ``WordPieceChunkTokenizer`` therefore takes a
*serialised copy* of the tokenizer and disables truncation on the copy; the embedder's own
tokenizer is left untouched, and ``LocalDenseEmbedding._count_tokens`` keeps relying on truncation
being on.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tokenizers import Tokenizer


@runtime_checkable
class ChunkTokenizer(Protocol):
    """Measures text in the embedding model's own tokens.

    Implementations return spans rather than tokens so callers can only ever slice the original
    text. There is deliberately no ``decode`` on this interface: an interface that offers one invites
    the destructive round-trip this module exists to prevent.
    """

    @property
    def sequence_limit(self) -> int:
        """Total tokens the model accepts per input, including any special tokens it adds."""

    @property
    def special_token_overhead(self) -> int:
        """Tokens the model prepends/appends to every input.

        A slice of N content tokens is presented to the model as N + this. For a BERT-family model
        it is 2 ([CLS] and [SEP]), measured rather than assumed in ``WordPieceChunkTokenizer``.
        """

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        """Character spans of ``text``, one per content token, in order.

        Special tokens are excluded because they occupy no characters. Spans are half-open and index
        into ``text`` exactly as given -- no normalisation, no stripping.
        """


class WordPieceChunkTokenizer:
    """``ChunkTokenizer`` over a HuggingFace ``tokenizers.Tokenizer``."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        # Serialise/deserialise to get an independent instance. Calling no_truncation() on the
        # embedder's own tokenizer would disable truncation for inference too, which is the one
        # thing standing between an over-long input and an ONNX shape error.
        self._tokenizer = Tokenizer.from_str(tokenizer.to_str())
        limit = tokenizer.truncation or {}
        self._sequence_limit = int(limit.get("max_length") or 0)
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        self._overhead = self._measure_overhead()

    def _measure_overhead(self) -> int:
        """Count the model's added tokens by encoding a single known word.

        Read, not hardcoded to 2. A model whose post-processor adds a different number of special
        tokens would otherwise silently shift the usable budget and reintroduce truncation.
        """
        encoding = self._tokenizer.encode("token")
        return sum(1 for flag in encoding.special_tokens_mask if flag == 1)

    @property
    def sequence_limit(self) -> int:
        return self._sequence_limit

    @property
    def special_token_overhead(self) -> int:
        return self._overhead

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        if not text:
            return []
        encoding = self._tokenizer.encode(text)
        # special_tokens_mask, not a (0, 0) offset test: a real token can legitimately start at 0,
        # and filtering by offset would drop the first word of every text.
        return [
            (start, end)
            for (start, end), special in zip(
                encoding.offsets, encoding.special_tokens_mask
            )
            if special == 0
        ]
