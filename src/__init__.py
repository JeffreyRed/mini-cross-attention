"""mini-cross-attention — source package."""
from src.attention  import (
    scaled_dot_product_attention,
    SelfAttention, CrossAttention,
    MultiHeadSelfAttention, MultiHeadCrossAttention,
)
from src.dataset    import (
    VOCAB, IDX2TOK, VOCAB_SIZE, PAD_IDX, BOS_IDX, EOS_IDX,
    generate_pairs, ReversalDataset, collate_fn,
)
from src.model      import (
    PositionalEncoding, FeedForward,
    EncoderBlock, Encoder,
    DecoderBlock, Decoder,
    EncoderDecoder,
)
from src.train      import train, evaluate
from src.visualize  import plot_alignment, plot_self_vs_cross, plot_training

__all__ = [
    "scaled_dot_product_attention",
    "SelfAttention", "CrossAttention",
    "MultiHeadSelfAttention", "MultiHeadCrossAttention",
    "VOCAB", "IDX2TOK", "VOCAB_SIZE", "PAD_IDX", "BOS_IDX", "EOS_IDX",
    "generate_pairs", "ReversalDataset", "collate_fn",
    "PositionalEncoding", "FeedForward",
    "EncoderBlock", "Encoder",
    "DecoderBlock", "Decoder",
    "EncoderDecoder",
    "train", "evaluate",
    "plot_alignment", "plot_self_vs_cross", "plot_training",
]
