import torch.nn.functional as F
import torch
from transformers import AutoModel
from torch import nn


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.RMSNorm):
        nn.init.constant_(module.weight, 1.0)


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim, image_size):
        super(ImageEncoder, self).__init__()

        model_name = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
        self.backbone = AutoModel.from_pretrained(model_name)

        self.hidden_size = self.backbone.config.hidden_size
        self.grid_size = image_size // 16

        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=2)
        self.proj = nn.Sequential(
            nn.LayerNorm(self.hidden_size * 4),
            nn.Linear(self.hidden_size * 4, embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )
        self.norm = nn.RMSNorm(embed_dim)

        _init_weights(self.pixel_unshuffle)
        _init_weights(self.proj)
        _init_weights(self.norm)

    def forward(self, x):
        features = self.backbone(x).last_hidden_state[:, 5:, :]
        grid_features = features.transpose(1, 2).view(
            x.shape[0], self.hidden_size, self.grid_size, self.grid_size
        )

        unshuffled = self.pixel_unshuffle(grid_features)
        tokens = unshuffled.flatten(2).transpose(1, 2)
        projected = self.norm(self.proj(tokens))
        return projected


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, max_seq_len, embed_dim):
        super(RotaryPositionalEmbeddings, self).__init__()

        inv_freq = 1.0 / (10_000 ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)

        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x, start_pos=0):
        seq_len = x.shape[-2]

        cos = self.cos[start_pos : start_pos + seq_len].view(1, 1, seq_len, -1)
        sin = self.sin[start_pos : start_pos + seq_len].view(1, 1, seq_len, -1)

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        x_out1 = x1 * cos - x2 * sin
        x_out2 = x1 * sin + x2 * cos

        return torch.stack((x_out1, x_out2), dim=-1).flatten(-2)


class BaseMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_length):
        super(BaseMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.pre_norm = nn.RMSNorm(embed_dim)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.rope = RotaryPositionalEmbeddings(max_length, self.head_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def _prepare_qkv(self, q, k, v, start_pos):
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = self.rope(q, start_pos)
        k = self.rope(k, start_pos)
        return q, k, v

    def _compute_attention(self, q, k, v, attn_mask=None, padding_mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(padding_mask == 0, float("-inf"))

        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, v)
        return context


class SelfMultiHeadAttention(BaseMultiHeadAttention):
    def __init__(self, embed_dim, num_heads, max_length):
        super(SelfMultiHeadAttention, self).__init__(embed_dim, num_heads, max_length)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, attn_mask=None, padding_mask=None, key_value_cache=None):
        x_norm = self.pre_norm(x)

        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        start_pos = key_value_cache[0].shape[-2] if key_value_cache is not None else 0
        q, k, v = self._prepare_qkv(q, k, v, start_pos=start_pos)

        if key_value_cache is not None:
            past_k, past_v = key_value_cache

            full_k = torch.cat([past_k, k], dim=-2)
            full_v = torch.cat([past_v, v], dim=-2)
        else:
            full_k = k
            full_v = v

        context = self._compute_attention(q, full_k, full_v, attn_mask, padding_mask)

        batch_size, seq_len, _ = x.shape
        out = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return x + self.proj(out), (full_k, full_v)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(FeedForward, self).__init__()

        self.pre_norm = nn.RMSNorm(embed_dim)
        self.feature = nn.Linear(embed_dim, hidden_dim)
        self.gate = nn.Linear(embed_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x_norm = self.pre_norm(x)
        feature = F.silu(self.feature(x_norm))
        gate = self.gate(x_norm)
        return x + self.proj(feature * gate)


class DecoderTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, max_length):
        super(DecoderTransformerBlock, self).__init__()

        self.self_attention = SelfMultiHeadAttention(embed_dim, num_heads, max_length)
        self.feed_forward = FeedForward(embed_dim, embed_dim * 3)
        self.ffn_norm = nn.RMSNorm(embed_dim)

    def forward(self, x, attn_mask=None, padding_mask=None, key_value_cache=None):
        x, kv = self.self_attention(
            x, attn_mask, padding_mask, key_value_cache=key_value_cache
        )
        x = self.feed_forward(x)
        return x, kv


class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, max_length, depth):
        super(DecoderTransformer, self).__init__()

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderTransformerBlock(embed_dim, num_heads, max_length)
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.RMSNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

        self.apply(_init_weights)

    def forward(self, x, attn_mask=None, padding_mask=None, past_key_values=None):
        new_key_values = []

        for i, block in enumerate(self.decoder_blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None

            x, kv = block(x, attn_mask, padding_mask, key_value_cache=layer_past)
            new_key_values.append(kv)

        logits = self.output(self.final_norm(x))
        return logits, new_key_values


class ImageCaption(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        depth,
        image_size,
        max_text_tokens,
        token_embeddings,
        tokenizer,
    ):
        super(ImageCaption, self).__init__()

        self.num_image_tokens = (((image_size // 16) - 2) // 2 + 1) ** 2 + 2
        self.total_len = self.num_image_tokens + max_text_tokens + 1

        self.token_embeddings = token_embeddings
        self.vision_encoder = ImageEncoder(embed_dim, image_size)
        self.text_transformer = DecoderTransformer(
            tokenizer.vocab_size, embed_dim, num_heads, self.total_len, depth
        )

        mask = torch.ones((self.total_len, self.total_len), dtype=torch.bool)
        mask[: self.num_image_tokens, self.num_image_tokens :] = False
        text_causal = torch.tril(
            torch.ones(max_text_tokens + 1, max_text_tokens + 1, dtype=torch.bool)
        )
        mask[self.num_image_tokens :, self.num_image_tokens :] = text_causal

        token_ids = tokenizer(
            ["[IMG_START]", "[IMG_END]"], add_special_tokens=False, return_tensors="pt"
        ).input_ids
        with torch.no_grad():
            img_start_embed, img_end_embed = token_embeddings(token_ids)

        self.register_buffer("multimodal_mask", mask, persistent=False)
        self.register_buffer("img_start_embed", img_start_embed, persistent=False)
        self.register_buffer("img_end_embed", img_end_embed, persistent=False)

    def forward(
        self,
        pixel_values,
        input_ids,
        style_ids,
        padding_mask=None,
        past_key_values=None,
    ):
        batch_size = input_ids.shape[0]

        if past_key_values is not None:
            tokens = self.token_embeddings(input_ids)
        else:
            image_tokens = self.vision_encoder(pixel_values)
            style_tokens = self.token_embeddings(style_ids)
            text_tokens = self.token_embeddings(input_ids)

            start_expanded = self.img_start_embed.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            end_expanded = self.img_end_embed.unsqueeze(0).expand(batch_size, -1, -1)

            tokens = torch.cat(
                (start_expanded, image_tokens, end_expanded, style_tokens, text_tokens),
                dim=1,
            )

        seq_q = tokens.shape[1]

        if past_key_values is not None:
            prev_k_len = past_key_values[0][0].shape[-2]
            seq_k = prev_k_len + seq_q
        else:
            seq_k = seq_q

        attn_mask = self.multimodal_mask[seq_k - seq_q : seq_k, :seq_k]

        full_padding_mask = None
        if padding_mask is not None:
            if past_key_values is not None:
                full_padding_mask = padding_mask
            else:
                prefix_mask = torch.ones(
                    (batch_size, self.num_image_tokens + 1),
                    dtype=padding_mask.dtype,
                    device=padding_mask.device,
                )
                full_padding_mask = torch.cat((prefix_mask, padding_mask), dim=1)

        logits, new_key_values = self.text_transformer(
            tokens, attn_mask, full_padding_mask, past_key_values
        )

        if past_key_values is not None:
            return logits, new_key_values
        else:
            return logits[:, self.num_image_tokens + 1 :, :], new_key_values
