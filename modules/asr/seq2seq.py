import torch
from torch import nn
from modules.commons.common_layers import SinusoidalPositionalEmbedding, Linear
from modules.fastspeech.tts_modules import TransformerDecoderLayer, DEFAULT_MAX_TARGET_POSITIONS, LayerNorm
import torch.nn.functional as F
from utils.hparams import hparams
from utils.tts_utils import fill_with_neg_inf2


class TransformerASRDecoder(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout, out_dim, use_pos_embed=True, num_heads=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.padding_idx = 0
        self.dropout = dropout
        self.out_dim = out_dim
        self.use_pos_embed = use_pos_embed
        if self.use_pos_embed:
            self.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS
            self.embed_positions = SinusoidalPositionalEmbedding(
                self.hidden_size, self.padding_idx,
                init_size=self.max_target_positions + self.padding_idx + 1,
            )
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(self.hidden_size, self.dropout, num_heads=num_heads)
            for i in range(self.num_layers)
        ])
        self.layer_norm = LayerNorm(self.hidden_size)
        self.project_out_dim = Linear(self.hidden_size, self.out_dim, bias=False)
        if hparams.get('save_hidden'):
            self.hiddens = []

    def forward(self, dec_inputs, encoder_out, incremental_state=None):
        """

        :param dec_inputs:  [B, T, H]
        :param encoder_out: [B, T, H]
        :return: [B, T, W]
        """
        self.hiddens = []
        self_attn_padding_mask = dec_inputs.abs().sum(-1).eq(0).data
        encoder_padding_mask = encoder_out.abs().sum(-1).eq(0)
        # embed positions
        x = dec_inputs
        if self.use_pos_embed:
            if incremental_state is not None:
                positions = self.embed_positions(
                    x.abs().sum(-1),
                    incremental_state=incremental_state
                )
                x = x[:, -1:, :]
                positions = positions[:, -1:, :]
                self_attn_padding_mask = None
            else:
                positions = self.embed_positions(x.abs().sum(-1))
            x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        encoder_out = encoder_out.transpose(0, 1)
        all_attn_logits = []
        for layer in self.layers:
            if incremental_state is None:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None
            x, attn_logits = layer(
                x,
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
            )
            all_attn_logits.append(attn_logits)
            if hparams.get('save_hidden'):
                self.hiddens.append(x.data.cpu().transpose(0, 1))

        x = self.layer_norm(x)
        if hparams.get('save_hidden'):
            self.hiddens.append(x.data.cpu().transpose(0, 1))

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # B x T x C -> B x T x W
        x = self.project_out_dim(x)
        return x, all_attn_logits

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
                not hasattr(self, '_future_mask')
                or self._future_mask is None
                or self._future_mask.device != tensor.device
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(fill_with_neg_inf2(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]
