from torch import nn
from modules.commons.common_layers import Embedding
from modules.commons.espnet_positional_embedding import RelPositionalEncoding
from modules.commons.espnet_transformer_attn import RelPositionMultiHeadedAttention
from modules.fastspeech.conformer.layers import Swish, ConvolutionModule, EncoderLayer, MultiLayeredConv1d
from utils.hparams import hparams


class ConformerLayers(nn.Module):
    def __init__(self, hidden_size, num_layers, kernel_size=None, dropout=None, num_heads=4,
                 use_last_norm=True):
        super().__init__()
        hidden_size = hparams['hidden_size'] if hidden_size is None else hidden_size
        kernel_size = hparams['enc_ffn_kernel_size'] if kernel_size is None else kernel_size
        num_heads = hparams['num_heads'] if num_heads is None else num_heads
        dropout = hparams['dropout'] if dropout is None else dropout
        self.use_last_norm = use_last_norm
        self.layers = nn.ModuleList()
        positionwise_layer = MultiLayeredConv1d
        positionwise_layer_args = (hidden_size, hidden_size * 4, 1, dropout)
        self.pos_embed = RelPositionalEncoding(hidden_size, dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer(
            hidden_size,
            RelPositionMultiHeadedAttention(num_heads, hidden_size, 0.0),
            positionwise_layer(*positionwise_layer_args),
            positionwise_layer(*positionwise_layer_args),
            ConvolutionModule(hidden_size, kernel_size, Swish()),
            dropout,
        ) for _ in range(num_layers)])
        if self.use_last_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = nn.Linear(hidden_size, hidden_size)
        if hparams.get('save_hidden'):
            self.hiddens = []

    def forward(self, x, padding_mask=None):
        """

        :param x: [B, T, H]
        :param padding_mask: [B, T]
        :return: [B, T, H]
        """
        self.hiddens = []
        nonpadding_mask = x.abs().sum(-1) > 0
        x = self.pos_embed(x)
        for l in self.encoder_layers:
            x, mask = l(x, nonpadding_mask[:, None, :])
            if hparams.get('save_hidden'):
                self.hiddens.append(x[0].data.cpu())
        x = x[0]
        x = self.layer_norm(x) * nonpadding_mask.float()[:, :, None]
        return x


class ConformerEncoder(ConformerLayers):
    def __init__(self, hidden_size, dict_size, num_layers=None):
        conformer_enc_kernel_size = 9
        num_layers = num_layers if num_layers is not None else hparams['enc_layers']
        super().__init__(hidden_size, num_layers, conformer_enc_kernel_size)
        self.embed = Embedding(dict_size, hidden_size, padding_idx=0)

    def forward(self, x):
        """

        :param src_tokens: [B, T]
        :return: [B x T x C]
        """
        x = self.embed(x)  # [B, T, H]
        x = super(ConformerEncoder, self).forward(x)
        return x


class ConformerDecoder(ConformerLayers):
    def __init__(self, hidden_size):
        conformer_dec_kernel_size = 9
        num_layers = hparams['dec_layers']
        super().__init__(hidden_size, num_layers, conformer_dec_kernel_size)
