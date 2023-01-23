import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.fastspeech.wavenet_decoder import WN
from modules.fastspeech.tts_modules import LayerNorm


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def init_weights_func(m):
    classname = m.__class__.__name__
    if classname.find("Conv1d") != -1:
        torch.nn.init.xavier_uniform_(m.weight)


class ResidualBlock(nn.Module):
    """Implements conv->PReLU->norm n-times"""

    def __init__(self, channels, kernel_size, dilation, n=2, norm_type='bn', dropout=0.0,
                 c_multiple=2, ln_eps=1e-12):
        super(ResidualBlock, self).__init__()

        if norm_type == 'bn':
            norm_builder = lambda: nn.BatchNorm1d(channels)
        elif norm_type == 'in':
            norm_builder = lambda: nn.InstanceNorm1d(channels, affine=True)
        elif norm_type == 'gn':
            norm_builder = lambda: nn.GroupNorm(8, channels)
        elif norm_type == 'ln':
            norm_builder = lambda: LayerNorm(channels, dim=1, eps=ln_eps)
        else:
            norm_builder = lambda: nn.Identity()

        self.blocks = [
            nn.Sequential(
                norm_builder(),
                nn.Conv1d(channels, c_multiple * channels, kernel_size, dilation=dilation,
                          padding=(dilation * (kernel_size - 1)) // 2),
                LambdaLayer(lambda x: x * kernel_size ** -0.5),
                nn.GELU(),
                nn.Conv1d(c_multiple * channels, channels, 1, dilation=dilation),
            )
            for i in range(n)
        ]

        self.blocks = nn.ModuleList(self.blocks)
        self.dropout = dropout

    def forward(self, x):
        nonpadding = (x.abs().sum(1) > 0).float()[:, None, :]
        for b in self.blocks:
            x_ = b(x)
            if self.dropout > 0 and self.training:
                x_ = F.dropout(x_, self.dropout, training=self.training)
            x = x + x_
            x = x * nonpadding
        return x


class ConvBlocks(nn.Module):
    """Decodes the expanded phoneme encoding into spectrograms"""

    def __init__(self, channels, out_dims, dilations, kernel_size,
                 norm_type='ln', layers_in_block=2, c_multiple=2,
                 dropout=0.0, ln_eps=1e-5, init_weights=True):
        super(ConvBlocks, self).__init__()
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels, kernel_size, d,
                            n=layers_in_block, norm_type=norm_type, c_multiple=c_multiple,
                            dropout=dropout, ln_eps=ln_eps)
              for d in dilations],
        )
        if norm_type == 'bn':
            norm = nn.BatchNorm1d(channels)
        elif norm_type == 'in':
            norm = nn.InstanceNorm1d(channels, affine=True)
        elif norm_type == 'gn':
            norm = nn.GroupNorm(8, channels)
        elif norm_type == 'ln':
            norm = LayerNorm(channels, dim=1, eps=ln_eps)
        self.last_norm = norm
        self.post_net1 = nn.Conv1d(channels, out_dims, kernel_size=3, padding=1)
        if init_weights:
            self.apply(init_weights_func)

    def forward(self, x):
        """

        :param x: [B, T, H]
        :return:  [B, T, H]
        """
        x = x.transpose(1, 2)
        nonpadding = (x.abs().sum(1) > 0).float()[:, None, :]
        x = self.res_blocks(x) * nonpadding
        x = self.last_norm(x) * nonpadding
        x = self.post_net1(x) * nonpadding
        return x.transpose(1, 2)


class ConditionalConvBlocks(ConvBlocks):
    def __init__(self, channels, g_channels, out_dims, dilations, kernel_size,
                 norm_type='ln', layers_in_block=2, c_multiple=2,
                 dropout=0.0, ln_eps=1e-5, init_weights=True, is_BTC=True):
        super().__init__(channels, out_dims, dilations, kernel_size,
                         norm_type, layers_in_block, c_multiple,
                         dropout, ln_eps, init_weights)
        self.g_prenet = nn.Conv1d(g_channels, channels, 3, padding=1)
        self.is_BTC = is_BTC
        if init_weights:
            self.g_prenet.apply(init_weights_func)

    def forward(self, x, g, x_mask):
        if self.is_BTC:
            x = x.transpose(1, 2)
            g = g.transpose(1, 2)
            x_mask = x_mask.transpose(1, 2)
        x = x + self.g_prenet(g)
        x = x * x_mask

        if not self.is_BTC:
            x = x.transpose(1, 2)
        x = super(ConditionalConvBlocks, self).forward(x)  # input needs to be BTC
        if not self.is_BTC:
            x = x.transpose(1, 2)
        return x


class ResidualCouplingLayer(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 p_dropout=0,
                 gin_channels=0,
                 mean_only=False,
                 nn_type='wn'):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        if nn_type == 'wn':
            self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout,
                          gin_channels=gin_channels)
        elif nn_type == 'conv':
            self.enc = ConditionalConvBlocks(
                hidden_channels, gin_channels, hidden_channels, [1] * n_layers, kernel_size,
                layers_in_block=1, is_BTC=False)
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask=x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = -torch.sum(logs, [1, 2])
            return x, logdet


class ResidualCouplingBlock(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 n_flows=4,
                 gin_channels=0,
                 nn_type='wn'):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                                      gin_channels=gin_channels, mean_only=True, nn_type=nn_type))
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        return x