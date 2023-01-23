import numpy as np
import torch
import torch.distributions as dist
from torch import nn
from modules.fastspeech.fs2 import FastSpeech2
from modules.glow.glow_tts_modules import ResidualCouplingBlock
from utils.hparams import hparams


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class WN(torch.nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0,
                 p_dropout=0, share_cond_layers=False, is_BTC=False):
        super(WN, self).__init__()

        self.is_BTC = is_BTC
        assert (kernel_size % 2 == 1)
        assert (hidden_channels % 2 == 0)
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.share_cond_layers = share_cond_layers

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0 and not share_cond_layers:
            cond_layer = torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask=None, g=None, **kwargs):
        if self.is_BTC:
            x = x.transpose(1, 2)
            x_mask = x_mask.transpose(1, 2) if x_mask is not None else None
        if x_mask is None:
            x_mask = 1

        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None and not self.share_cond_layers:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            x_in = self.drop(x_in)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + res_skip_acts[:, :self.hidden_channels, :]) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)


class FVAEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, kernel_size,
                 n_layers, gin_channels=0, p_dropout=0, strides=[4]):
        super().__init__()
        self.strides = strides
        self.hidden_size = hidden_channels
        self.pre_net = nn.Sequential(*[
            nn.Conv1d(in_channels, hidden_channels, kernel_size=s * 2, stride=s, padding=s // 2)
            if i == 0 else
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=s * 2, stride=s, padding=s // 2)
            for i, s in enumerate(strides)
        ])
        self.wn = WN(hidden_channels, kernel_size, 1, n_layers, gin_channels, p_dropout)
        self.out_proj = nn.Conv1d(hidden_channels, latent_channels * 2, 1)
        self.latent_channels = latent_channels

    def forward(self, x, x_mask, g):
        x = self.pre_net(x)
        x_mask = x_mask[:, :, ::np.prod(self.strides)][:, :, :x.shape[-1]]
        x = x * x_mask
        x = self.wn(x, x_mask, g) * x_mask
        x = self.out_proj(x)
        m, logs = torch.split(x, self.latent_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs))
        return z, m, logs, x_mask


class FVAEDecoder(nn.Module):
    def __init__(self, latent_channels, hidden_channels, out_channels, kernel_size,
                 n_layers, gin_channels=0, p_dropout=0,
                 strides=[4]):
        super().__init__()
        self.strides = strides
        self.hidden_size = hidden_channels
        self.pre_net = nn.Sequential(*[
            nn.ConvTranspose1d(latent_channels, hidden_channels, kernel_size=s, stride=s)
            if i == 0 else
            nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=s, stride=s)
            for i, s in enumerate(strides)
        ])
        self.wn = WN(hidden_channels, kernel_size, 1, n_layers, gin_channels, p_dropout)
        self.out_proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, x_mask, g):
        x = self.pre_net(x)
        x = x * x_mask
        x = self.wn(x, x_mask, g) * x_mask
        x = self.out_proj(x)
        return x


class FVAE(nn.Module):
    def __init__(self,
                 in_out_channels, hidden_channels, latent_size,
                 kernel_size, enc_n_layers, dec_n_layers, gin_channels, strides,
                 use_prior_glow, glow_hidden=None, glow_kernel_size=None, glow_n_blocks=None):
        super(FVAE, self).__init__()
        self.strides = strides
        self.hidden_size = hidden_channels
        self.latent_size = latent_size
        self.use_prior_glow = use_prior_glow
        self.g_pre_net = nn.Sequential(*[
            nn.Conv1d(gin_channels, gin_channels, kernel_size=s * 2, stride=s, padding=s // 2)
            for i, s in enumerate(strides)
        ])
        self.encoder = FVAEEncoder(in_out_channels, hidden_channels, latent_size, kernel_size,
                                   enc_n_layers, gin_channels, strides=strides)
        if use_prior_glow:
            self.prior_flow = ResidualCouplingBlock(
                latent_size, glow_hidden, glow_kernel_size, 1, glow_n_blocks, 4, gin_channels=gin_channels)
        self.decoder = FVAEDecoder(latent_size, hidden_channels, in_out_channels, kernel_size,
                                   dec_n_layers, gin_channels, strides=strides)
        self.prior_dist = dist.Normal(0, 1)

    def forward(self, x=None, x_mask=None, g=None, infer=False):
        """

        :param x: [B, C_in_out, T]
        :param x_mask: [B, T]
        :param g: [B, C_g, T]
        :return:
        """
        g_sqz = self.g_pre_net(g)
        if not infer:
            z_q, m_q, logs_q, x_mask_sqz = self.encoder(x, x_mask, g_sqz)
            x_recon = self.decoder(z_q, x_mask, g)
            q_dist = dist.Normal(m_q, logs_q.exp())
            if self.use_prior_glow:
                logqx = q_dist.log_prob(z_q)
                z_p = self.prior_flow(z_q, x_mask_sqz, g_sqz)
                logpx = self.prior_dist.log_prob(z_p)
                loss_kl = ((logqx - logpx) * x_mask_sqz).sum() / x_mask_sqz.sum() / logqx.shape[1]
            else:
                loss_kl = torch.distributions.kl_divergence(q_dist, self.prior_dist)
                loss_kl = (loss_kl * x_mask_sqz).sum() / x_mask_sqz.sum() / z_q.shape[1]
                z_p = None
            return x_recon, loss_kl, z_p, m_q, logs_q
        else:
            latent_shape = [g_sqz.shape[0], self.latent_size, g_sqz.shape[2]]
            z_p = self.prior_dist.sample(latent_shape).to(g.device)
            if self.use_prior_glow:
                z_p = self.prior_flow(z_p, 1, g_sqz, reverse=True)
            x_recon = self.decoder(z_p, 1, g)
            return x_recon, z_p


class FastSpeech2VAE(FastSpeech2):
    def __init__(self, dictionary, out_dims=None):
        super().__init__(dictionary, out_dims)
        del self.decoder
        self.fvae = FVAE(
            in_out_channels=self.out_dims,
            hidden_channels=self.hidden_size * 3 // 4, latent_size=hparams['latent_size'],
            kernel_size=hparams['fvae_kernel_size'],
            enc_n_layers=hparams['fvae_enc_n_layers'],
            dec_n_layers=hparams['fvae_dec_n_layers'],
            gin_channels=self.hidden_size,
            use_prior_glow=hparams['use_prior_glow'],
            glow_hidden=hparams['prior_glow_hidden'],
            glow_kernel_size=hparams['glow_kernel_size'],
            glow_n_blocks=hparams['prior_glow_n_blocks'],
            strides=[4]
        )

    def run_decoder(self, x, tgt_nonpadding, ret, infer, **kwargs):
        x = x.transpose(1, 2)  # [B, H, T]
        tgt_nonpadding = tgt_nonpadding.transpose(1, 2)  # [B, H, T]
        if infer:
            mel_out, ret['z_p'] = self.fvae(g=x, infer=True)
        else:
            tgt_mels = kwargs['tgt_mels']
            tgt_mels = tgt_mels.transpose(1, 2)  # [B, 80, T]
            mel_out, ret['kl'], ret['z_p'], ret['m_q'], ret['logs_q'] = \
                self.fvae(tgt_mels, tgt_nonpadding, g=x)
        return (mel_out * tgt_nonpadding).transpose(1, 2)
