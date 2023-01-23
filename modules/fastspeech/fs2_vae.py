import numpy as np
import torch
import torch.distributions as dist
from torch import nn
from modules.fastspeech.fs2 import FastSpeech2
from modules.voice_conversion.vae_models import WN
from modules.glow.glow_tts_modules import ResidualCouplingBlock
from utils.hparams import hparams


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
