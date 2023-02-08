from modules.commons.common_layers import *
from modules.fastspeech.fs2_vae import FVAE, FVAEEncoder, FVAEDecoder, WN
import torch.distributions as dist
import numpy as np
from utils.hparams import hparams


#############################################################
# local latent
#############################################################
class TMPFVAE(FVAE):
    def forward(self, x=None, x_mask=None, g=None, infer=False):
        """
        #  多返回一个mask   用来后续算KL
        :param x: [B, C_in_out, T]
        :param x_mask: [B, T]
        :param g: [B, C_g, T]
        :return:
        """
        g_sqz = self.g_pre_net(g)
        if not infer:
            z_q, m_q, logs_q, x_mask_sqz = self.encoder(x, x_mask, g_sqz)
            x_recon = self.decoder(z_q, x_mask, g)
            from torch.distributions import constraints
            if not constraints.positive.check(logs_q.exp()).all():
                invalid_index = (logs_q.exp() < 0).nonzero(as_tuple=True)[0]
                print('exp index:', invalid_index)
                print('logs_q:', logs_q[invalid_index])
                print('logs_q_exp:', logs_q.exp()[invalid_index])

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
            return x_recon, loss_kl, z_p, m_q, logs_q, x_mask_sqz, z_q
        else:
            latent_shape = [g_sqz.shape[0], self.latent_size, g_sqz.shape[2]]
            z_p = self.prior_dist.sample(latent_shape).to(g.device)
            if self.use_prior_glow:
                z_p = self.prior_flow(z_p, 1, g_sqz, reverse=True)
            x_recon = self.decoder(z_p, 1, g)
            return x_recon, z_p


class LatentMap(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.convs = nn.Sequential(
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1),
        )
        self.spk_proj = nn.Sequential(
            nn.Conv1d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 16, kernel_size=3, padding=1),
        )

    def forward(self, x, spk_emb):
        # spk_emb: B, 256, ?
        spk_emb = spk_emb[:, :, :x.shape[-1]]  # [B, 256, T // 4]
        spk_emb = self.spk_proj(spk_emb)  # [B, 256, T // 4] -> [B, 16, T // 4]
        x = x + spk_emb
        return self.convs(x)


#############################################################
# global latent
#############################################################
class GlobalFVAEEncoder(FVAEEncoder):
    def __init__(self, in_channels, hidden_channels, latent_channels, kernel_size,
                 n_layers, gin_channels=0, p_dropout=0, strides=[4]):
        super().__init__(in_channels, hidden_channels, latent_channels, kernel_size,
                 n_layers, gin_channels, p_dropout, strides)
        self.poolings = nn.Sequential(
            nn.Conv1d(latent_channels * 2, latent_channels * 2, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(latent_channels * 2),
            nn.Conv1d(latent_channels * 2, latent_channels * 2, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(latent_channels * 2),
            nn.Conv1d(latent_channels * 2, latent_channels * 2, kernel_size=3, stride=2),
        )

    def forward(self, x, x_mask, g):
        x = self.pre_net(x)
        x_mask = x_mask[:, :, ::np.prod(self.strides)][:, :, :x.shape[-1]]
        x = x * x_mask
        x = self.wn(x, x_mask, g) * x_mask
        x = self.out_proj(x)         # [B, 128 * 2, T // 4]
        x = torch.mean(self.poolings(x), dim=-1, keepdim=True)   # [B, 128 * 2, T //4 //8] -> [B, 128 * 2, 1]
        m, logs = torch.split(x, self.latent_channels, dim=1)   # [B, 128, 1]
        z = (m + torch.randn_like(m) * torch.exp(logs))
        return z, m, logs, x_mask

class GlobalFVAEDecoder(FVAEDecoder):
    def __init__(self, latent_channels, hidden_channels, out_channels, kernel_size,
                 n_layers, gin_channels=0, p_dropout=0,
                 strides=[4]):
        super().__init__(latent_channels, hidden_channels, out_channels, kernel_size,
                 n_layers, gin_channels, p_dropout, strides)
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
        # x : [B, latent, 1]
        x = x.repeat(1, 1, g.shape[-1] // 4)  # [B, H, T // 4]
        x = self.pre_net(x)
        x = x * x_mask
        x = self.wn(x, x_mask, g) * x_mask
        x = self.out_proj(x)
        return x

class GlobalFVAE(TMPFVAE):
    def __init__(self,
                 in_out_channels, hidden_channels, latent_size,
                 kernel_size, enc_n_layers, dec_n_layers, gin_channels, strides,
                 use_prior_glow, glow_hidden=None, glow_kernel_size=None, glow_n_blocks=None):
        super(GlobalFVAE, self).__init__(in_out_channels, hidden_channels, latent_size,
                 kernel_size, enc_n_layers, dec_n_layers, gin_channels, strides,
                 use_prior_glow, glow_hidden, glow_kernel_size, glow_n_blocks)
        del self.encoder
        del self.decoder
        self.encoder = GlobalFVAEEncoder(in_out_channels, hidden_channels, latent_size, kernel_size,
                                   enc_n_layers, gin_channels, strides=strides)
        self.decoder = GlobalFVAEDecoder(latent_size, hidden_channels, in_out_channels, kernel_size,
                                   dec_n_layers, gin_channels, strides=strides)


class GlobalLatentMap(LatentMap):
    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        self.convs = nn.Sequential(
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=1,),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=1,),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=1,),
        )
        self.spk_proj = nn.Sequential(
            nn.Conv1d(256, hidden_size, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
        )

    def forward(self, x, spk_emb):
        # spk_emb: B, 256, ?
        spk_emb = spk_emb[:, :, :x.shape[-1]]  # [B, 256, 1]
        spk_emb = self.spk_proj(spk_emb)  # [B, 256, 1] -> [B, 16, 1]
        x = x + spk_emb
        return self.convs(x)


#############################################################
# TechPrior
#############################################################
class TechPriorGlobalFVAE(GlobalFVAE):
    def __init__(self,
                 in_out_channels, hidden_channels, latent_size,
                 kernel_size, enc_n_layers, dec_n_layers, gin_channels, strides,
                 use_prior_glow, glow_hidden=None, glow_kernel_size=None, glow_n_blocks=None):
        super(TechPriorGlobalFVAE, self).__init__(in_out_channels, hidden_channels, latent_size,
                 kernel_size, enc_n_layers, dec_n_layers, gin_channels, strides,
                 use_prior_glow, glow_hidden, glow_kernel_size, glow_n_blocks)
        del self.prior_dist
        self.tech_embed = Embedding(hparams['num_techs'], hidden_channels)
        self.prior_predictor = nn.Sequential(
            nn.Linear(hidden_channels, latent_size),
            nn.ReLU(),
            nn.BatchNorm1d(latent_size),
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.BatchNorm1d(latent_size),
            nn.Linear(latent_size, latent_size),
        )

    def get_prior_dist(self, tech_cond):
        # tech_embed = self.tech_embed(tech_cond)  # [B, H]
        # predicted_m = self.prior_predictor(tech_embed)[:, :, None]  # [B, latent prior, 1]
        predicted_m = tech_cond[:, None, None].repeat(1, self.latent_size, 1).float()  #  [B, latent prior, 1]
        return dist.Normal(predicted_m, 1)

    def forward(self, x=None, x_mask=None, g=None, tech_cond=None, infer=False):
        """
        #  多返回一个mask   用来后续算KL
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
                logpx = self.get_prior_dist(tech_cond).log_prob(z_p)
                loss_kl = ((logqx - logpx) * x_mask_sqz).sum() / x_mask_sqz.sum() / logqx.shape[1]
            else:
                loss_kl = torch.distributions.kl_divergence(q_dist, self.get_prior_dist(tech_cond))
                loss_kl = (loss_kl * x_mask_sqz).sum() / x_mask_sqz.sum() / z_q.shape[1]
                z_p = None
            return x_recon, loss_kl, z_p, m_q, logs_q, x_mask_sqz, z_q
        else:
            latent_shape = [g_sqz.shape[0], self.latent_size, g_sqz.shape[2]]
            z_p = self.get_prior_dist(tech_cond).sample(latent_shape).to(g.device)
            if self.use_prior_glow:
                z_p = self.prior_flow(z_p, 1, g_sqz, reverse=True)
            x_recon = self.decoder(z_p, 1, g)
            return x_recon, z_p

#############################################################
# tech classifier
#############################################################
class TechClassifier(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv1d(latent_size, latent_size // 2, kernel_size=1, ),  # 64
            nn.BatchNorm1d(latent_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(latent_size // 2, latent_size // 4, kernel_size=1, ),  # 32
            nn.BatchNorm1d(latent_size // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(latent_size // 4, 2, kernel_size=1, ),  # 2
        )
        self.spk_proj = nn.Sequential(
            nn.Conv1d(256, latent_size, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(latent_size, latent_size, kernel_size=1),
        )

    def forward(self, x, spk_emb):
        # spk_emb: B, 256, ?
        spk_emb = spk_emb[:, :, :x.shape[-1]]  # [B, 256, 1]
        spk_emb = self.spk_proj(spk_emb)  # [B, 256, 1] -> [B, latent_size, 1]
        x = x + spk_emb
        return self.classifier(x)[:, :, 0]   # [B, 2]