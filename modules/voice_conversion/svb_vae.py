from modules.commons.common_layers import *
from modules.fastspeech.fs2 import FS_DECODERS
from modules.voice_conversion.vc_modules import VCASR
from utils.hparams import hparams
from modules.voice_conversion.vae_models import TMPFVAE, GlobalFVAE, TechPriorGlobalFVAE
from modules.voice_conversion.vae_models import LatentMap, GlobalLatentMap, TechClassifier
import torch.distributions as dist
import numpy as np


class SVBVAE(nn.Module):
    def __init__(self, dict_size):
        super().__init__()
        self.hidden_size = hparams['hidden_size']
        self.c_content = self.c_out = hparams['audio_num_mel_bins']

        self.vae_model = TMPFVAE(
            in_out_channels=self.c_content,
            hidden_channels=hparams['fvae_enc_dec_hidden'],
            latent_size=hparams['latent_size'],
            kernel_size=hparams['fvae_kernel_size'],
            enc_n_layers=hparams['fvae_enc_n_layers'],
            dec_n_layers=hparams['fvae_dec_n_layers'],
            gin_channels=self.hidden_size,
            use_prior_glow=False,
            strides=[4]
        )

        self.encoder_proj_in = 0
        # pitch encoder
        self.pitch_embed = Embedding(300, self.hidden_size, 0)
        self.pitch_encoder = ConvStacks(
            idim=self.hidden_size, n_chans=self.hidden_size, odim=self.hidden_size, n_layers=3)
        self.encoder_proj_in += self.hidden_size
        # asr content encoder
        self.vc_asr = VCASR(dict_size, self.c_content)
        self.upsample_layer = nn.Sequential(
            *([nn.Sequential(
                nn.Upsample(scale_factor=scale, mode='nearest'),
                nn.Conv1d(self.hidden_size, self.hidden_size, scale * 2 + 1, padding=scale),
                nn.ReLU(), nn.BatchNorm1d(self.hidden_size))
                  for scale in hparams['mel_strides'] if scale > 1
              ] + [nn.Conv1d(self.hidden_size, self.hidden_size, 5, padding=2)]))
        self.encoder_proj_in += self.hidden_size
        # mel content encoder
        # if hparams['use_energy']:
        #     self.energy_embed = Embedding(256, self.hidden_size, 0)
        #     self.encoder_proj_in += self.hidden_size
        self.spk_embed_proj = Linear(256, self.hidden_size, bias=True)
        self.encoder_proj_in += self.hidden_size
        # if hparams['use_tech']:
        #     self.tech_embed = Embedding(hparams['num_techs'], self.hidden_size)
        #     self.encoder_proj_in += self.hidden_size
        self.encoded_embed_proj = Linear(self.encoder_proj_in, self.hidden_size)
        self.m_mapping_function = LatentMap(hparams['latent_size'])
        self.logs_mapping_function = LatentMap(hparams['latent_size'])

    def prepare_condition(self, mels_content=None, pitch=None, spk_ids=None):
        ret = {}
        T = pitch.shape[1]
        # Pitch embedding
        h_pitch = self.pitch_encoder(self.pitch_embed(pitch))
        ret['h_pitch'] = h_pitch

        # ASR content embedding
        h_content = self.vc_asr(mels_content)['h_content'].detach()
        ret['h_content'] = self.upsample_layer(h_content.transpose(1, 2)).transpose(1, 2)[:, :mels_content.shape[1]]  # [B, S, H]

        # if hparams['use_energy']:
        #     energy = torch.clamp(energy * 256 // 4, max=255).long()
        #     h_energy = self.energy_embed(energy)
        #     ret['h_energy'] = h_energy
        #
        # if hparams['use_tech']:
        #     h_tech = self.tech_embed(tech_id)[:, None, :].repeat(1, T, 1)
        #     ret['h_tech'] = h_tech

        # Spk embedding
        h_style = self.spk_embed_proj(spk_ids)[:, None, :].repeat(1, T, 1)
        ret['h_style'] = h_style

        tgt_nonpadding = (pitch > 0).float()[:, :, None]
        ret['tgt_nonpadding'] = tgt_nonpadding
        return ret

    # self.encoded_embed_proj(torch.cat(encoded_embed, -1))
    def forward(self, amateur_mel=None, prof_mel=None,
                amateur_pitch=None, prof_pitch=None,
                amateur_spk_id=None, prof_spk_id=None,
                a2p_alignment=None, p2a_alignment=None,
                infer=False, disable_map=False, **kwargs):
        # print(amateur_mel.shape, prof_mel.shape, amateur_pitch.shape, prof_pitch.shape)
        ret = {}
        concurrent_ways = kwargs['concurrent_ways']

        # prepare all conditions:
        amateur_conds = self.prepare_condition(amateur_mel, amateur_pitch, spk_ids=amateur_spk_id)
        prof_conds = self.prepare_condition(prof_mel, prof_pitch, spk_ids=prof_spk_id)

        if 'a2a' in concurrent_ways:
            a2a_out = self.normal_vae(amateur_mel, amateur_conds['h_pitch'], amateur_conds['h_content'],
                                     amateur_conds['h_style'], amateur_conds['tgt_nonpadding'], infer=infer)
            ret['a2a'] = a2a_out
        if 'p2p' in concurrent_ways:
            p2p_out = self.normal_vae(prof_mel, prof_conds['h_pitch'], prof_conds['h_content'],
                                     prof_conds['h_style'], prof_conds['tgt_nonpadding'], infer=infer)
            ret['p2p'] = p2p_out
        if 'a2p' in concurrent_ways:
            a2p_out = {}
            amatuer_m_q, amatuer_logs_q = a2a_out['m_q'], a2a_out['logs_q']
            prof_m_q, prof_logs_q = p2p_out['m_q'], p2p_out['logs_q']

            # align  miu and sigma to the length of prof mel
            a2p_alignment = a2p_alignment[:, :, None].repeat(1, 1, self.hidden_size)  # [B, S, H]
            a2p_alignment_shrink = F.interpolate(
                a2p_alignment[:, :, :hparams['latent_size']].transpose(1, 2).float() / hparams['frames_multiple'],
                scale_factor=1 / hparams['frames_multiple']).long()  # [B, 16, S]
            amatuer_m_q = torch.gather(amatuer_m_q, 2, a2p_alignment_shrink)  # [B, 16, S]
            amatuer_logs_q = torch.gather(amatuer_logs_q, 2, a2p_alignment_shrink)

            if disable_map:
                mapped_amatuer_m_q = amatuer_m_q
                mapped_amatuer_logs_q = amatuer_logs_q
            else:
                mapped_amatuer_m_q = self.m_mapping_function(amatuer_m_q, amateur_conds['h_style'].transpose(1,2))  # miu_a -> miu_p'
                mapped_amatuer_logs_q = self.logs_mapping_function(amatuer_logs_q, amateur_conds['h_style'].transpose(1,2))  # log-sigma_a -> log-sigma

            mapped_amatuer_dist = dist.Normal(mapped_amatuer_m_q, mapped_amatuer_logs_q.exp())
            prof_dist = dist.Normal(prof_m_q, prof_logs_q.exp())

            mapping_loss_kl = torch.distributions.kl_divergence(mapped_amatuer_dist, prof_dist)
            a2p_out['kl'] = (mapping_loss_kl * p2p_out['x_mask_sqz']).sum() / p2p_out['x_mask_sqz'].sum() / p2p_out['z_q'].shape[1]

            # align
            a2p_cond_sum = self.encoded_embed_proj(torch.cat([
                                prof_conds['h_pitch'],
                                torch.gather(amateur_conds['h_content'], 1, a2p_alignment),
                                amateur_conds['h_style'][:, :1, :].repeat(1, prof_conds['h_pitch'].shape[1], 1)
                                ], -1)).transpose(1, 2)  # [B, H, T]
            # recon a2p from miu
            a2p_out['mel_out'] = self.vae_model.decoder(mapped_amatuer_m_q, prof_conds['tgt_nonpadding'].transpose(1, 2), g=a2p_cond_sum).transpose(1, 2)

            # recon a2p from sampling
            mapped_amatuer_z_q = mapped_amatuer_m_q + torch.randn_like(mapped_amatuer_m_q) * torch.exp(mapped_amatuer_logs_q)  # sample a data point from mapped posterior
            a2p_out['a2p_sample_recon'] = self.vae_model.decoder(mapped_amatuer_z_q, prof_conds['tgt_nonpadding'].transpose(1, 2), g=a2p_cond_sum).transpose(1, 2)
            ret['a2p'] = a2p_out

        return ret

    def normal_vae(self, tgt_mel, pitch_cond, content_cond, timbre_cond, padding_cond, infer):
        cond_sum = self.encoded_embed_proj(torch.cat([pitch_cond, content_cond, timbre_cond], -1)).transpose(1, 2)  # [B, H, T]
        vae_out = {}
        if infer:
            vae_mel_out = self.vae_model(g=cond_sum, infer=True)
            vae_out['mel_out'] = vae_mel_out.transpose(1, 2)
        else:
            vae_mel_out, vae_out['kl'], vae_out['z_p'], vae_out['m_q'], vae_out['logs_q'], vae_out['x_mask_sqz'], vae_out['z_q'] = \
                self.vae_model(tgt_mel.transpose(1, 2), padding_cond.transpose(1, 2), g=cond_sum)
            vae_out['mel_out'] = vae_mel_out.transpose(1, 2)
        return vae_out

    def train_vc_asr(self, mels, tokens):
        prev_tokens = F.pad(tokens[:, :-1], [1, 0], mode='constant', value=0)
        return self.vc_asr(mels, prev_tokens)['tokens']


### code below for global vae latent


class GlobalSVBVAE(SVBVAE):
    def __init__(self, dict_size):
        super().__init__(dict_size)
        del self.vae_model
        del self.m_mapping_function
        del self.logs_mapping_function
        self.m_mapping_function = GlobalLatentMap(hparams['latent_size'])
        self.logs_mapping_function = GlobalLatentMap(hparams['latent_size'])

        self.vae_model = GlobalFVAE(
            in_out_channels=self.c_content,
            hidden_channels=hparams['fvae_enc_dec_hidden'],
            latent_size=hparams['latent_size'],
            kernel_size=hparams['fvae_kernel_size'],
            enc_n_layers=hparams['fvae_enc_n_layers'],
            dec_n_layers=hparams['fvae_dec_n_layers'],
            gin_channels=self.hidden_size,
            use_prior_glow=False,
            strides=[4]
        )

    def forward(self, amateur_mel=None, prof_mel=None,
                amateur_pitch=None, prof_pitch=None,
                amateur_spk_id=None, prof_spk_id=None,
                a2p_alignment=None, p2a_alignment=None,
                infer=False, disable_map=False, **kwargs):
        # print(amateur_mel.shape, prof_mel.shape, amateur_pitch.shape, prof_pitch.shape)
        ret = {}
        concurrent_ways = kwargs['concurrent_ways']

        # prepare all conditions:
        amateur_conds = self.prepare_condition(amateur_mel, amateur_pitch, spk_ids=amateur_spk_id)
        prof_conds = self.prepare_condition(prof_mel, prof_pitch, spk_ids=prof_spk_id)

        if 'a2a' in concurrent_ways:
            a2a_out = self.normal_vae(amateur_mel, amateur_conds['h_pitch'], amateur_conds['h_content'],
                                     amateur_conds['h_style'], amateur_conds['tgt_nonpadding'], infer=infer)
            ret['a2a'] = a2a_out
        if 'p2p' in concurrent_ways:
            p2p_out = self.normal_vae(prof_mel, prof_conds['h_pitch'], prof_conds['h_content'],
                                     prof_conds['h_style'], prof_conds['tgt_nonpadding'], infer=infer)
            ret['p2p'] = p2p_out
        if 'a2p' in concurrent_ways:
            a2p_out = {}
            amatuer_m_q, amatuer_logs_q = a2a_out['m_q'], a2a_out['logs_q']
            prof_m_q, prof_logs_q = p2p_out['m_q'], p2p_out['logs_q']

            # align  miu and sigma to the length of prof mel
            a2p_alignment = a2p_alignment[:, :, None].repeat(1, 1, self.hidden_size)  # [B, S, H]

            if disable_map:
                mapped_amatuer_m_q = amatuer_m_q
                mapped_amatuer_logs_q = amatuer_logs_q
            else:
                mapped_amatuer_m_q = self.m_mapping_function(amatuer_m_q, amateur_conds['h_style'].transpose(1,2))  # miu_a -> miu_p'
                mapped_amatuer_logs_q = self.logs_mapping_function(amatuer_logs_q, amateur_conds['h_style'].transpose(1,2))  # log-sigma_a -> log-sigma
            mapped_amatuer_dist = dist.Normal(mapped_amatuer_m_q, mapped_amatuer_logs_q.exp())
            prof_dist = dist.Normal(prof_m_q, prof_logs_q.exp())

            mapping_loss_kl = torch.distributions.kl_divergence(mapped_amatuer_dist, prof_dist)
            a2p_out['kl'] = mapping_loss_kl.sum() / p2p_out['z_q'].shape[0] / p2p_out['z_q'].shape[1]

            # align
            a2p_cond_sum = self.encoded_embed_proj(torch.cat([
                                prof_conds['h_pitch'],
                                torch.gather(amateur_conds['h_content'], 1, a2p_alignment),
                                amateur_conds['h_style'][:, :1, :].repeat(1, prof_conds['h_pitch'].shape[1], 1)
                                ], -1)).transpose(1, 2)  # [B, H, T]
            # recon a2p from miu
            a2p_out['mel_out'] = self.vae_model.decoder(mapped_amatuer_m_q, prof_conds['tgt_nonpadding'].transpose(1, 2), g=a2p_cond_sum).transpose(1, 2)

            # recon a2p from sampling
            mapped_amatuer_z_q = mapped_amatuer_m_q + torch.randn_like(mapped_amatuer_m_q) * torch.exp(mapped_amatuer_logs_q)  # sample a data point from mapped posterior
            a2p_out['a2p_sample_recon'] = self.vae_model.decoder(mapped_amatuer_z_q, prof_conds['tgt_nonpadding'].transpose(1, 2), g=a2p_cond_sum).transpose(1, 2)
            ret['a2p'] = a2p_out

        return ret


class MleSVBVAE(GlobalSVBVAE):
    def __init__(self, dict_size):
        super().__init__(dict_size)
        del self.m_mapping_function
        del self.logs_mapping_function
        self.z_mapping_function = GlobalLatentMap(hparams['latent_size'])

    def forward(self, amateur_mel=None, prof_mel=None,
                amateur_pitch=None, prof_pitch=None,
                amateur_spk_id=None, prof_spk_id=None,
                a2p_alignment=None, p2a_alignment=None,
                infer=False, disable_map=False, **kwargs):
        # print(amateur_mel.shape, prof_mel.shape, amateur_pitch.shape, prof_pitch.shape)
        ret = {}
        concurrent_ways = kwargs['concurrent_ways']

        # prepare all conditions:
        amateur_conds = self.prepare_condition(amateur_mel, amateur_pitch, spk_ids=amateur_spk_id)
        prof_conds = self.prepare_condition(prof_mel, prof_pitch, spk_ids=prof_spk_id)

        if 'a2a' in concurrent_ways:
            a2a_out = self.normal_vae(amateur_mel, amateur_conds['h_pitch'], amateur_conds['h_content'],
                                     amateur_conds['h_style'], amateur_conds['tgt_nonpadding'], infer=infer)
            ret['a2a'] = a2a_out
        if 'p2p' in concurrent_ways:
            p2p_out = self.normal_vae(prof_mel, prof_conds['h_pitch'], prof_conds['h_content'],
                                     prof_conds['h_style'], prof_conds['tgt_nonpadding'], infer=infer)
            ret['p2p'] = p2p_out
        if 'a2p' in concurrent_ways:
            a2p_out = {}
            amatuer_z_q = a2a_out['z_q']
            prof_m_q, prof_logs_q = p2p_out['m_q'], p2p_out['logs_q']

            # align  miu and sigma to the length of prof mel
            a2p_alignment = a2p_alignment[:, :, None].repeat(1, 1, self.hidden_size)  # [B, S, H]

            if disable_map:
                print('here disable map!!!')
                mapped_amatuer_z_q = amatuer_z_q
            else:
                mapped_amatuer_z_q = self.z_mapping_function(amatuer_z_q, amateur_conds['h_style'].transpose(1,2))  # amateur_z -> prof_z'

            prof_dist = dist.Normal(prof_m_q, prof_logs_q.exp())

            a2p_out['mle'] = - prof_dist.log_prob(mapped_amatuer_z_q).sum() / mapped_amatuer_z_q.shape[0] / mapped_amatuer_z_q.shape[1]  # [B, H, 1] 除以B, 除以H.

            # align
            a2p_cond_sum = self.encoded_embed_proj(torch.cat([
                                prof_conds['h_pitch'],
                                torch.gather(amateur_conds['h_content'], 1, a2p_alignment),
                                amateur_conds['h_style'][:, :1, :].repeat(1, prof_conds['h_pitch'].shape[1], 1)
                                ], -1)).transpose(1, 2)  # [B, H, T]
            # recon a2p from mapped z
            a2p_out['mel_out'] = self.vae_model.decoder(mapped_amatuer_z_q, prof_conds['tgt_nonpadding'].transpose(1, 2), g=a2p_cond_sum).transpose(1, 2)

            # logs for vis
            a2p_out['logs_amateur_zq'] = a2a_out['z_q']
            a2p_out['logs_prof_zq'] = p2p_out['z_q']

            ret['a2p'] = a2p_out

        return ret


class ClassifyMleSVCVAE(MleSVBVAE):
    def __init__(self, dict_size):
        super().__init__(dict_size)
        self.tech_classifier = TechClassifier(hparams['latent_size'])

    def forward(self, amateur_mel=None, prof_mel=None,
                amateur_pitch=None, prof_pitch=None,
                amateur_spk_id=None, prof_spk_id=None,
                a2p_alignment=None, p2a_alignment=None,
                infer=False, disable_map=False, **kwargs):
        # print(amateur_mel.shape, prof_mel.shape, amateur_pitch.shape, prof_pitch.shape)
        ret = {}
        concurrent_ways = kwargs['concurrent_ways']

        # prepare all conditions:
        amateur_conds = self.prepare_condition(amateur_mel, amateur_pitch, spk_ids=amateur_spk_id)
        prof_conds = self.prepare_condition(prof_mel, prof_pitch, spk_ids=prof_spk_id)

        if 'a2a' in concurrent_ways:
            a2a_out = self.normal_vae(amateur_mel, amateur_conds['h_pitch'], amateur_conds['h_content'],
                                     amateur_conds['h_style'], amateur_conds['tgt_nonpadding'], infer=infer)
            amateur_tech_id = F.one_hot(torch.zeros([amateur_mel.shape[0]], dtype=torch.long, device=amateur_mel.device), 2).float()
            amt_latent_pred = self.tech_classifier(a2a_out['z_q'], amateur_conds['h_style'].transpose(1,2))   # [B, 2,]
            a2a_out['latent_classify'] = F.binary_cross_entropy_with_logits(amt_latent_pred, amateur_tech_id)
            ret['a2a'] = a2a_out
        if 'p2p' in concurrent_ways:
            p2p_out = self.normal_vae(prof_mel, prof_conds['h_pitch'], prof_conds['h_content'],
                                     prof_conds['h_style'], prof_conds['tgt_nonpadding'], infer=infer)
            prof_tech_id = F.one_hot(torch.ones([prof_mel.shape[0]], dtype=torch.long, device=prof_mel.device), 2).float()
            prof_latent_pred = self.tech_classifier(p2p_out['z_q'], prof_conds['h_style'].transpose(1,2))
            p2p_out['latent_classify'] = F.binary_cross_entropy_with_logits(prof_latent_pred, prof_tech_id)
            ret['p2p'] = p2p_out
        if 'a2p' in concurrent_ways:
            a2p_out = {}
            amatuer_z_q = a2a_out['z_q']
            prof_m_q, prof_logs_q = p2p_out['m_q'], p2p_out['logs_q']

            # align  miu and sigma to the length of prof mel
            a2p_alignment = a2p_alignment[:, :, None].repeat(1, 1, self.hidden_size)  # [B, S, H]

            if disable_map:
                print('here disable map!!!')
                mapped_amatuer_z_q = amatuer_z_q
            else:
                mapped_amatuer_z_q = self.z_mapping_function(amatuer_z_q.detach(), amateur_conds['h_style'].transpose(1,2).detach())  # amateur_z -> prof_z'

            prof_dist = dist.Normal(prof_m_q.detach(), prof_logs_q.detach().exp())

            a2p_out['mle'] = - prof_dist.log_prob(mapped_amatuer_z_q).sum() / mapped_amatuer_z_q.shape[0] / mapped_amatuer_z_q.shape[1]  # [B, H, 1] 除以B, 除以H.

            # align
            a2p_cond_sum = self.encoded_embed_proj(torch.cat([
                                prof_conds['h_pitch'],
                                torch.gather(amateur_conds['h_content'], 1, a2p_alignment),
                                amateur_conds['h_style'][:, :1, :].repeat(1, prof_conds['h_pitch'].shape[1], 1)
                                ], -1)).transpose(1, 2)  # [B, H, T]
            # recon a2p from mapped z
            a2p_out['mel_out'] = self.vae_model.decoder(mapped_amatuer_z_q, prof_conds['tgt_nonpadding'].transpose(1, 2), g=a2p_cond_sum).transpose(1, 2)

            # logs for vis
            a2p_out['vis_amateur_zq'] = a2a_out['z_q']
            a2p_out['vis_prof_zq'] = p2p_out['z_q']
            a2p_out['vis_mapped_zq'] = mapped_amatuer_z_q

            ret['a2p'] = a2p_out

        return ret


class TechPriorMleSVBVAE(MleSVBVAE):
    def __init__(self, dict_size):
        super().__init__(dict_size)
        del self.vae_model
        self.vae_model = TechPriorGlobalFVAE(
            in_out_channels=self.c_content,
            hidden_channels=hparams['fvae_enc_dec_hidden'],
            latent_size=hparams['latent_size'],
            kernel_size=hparams['fvae_kernel_size'],
            enc_n_layers=hparams['fvae_enc_n_layers'],
            dec_n_layers=hparams['fvae_dec_n_layers'],
            gin_channels=self.hidden_size,
            use_prior_glow=False,
            strides=[4]
        )

    def forward(self, amateur_mel=None, prof_mel=None,
                amateur_pitch=None, prof_pitch=None,
                amateur_spk_id=None, prof_spk_id=None,
                a2p_alignment=None, p2a_alignment=None,
                infer=False, disable_map=False, **kwargs):
        # print(amateur_mel.shape, prof_mel.shape, amateur_pitch.shape, prof_pitch.shape)
        ret = {}
        concurrent_ways = kwargs['concurrent_ways']

        # prepare all conditions:
        amateur_conds = self.prepare_condition(amateur_mel, amateur_pitch, spk_ids=amateur_spk_id)
        prof_conds = self.prepare_condition(prof_mel, prof_pitch, spk_ids=prof_spk_id)

        if 'a2a' in concurrent_ways:
            amateur_tech_id = torch.zeros([amateur_mel.shape[0]], dtype=torch.long, device=amateur_mel.device)
            a2a_out = self.normal_vae(amateur_mel, amateur_conds['h_pitch'], amateur_conds['h_content'],
                                     amateur_conds['h_style'], amateur_conds['tgt_nonpadding'], tech_cond=amateur_tech_id, infer=infer)
            ret['a2a'] = a2a_out
        if 'p2p' in concurrent_ways:
            prof_tech_id = torch.ones([prof_mel.shape[0]], dtype=torch.long, device=prof_mel.device)
            p2p_out = self.normal_vae(prof_mel, prof_conds['h_pitch'], prof_conds['h_content'],
                                     prof_conds['h_style'], prof_conds['tgt_nonpadding'], tech_cond=prof_tech_id, infer=infer)
            ret['p2p'] = p2p_out
        if 'a2p' in concurrent_ways:
            a2p_out = {}
            amatuer_z_q = a2a_out['z_q']
            prof_m_q, prof_logs_q = p2p_out['m_q'], p2p_out['logs_q']

            # align  miu and sigma to the length of prof mel
            a2p_alignment = a2p_alignment[:, :, None].repeat(1, 1, self.hidden_size)  # [B, S, H]

            if disable_map:
                mapped_amatuer_z_q = amatuer_z_q
            else:
                mapped_amatuer_z_q = self.z_mapping_function(amatuer_z_q, amateur_conds['h_style'].transpose(1,2))  # amateur_z -> prof_z'

            prof_dist = dist.Normal(prof_m_q, prof_logs_q.exp())

            a2p_out['mle'] = - prof_dist.log_prob(mapped_amatuer_z_q).sum() / mapped_amatuer_z_q.shape[0] / mapped_amatuer_z_q.shape[1]  # [B, H, 1] 除以B, 除以H.

            # align
            a2p_cond_sum = self.encoded_embed_proj(torch.cat([
                                prof_conds['h_pitch'],
                                torch.gather(amateur_conds['h_content'], 1, a2p_alignment),
                                amateur_conds['h_style'][:, :1, :].repeat(1, prof_conds['h_pitch'].shape[1], 1)
                                ], -1)).transpose(1, 2)  # [B, H, T]
            # recon a2p from mapped z
            a2p_out['mel_out'] = self.vae_model.decoder(mapped_amatuer_z_q, prof_conds['tgt_nonpadding'].transpose(1, 2), g=a2p_cond_sum).transpose(1, 2)

            # logs for vis
            a2p_out['logs_amateur_zq'] = a2a_out['z_q']
            a2p_out['logs_prof_zq'] = p2p_out['z_q']

            ret['a2p'] = a2p_out
        return ret

    def normal_vae(self, tgt_mel, pitch_cond, content_cond, timbre_cond, padding_cond, tech_cond, infer):
        cond_sum = self.encoded_embed_proj(torch.cat([pitch_cond, content_cond, timbre_cond], -1)).transpose(1, 2)  # [B, H, T]
        vae_out = {}
        if infer:
            vae_mel_out, _ = self.vae_model(g=cond_sum, tech_cond=tech_cond, infer=True)
            vae_out['mel_out'] = vae_mel_out.transpose(1, 2)
        else:
            vae_mel_out, vae_out['kl'], vae_out['z_p'], vae_out['m_q'], vae_out['logs_q'], vae_out['x_mask_sqz'], vae_out['z_q'] = \
                self.vae_model(tgt_mel.transpose(1, 2), padding_cond.transpose(1, 2), g=cond_sum, tech_cond=tech_cond)
            vae_out['mel_out'] = vae_mel_out.transpose(1, 2)
        return vae_out

##################################
# tech prior + Attn alignment
##################################
class SegTechPriorMleSVBVAE(TechPriorMleSVBVAE):
    def __init__(self, dict_size):
        super().__init__(dict_size)
        self.k_mel_encoder = nn.Sequential(
            nn.Conv1d(self.c_content, self.hidden_size, kernel_size=1),
            nn.ReLU(self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=1)
        )
        self.ref_attn = MultiheadAttention(self.hidden_size, 4, encoder_decoder_attention=True)

    def get_aligned_ppg(self, src_ppg, src_mel, alignment):
        alignment = alignment[:, :, None].repeat(1, 1, self.hidden_size)  # [B, S, H]
        gathered_ppg = torch.gather(src_ppg, 1, alignment)

        k_src_mel = self.k_mel_encoder(src_mel.transpose(1, 2)).permute(2, 0, 1) # B,S,80->B,H,S->S,B,H
        attn, attn_weights = self.ref_attn(gathered_ppg.transpose(0, 1), k_src_mel,
                                src_ppg.transpose(0, 1), attn_mask=None)  # query, key, value,
        return attn.transpose(0, 1), attn_weights

    def forward(self, amateur_mel=None, prof_mel=None,
                amateur_pitch=None, prof_pitch=None,
                amateur_spk_id=None, prof_spk_id=None,
                a2p_alignment=None, p2a_alignment=None,
                infer=False, disable_map=False, **kwargs):
        # print(amateur_mel.shape, prof_mel.shape, amateur_pitch.shape, prof_pitch.shape)
        ret = {}
        concurrent_ways = kwargs['concurrent_ways']

        # prepare all conditions:
        amateur_conds = self.prepare_condition(amateur_mel, amateur_pitch, spk_ids=amateur_spk_id)
        prof_conds = self.prepare_condition(prof_mel, prof_pitch, spk_ids=prof_spk_id)

        if 'a2a' in concurrent_ways:
            amateur_tech_id = torch.zeros([amateur_mel.shape[0]], dtype=torch.long, device=amateur_mel.device)
            a2a_out = self.normal_vae(amateur_mel, amateur_conds['h_pitch'], amateur_conds['h_content'],
                                     amateur_conds['h_style'], amateur_conds['tgt_nonpadding'], tech_cond=amateur_tech_id, infer=infer)
            ret['a2a'] = a2a_out
        if 'p2p' in concurrent_ways:
            prof_tech_id = torch.ones([prof_mel.shape[0]], dtype=torch.long, device=prof_mel.device)
            # align amateur ppg to prof length

            fake_prof_ppg, attn_weights = self.get_aligned_ppg(amateur_conds['h_content'], amateur_mel, a2p_alignment)
            p2p_out = self.normal_vae(prof_mel, prof_conds['h_pitch'], fake_prof_ppg,
                                     prof_conds['h_style'], prof_conds['tgt_nonpadding'], tech_cond=prof_tech_id, infer=infer)
            p2p_out['attn'] = attn_weights
            ret['p2p'] = p2p_out
        if 'a2p' in concurrent_ways:
            a2p_out = {}
            amatuer_z_q = a2a_out['z_q']
            prof_m_q, prof_logs_q = p2p_out['m_q'], p2p_out['logs_q']

            if disable_map:
                mapped_amatuer_z_q = amatuer_z_q
            else:
                mapped_amatuer_z_q = self.z_mapping_function(amatuer_z_q, amateur_conds['h_style'].transpose(1,2))  # amateur_z -> prof_z'

            prof_dist = dist.Normal(prof_m_q, prof_logs_q.exp())

            a2p_out['mle'] = - prof_dist.log_prob(mapped_amatuer_z_q).sum() / mapped_amatuer_z_q.shape[0] / mapped_amatuer_z_q.shape[1]  # [B, H, 1] 除以B, 除以H.

            # align
            fake_prof_ppg, attn_weights = self.get_aligned_ppg(amateur_conds['h_content'], amateur_mel, a2p_alignment)
            a2p_cond_sum = self.encoded_embed_proj(torch.cat([
                                prof_conds['h_pitch'],
                                fake_prof_ppg,
                                amateur_conds['h_style'][:, :1, :].repeat(1, prof_conds['h_pitch'].shape[1], 1)
                                ], -1)).transpose(1, 2)  # [B, H, T]
            # recon a2p from mapped z
            a2p_out['mel_out'] = self.vae_model.decoder(mapped_amatuer_z_q, prof_conds['tgt_nonpadding'].transpose(1, 2), g=a2p_cond_sum).transpose(1, 2)

            # logs for vis
            a2p_out['logs_amateur_zq'] = a2a_out['z_q']
            a2p_out['logs_prof_zq'] = p2p_out['z_q']

            ret['a2p'] = a2p_out

        return ret