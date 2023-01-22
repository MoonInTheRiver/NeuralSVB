from modules.commons.common_layers import *
from modules.voice_conversion.vc_modules import VCASR
from utils.hparams import hparams
from modules.voice_conversion.vc_ppg import VCPPG
import numpy as np


class SVBPPG(VCPPG):
    def __init__(self, dict_size):
        super().__init__(dict_size)
        if hparams['use_tech']:
            self.tech_embed = Embedding(hparams['num_techs'], self.hidden_size)
            self.encoder_proj_in += self.hidden_size
        self.encoded_embed_proj = Linear(self.encoder_proj_in, self.hidden_size)

    def forward(self, mels_content=None, mels_timbre=None, pitch=None, energy=None, spk_ids=None, tech_ids=None):
        ret = {}
        T = pitch.shape[1]
        encoded_embed = []
        # Pitch embedding
        h_pitch = self.pitch_encoder(self.pitch_embed(pitch))
        encoded_embed.append(h_pitch)
        ret['h_pitch'] = h_pitch
        # ASR content embedding
        h_content = self.vc_asr(mels_content)['h_content'].detach()
        h_content = self.upsample_layer(h_content.transpose(1, 2)).transpose(1, 2)  # [:, :T]
        # align h_content to pitch length
        if hparams['interpo_ppg']:
            h_content = F.interpolate(h_content.transpose(1, 2), size=T).transpose(1, 2)  # [B, T, H]

        encoded_embed.append(h_content)
        ret['h_content'] = h_content
        if hparams['use_energy']:
            energy = torch.clamp(energy * 256 // 4, max=255).long()
            h_energy = self.energy_embed(energy)
            encoded_embed.append(h_energy)
            ret['h_energy'] = h_energy
        # Ref embedding
        if hparams['use_spk_id']:
            h_style = self.spk_embed(spk_ids)[:, None, :].repeat(1, T, 1)
        else:
            h_style = self.ref_encoder(mels_timbre)[:, None, :].repeat(1, T, 1)
        encoded_embed.append(h_style)
        # tech embedding
        if hparams['use_tech']:
            h_tech = self.tech_embed(tech_ids)[:, None, :].repeat(1, T, 1)
            encoded_embed.append(h_tech)
        # ret['h_style'] = h_style

        ret['dec_inputs'] = dec_inputs = self.encoded_embed_proj(torch.cat(encoded_embed, -1))
        if hparams['ref_attn'] and not hparams['use_spk_id']:
            kv_refattn_timbre = self.ref_attn_kv_encoder(mels_timbre).transpose(0, 1)
            q_len, kv_len = dec_inputs.shape[1], kv_refattn_timbre.shape[0]
            attn_mask = self.attn_mask.to(mels_timbre.device)[:q_len, :kv_len] * -1e9
            attn, _ = self.ref_attn(dec_inputs.transpose(0, 1), kv_refattn_timbre, kv_refattn_timbre,
                                    attn_mask=attn_mask)
            dec_inputs = dec_inputs + attn.transpose(0, 1)
        nonpadding = (pitch > 0).float()[:, :, None]
        ret['mel_out'] = self.run_decoder(dec_inputs, nonpadding)
        return ret


class ParaSVBPPG(SVBPPG):
    def forward(self, mels_content=None, mels_timbre=None, pitch=None, energy=None, spk_ids=None, tech_ids=None,
                conversion_alignment=None, infer=False):
        ret = {}
        T = pitch.shape[1]
        encoded_embed = []
        # Pitch embedding
        h_pitch = self.pitch_encoder(self.pitch_embed(pitch))
        encoded_embed.append(h_pitch)
        ret['h_pitch'] = h_pitch
        # ASR content embedding
        h_content = self.vc_asr(mels_content)['h_content'].detach()
        h_content = self.upsample_layer(h_content.transpose(1, 2)).transpose(1, 2)[:,
                    :mels_content.shape[1]]  # [B, S, H]
        # align h_content to pitch length
        if conversion_alignment is not None:
            conversion_alignment = conversion_alignment[:, :, None].repeat(1, 1, self.hidden_size)  # [B, S, H]
            h_content = torch.gather(h_content, 1, conversion_alignment)  # [B, T, H]
        encoded_embed.append(h_content)
        ret['h_content'] = h_content
        if hparams['use_energy']:
            energy = torch.clamp(energy * 256 // 4, max=255).long()
            h_energy = self.energy_embed(energy)
            encoded_embed.append(h_energy)
            ret['h_energy'] = h_energy
        # Ref embedding
        if hparams['use_spk_id']:
            h_style = self.spk_embed(spk_ids)[:, None, :].repeat(1, T, 1)
        else:
            h_style = spk_ids[:, [0], :].repeat(1, T, 1)
            # h_style = self.ref_encoder(mels_timbre)[:, None, :].repeat(1, T, 1)
        ret['h_style'] = h_style
        encoded_embed.append(h_style)
        # tech embedding
        if hparams['use_tech']:
            h_tech = self.tech_embed(tech_ids)[:, None, :].repeat(1, T, 1)
            encoded_embed.append(h_tech)
        # ret['h_style'] = h_style

        ret['dec_inputs'] = dec_inputs = self.encoded_embed_proj(torch.cat(encoded_embed, -1))

        if hparams['ref_attn'] and not hparams['use_spk_id']:
            kv_refattn_timbre = self.ref_attn_kv_encoder(mels_timbre).transpose(0, 1)
            q_len, kv_len = dec_inputs.shape[1], kv_refattn_timbre.shape[0]
            attn_mask = self.attn_mask.to(mels_timbre.device)[:q_len, :kv_len] * -1e9
            attn, _ = self.ref_attn(dec_inputs.transpose(0, 1), kv_refattn_timbre, kv_refattn_timbre,
                                    attn_mask=attn_mask)
            dec_inputs = dec_inputs + attn.transpose(0, 1)
        nonpadding = (pitch > 0).float()[:, :, None]
        ret['mel_out'] = self.run_decoder(dec_inputs, nonpadding)
        ret['h_style_out'] = self.ref_encoder(ret['mel_out'])[:, None, :].repeat(1, T, 1)
        return ret


class ParaSVCAttnPPG(ParaSVBPPG):
    def __init__(self, dict_size):
        super().__init__(dict_size)
        # define attention
        num_heads = 4
        self.ppg_attn = MultiheadAttention(self.hidden_size, num_heads, encoder_decoder_attention=True)

    def forward(self, mels_content=None, mels_timbre=None, pitch=None, energy=None, spk_ids=None, tech_ids=None,
                conversion_alignment=None):
        ret = {}
        T = pitch.shape[1]
        encoded_embed = []
        # Pitch embedding
        h_pitch = self.pitch_encoder(self.pitch_embed(pitch))
        encoded_embed.append(h_pitch)
        ret['h_pitch'] = h_pitch
        # ASR content embedding
        raw_h_content = self.vc_asr(mels_content)['h_content'].detach()
        raw_h_content = self.upsample_layer(raw_h_content.transpose(1, 2)).transpose(1, 2)[:,
                        :mels_content.shape[1]]  # [B, S, H]
        # align h_content to pitch length
        if conversion_alignment is not None:
            conversion_alignment = conversion_alignment[:, :, None].repeat(1, 1, self.hidden_size)  # [B, S, H]
            aligned_h_content = torch.gather(raw_h_content, 1, conversion_alignment)  # [B, T, H]
            attn, _ = self.ppg_attn(aligned_h_content.transpose(0, 1), raw_h_content.transpose(0, 1),
                                    raw_h_content.transpose(0, 1),
                                    attn_mask=None)
            h_content = attn.transpose(0, 1)
        else:
            attn, _ = self.ppg_attn(raw_h_content.transpose(0, 1), raw_h_content.transpose(0, 1),
                                    raw_h_content.transpose(0, 1),
                                    attn_mask=None)
            h_content = attn.transpose(0, 1)

        encoded_embed.append(h_content)
        ret['h_content'] = h_content
        if hparams['use_energy']:
            energy = torch.clamp(energy * 256 // 4, max=255).long()
            h_energy = self.energy_embed(energy)
            encoded_embed.append(h_energy)
            ret['h_energy'] = h_energy
        # Ref embedding
        if hparams['use_spk_id']:
            h_style = self.spk_embed(spk_ids)[:, None, :].repeat(1, T, 1)
        else:
            h_style = self.ref_encoder(mels_timbre)[:, None, :].repeat(1, T, 1)
        encoded_embed.append(h_style)
        # tech embedding
        if hparams['use_tech']:
            h_tech = self.tech_embed(tech_ids)[:, None, :].repeat(1, T, 1)
            encoded_embed.append(h_tech)
        ret['h_style'] = h_style
        ret['h_tech'] = h_tech

        ret['dec_inputs'] = dec_inputs = self.encoded_embed_proj(torch.cat(encoded_embed, -1))
        if hparams['ref_attn'] and not hparams['use_spk_id']:
            kv_refattn_timbre = self.ref_attn_kv_encoder(mels_timbre).transpose(0, 1)
            q_len, kv_len = dec_inputs.shape[1], kv_refattn_timbre.shape[0]
            attn_mask = self.attn_mask.to(mels_timbre.device)[:q_len, :kv_len] * -1e9
            attn, _ = self.ref_attn(dec_inputs.transpose(0, 1), kv_refattn_timbre, kv_refattn_timbre,
                                    attn_mask=attn_mask)
            dec_inputs = dec_inputs + attn.transpose(0, 1)
        nonpadding = (pitch > 0).float()[:, :, None]
        ret['mel_out'] = self.run_decoder(dec_inputs, nonpadding)
        return ret


# 下面都是废案 .


class ParaPPGPreExp(ParaSVBPPG):
    def forward(self, mels_content=None, mels_timbre=None, pitch=None, energy=None, spk_ids=None, tech_ids=None,
                conversion_alignment=None):
        ret = {}
        T = pitch.shape[1]
        encoded_embed = []
        # Pitch embedding
        h_pitch = self.pitch_encoder(self.pitch_embed(pitch))
        encoded_embed.append(h_pitch)
        ret['h_pitch'] = h_pitch
        # align mel_content to pitch length
        if conversion_alignment is not None:
            conversion_alignment = conversion_alignment[:, :, None].repeat(1, 1,
                                                                           hparams['audio_num_mel_bins'])  # [B, S, 80]
            mels_content = torch.gather(mels_content, 1, conversion_alignment)  # [B, T, 80]
        # ASR content embedding
        h_content = self.vc_asr(mels_content)['h_content'].detach()
        h_content = self.upsample_layer(h_content.transpose(1, 2)).transpose(1, 2)[:,
                    :mels_content.shape[1]]  # [B, S, H]
        encoded_embed.append(h_content)
        ret['h_content'] = h_content
        if hparams['use_energy']:
            energy = torch.clamp(energy * 256 // 4, max=255).long()
            h_energy = self.energy_embed(energy)
            encoded_embed.append(h_energy)
            ret['h_energy'] = h_energy
        # Ref embedding
        if hparams['use_spk_id']:
            h_style = self.spk_embed(spk_ids)[:, None, :].repeat(1, T, 1)
        else:
            h_style = self.ref_encoder(mels_timbre)[:, None, :].repeat(1, T, 1)
        encoded_embed.append(h_style)
        # tech embedding
        if hparams['use_tech']:
            h_tech = self.tech_embed(tech_ids)[:, None, :].repeat(1, T, 1)
            encoded_embed.append(h_tech)
        # ret['h_style'] = h_style
        ret['dec_inputs'] = dec_inputs = self.encoded_embed_proj(torch.cat(encoded_embed, -1))
        if hparams['ref_attn'] and not hparams['use_spk_id']:
            kv_refattn_timbre = self.ref_attn_kv_encoder(mels_timbre).transpose(0, 1)
            q_len, kv_len = dec_inputs.shape[1], kv_refattn_timbre.shape[0]
            attn_mask = self.attn_mask.to(mels_timbre.device)[:q_len, :kv_len] * -1e9
            attn, _ = self.ref_attn(dec_inputs.transpose(0, 1), kv_refattn_timbre, kv_refattn_timbre,
                                    attn_mask=attn_mask)
            dec_inputs = dec_inputs + attn.transpose(0, 1)
        nonpadding = (pitch > 0).float()[:, :, None]
        ret['mel_out'] = self.run_decoder(dec_inputs, nonpadding)
        return ret

    def train_vc_asr(self, mels, tokens, conversion_alignment=None):
        if conversion_alignment is not None:
            conversion_alignment = conversion_alignment[:, :, None].repeat(1, 1,
                                                                           hparams['audio_num_mel_bins'])  # [B, S, 80]
            mels = torch.gather(mels, 1, conversion_alignment)
        prev_tokens = F.pad(tokens[:, :-1], [1, 0], mode='constant', value=0)
        return self.vc_asr(mels, prev_tokens)['tokens']


#### Para aligned ppg: 把ppg复制4份，对齐，mean pool下来，再送给网络（作为content ppg，以及decoder input）.


class AlignedVCASR(VCASR):
    def forward(self, mel_input, prev_tokens=None, conversion_alignment=None):
        ret = {}
        tmp_content = self.content_encoder(self.mel_prenet(mel_input)[1])
        if conversion_alignment is not None:
            tmp_content = tmp_content.unsqueeze(2)  # [B, S // scale, 1, H]
            B, s, _, H = tmp_content.shape
            scale = np.prod(hparams['mel_strides'])
            tmp_content = tmp_content.repeat(1, 1, scale, 1).reshape(B, s * scale,
                                                                     H)  # [B, S // strides, scale, H] -> [B, S, H]
            conversion_alignment = conversion_alignment[:, :, None].repeat(1, 1, self.hidden_size)  # [B, S, H]
            tmp_content = torch.gather(tmp_content, 1, conversion_alignment)  # [B, T, H]
            tmp_content = F.pad(tmp_content, pad=(0, 0, 0, scale))
            tmp_content = F.avg_pool1d(tmp_content.transpose(1, 2), kernel_size=scale, stride=scale).transpose(1,
                                                                                                               2)  # [B, T // scale, H]
        h_content = ret['h_content'] = tmp_content
        if prev_tokens is not None:
            ret['tokens'], ret['asr_attn'] = self.asr_decoder(self.token_embed(prev_tokens), h_content)
        return ret


class ParaAlignedPPG(ParaSVBPPG):
    def __init__(self, dict_size):
        super().__init__(dict_size)
        self.vc_asr = AlignedVCASR(dict_size, self.c_content)

    def forward(self, mels_content=None, mels_timbre=None, pitch=None, energy=None, spk_ids=None, tech_ids=None,
                conversion_alignment=None):
        ret = {}
        T = pitch.shape[1]
        encoded_embed = []
        # Pitch embedding
        h_pitch = self.pitch_encoder(self.pitch_embed(pitch))
        encoded_embed.append(h_pitch)
        ret['h_pitch'] = h_pitch
        # ASR content embedding
        h_content = self.vc_asr(mels_content, conversion_alignment=conversion_alignment)[
            'h_content'].detach()  # [B, T', H]
        h_content = self.upsample_layer(h_content.transpose(1, 2)).transpose(1, 2)[:, :T]  # [B, T, H]
        encoded_embed.append(h_content)
        ret['h_content'] = h_content
        if hparams['use_energy']:
            energy = torch.clamp(energy * 256 // 4, max=255).long()
            h_energy = self.energy_embed(energy)
            encoded_embed.append(h_energy)
            ret['h_energy'] = h_energy
        # Ref embedding
        if hparams['use_spk_id']:
            h_style = self.spk_embed(spk_ids)[:, None, :].repeat(1, T, 1)
        else:
            h_style = self.ref_encoder(mels_timbre)[:, None, :].repeat(1, T, 1)
        encoded_embed.append(h_style)
        # tech embedding
        if hparams['use_tech']:
            h_tech = self.tech_embed(tech_ids)[:, None, :].repeat(1, T, 1)
            encoded_embed.append(h_tech)
        # ret['h_style'] = h_style
        ret['dec_inputs'] = dec_inputs = self.encoded_embed_proj(torch.cat(encoded_embed, -1))
        if hparams['ref_attn'] and not hparams['use_spk_id']:
            kv_refattn_timbre = self.ref_attn_kv_encoder(mels_timbre).transpose(0, 1)
            q_len, kv_len = dec_inputs.shape[1], kv_refattn_timbre.shape[0]
            attn_mask = self.attn_mask.to(mels_timbre.device)[:q_len, :kv_len] * -1e9
            attn, _ = self.ref_attn(dec_inputs.transpose(0, 1), kv_refattn_timbre, kv_refattn_timbre,
                                    attn_mask=attn_mask)
            dec_inputs = dec_inputs + attn.transpose(0, 1)
        nonpadding = (pitch > 0).float()[:, :, None]
        ret['mel_out'] = self.run_decoder(dec_inputs, nonpadding)
        return ret

    def train_vc_asr(self, mels, tokens, conversion_alignment):
        prev_tokens = F.pad(tokens[:, :-1], [1, 0], mode='constant', value=0)
        return self.vc_asr(mels, prev_tokens, conversion_alignment)['tokens']


class ParaPPGConstraint(ParaAlignedPPG):
    def train_vc_asr(self, mels, tokens, conversion_alignment=None):
        prev_tokens = F.pad(tokens[:, :-1], [1, 0], mode='constant', value=0)
        asr_output = self.vc_asr(mels, prev_tokens, conversion_alignment=conversion_alignment)
        h_content = asr_output['h_content']  # with gradient
        out_tokens = asr_output['tokens']
        return out_tokens, h_content

