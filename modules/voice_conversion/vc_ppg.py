from modules.commons.common_layers import *
from modules.fastspeech.fs2 import FS_DECODERS
from modules.voice_conversion.vc_modules import VCASR
from utils.hparams import hparams


class VCPPG(nn.Module):
    def __init__(self, dict_size):
        super().__init__()
        self.hidden_size = hparams['hidden_size']
        self.c_content = self.c_out = hparams['audio_num_mel_bins']
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
        if hparams['use_energy']:
            self.energy_embed = Embedding(256, self.hidden_size, 0)
            self.encoder_proj_in += self.hidden_size
        # ref encoder
        if hparams['use_spk_id']:
            self.spk_embed = Embedding(hparams['num_spk'], hparams['ref_enc_out'])
        else:
            self.ref_encoder = ConvGlobalStacks(
                idim=self.c_content, n_chans=hparams['ref_enc_out'], odim=hparams['ref_enc_out'])
        self.encoder_proj_in += hparams['ref_enc_out']
        self.encoded_embed_proj = Linear(self.encoder_proj_in, self.hidden_size)
        if hparams['ref_attn']:
            self.ref_attn_kv_encoder = ConvStacks(
                idim=self.c_content, n_chans=self.hidden_size, n_layers=5,
                odim=self.hidden_size, strides=[2, 2, 2, 1, 1], res=False, norm='none')
            self.ref_attn = MultiheadAttention(self.hidden_size, 4, encoder_decoder_attention=True)
            self.attn_mask = self.build_attn_mask().long()
        self.decoder = FS_DECODERS[hparams['decoder_type']](hparams)
        self.mel_out = Linear(self.hidden_size, self.c_out, bias=True)

    def forward(self, mels_content=None, mels_timbre=None, pitch=None, energy=None, spk_ids=None):
        ret = {}
        T = pitch.shape[1]
        encoded_embed = []
        # Pitch embedding
        h_pitch = self.pitch_encoder(self.pitch_embed(pitch))
        encoded_embed.append(h_pitch)
        ret['h_pitch'] = h_pitch
        # ASR content embedding
        h_content = self.vc_asr(mels_content)['h_content'].detach()
        h_content = self.upsample_layer(h_content.transpose(1, 2)).transpose(1, 2)[:, :T]
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
        # Decode
        encoded_embed.append(h_style)
        ret['h_style'] = h_style

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

    def run_decoder(self, decoder_inp, tgt_nonpadding):
        x = decoder_inp  # [B, T, H]
        x = self.decoder(x)
        x = self.mel_out(x)
        return x * tgt_nonpadding

    def train_vc_asr(self, mels, tokens):
        prev_tokens = F.pad(tokens[:, :-1], [1, 0], mode='constant', value=0)
        return self.vc_asr(mels, prev_tokens)['tokens']

    def build_attn_mask(self, max_len=3000):
        q_len = max_len
        k_len = max_len // 8
        t = torch.arange(0, q_len)[:, None] - 8 * torch.arange(0, k_len)[None, :]  # y-4x
        return (t < 32) & (t > -32)
