from torch import nn
from modules.fastspeech.pe import Prenet
from modules.asr.seq2seq import TransformerASRDecoder
from modules.fastspeech.conformer.conformer import ConformerLayers
from modules.fastspeech.tts_modules import LayerNorm
from modules.fastspeech.wavenet_decoder import WN
from modules.commons.common_layers import ConvStacks, ConvGlobalStacks
from modules.commons.common_layers import Embedding, Linear, SinusoidalPositionalEmbedding, MultiheadAttention
import torch
from utils.hparams import hparams


class PitchPredictor(torch.nn.Module):
    def __init__(self, idim, n_layers=5, n_chans=384, odim=2, kernel_size=5,
                 dropout_rate=0.1, padding='SAME'):
        """Initilize pitch predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        super(PitchPredictor, self).__init__()
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2)
                                       if padding == 'SAME'
                                       else (kernel_size - 1, 0), 0),
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]
        self.linear = torch.nn.Linear(n_chans, odim)
        self.embed_positions = SinusoidalPositionalEmbedding(idim, 0, init_size=4096)
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))

    def forward(self, xs):
        """

        :param xs: [B, T, H]
        :return: [B, T, H]
        """
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
        xs = self.linear(xs.transpose(1, -1))  # (B, Tmax, H)
        return xs


class VCASR(nn.Module):
    def __init__(self, dict_size, n_mel_bins=80):
        super().__init__()
        self.asr_enc_layers = hparams['asr_enc_layers']
        self.asr_dec_layers = hparams['asr_dec_layers']
        self.hidden_size = hparams['asr_hidden_size']
        self.num_heads = 2
        self.mel_prenet = Prenet(n_mel_bins, self.hidden_size, strides=hparams['mel_strides'])
        if hparams['asr_enc_type'] == 'conv':
            self.content_encoder = ConvStacks(
                idim=self.hidden_size, n_chans=self.hidden_size, odim=self.hidden_size)
        elif hparams['asr_enc_type'] == 'conformer':
            self.content_encoder = ConformerLayers(self.hidden_size, self.asr_enc_layers, 31,
                                                   use_last_norm=hparams['asr_last_norm'])
        self.token_embed = Embedding(dict_size, self.hidden_size, 0)
        self.asr_decoder = TransformerASRDecoder(
            self.hidden_size, self.asr_dec_layers, hparams['dropout'], dict_size,
            num_heads=self.num_heads)

    def forward(self, mel_input, prev_tokens=None):
        ret = {}
        h_content = ret['h_content'] = self.content_encoder(self.mel_prenet(mel_input)[1])
        if prev_tokens is not None:
            ret['tokens'], ret['asr_attn'] = self.asr_decoder(self.token_embed(prev_tokens), h_content)
        return ret


class VCPitch3(nn.Module):
    def __init__(self, c_content=None, c_out=None, mel_content_dim=None):
        super().__init__()
        self.hidden_size = hparams['hidden_size']
        self.c_content = self.c_out = hparams['audio_num_mel_bins']
        self.encoder_proj_in = 0
        self.spk_indep_proj_in = 0
        self.pitch_indep_proj_in = 0
        if c_content is not None:
            self.c_content = c_content
        if c_out is not None:
            self.c_out = c_out
        # pitch encoder
        self.pitch_embed = Embedding(300, self.hidden_size, 0)
        self.pitch_encoder = ConvStacks(
            idim=self.hidden_size, n_chans=self.hidden_size, odim=self.hidden_size, n_layers=3)
        self.encoder_proj_in += self.hidden_size
        self.spk_indep_proj_in += self.hidden_size
        # asr content encoder
        if hparams['asr_content_encoder']:
            if hparams['asr_upsample_norm'] == 'bn':
                self.upsample_layer = nn.Sequential(
                    *([nn.Sequential(
                        nn.Upsample(scale_factor=scale, mode='nearest'),
                        nn.Conv1d(self.hidden_size, self.hidden_size, scale * 2 + 1, padding=scale),
                        nn.ReLU(), nn.BatchNorm1d(self.hidden_size))
                          for scale in hparams['mel_strides'] if scale > 1
                      ] + [nn.Conv1d(self.hidden_size, self.hidden_size, 5, padding=2)]))
            elif hparams['asr_upsample_norm'] == 'gn':
                self.upsample_layer = nn.Sequential(
                    *([nn.Sequential(
                        nn.Upsample(scale_factor=scale, mode='nearest'),
                        nn.Conv1d(self.hidden_size, self.hidden_size, scale * 2 + 1, padding=scale),
                        nn.ReLU(), nn.GroupNorm(8, self.hidden_size))
                          for scale in hparams['mel_strides'] if scale > 1
                      ] + [nn.Conv1d(self.hidden_size, self.hidden_size, 5, padding=2)]))
            else:
                self.upsample_layer = nn.Sequential(
                    *([nn.Sequential(
                        nn.Upsample(scale_factor=scale, mode='nearest'),
                        nn.Conv1d(self.hidden_size, self.hidden_size, scale * 2 + 1, padding=scale),
                        nn.ReLU())
                          for scale in hparams['mel_strides'] if scale > 1
                      ] + [nn.Conv1d(self.hidden_size, self.hidden_size, 5, padding=2)]))

            self.encoder_proj_in += self.hidden_size
            self.spk_indep_proj_in += self.hidden_size
            self.pitch_indep_proj_in += self.hidden_size
        # mel content encoder
        if hparams['use_energy']:
            self.energy_embed = Embedding(256, self.hidden_size, 0)
            self.encoder_proj_in += self.hidden_size
        if hparams['mel_content_encoder']:
            mel_content_dim = hparams['mel_content_dim'] if mel_content_dim is None else mel_content_dim
            self.mel_content_encoder = ConvStacks(
                idim=self.c_content, n_chans=self.hidden_size, n_layers=5, odim=mel_content_dim)
            self.encoder_proj_in += mel_content_dim
        # ref encoder
        self.ref_encoder = ConvGlobalStacks(
            idim=self.c_content, n_chans=hparams['ref_enc_out'], odim=hparams['ref_enc_out'])
        self.encoder_proj_in += hparams['ref_enc_out']

        self.encoded_embed_proj = Linear(self.encoder_proj_in, self.hidden_size)
        self.spk_indep_proj = Linear(self.spk_indep_proj_in, self.hidden_size)
        if hparams['ref_attn']:
            self.ref_attn_kv_encoder = ConvStacks(
                idim=self.c_content, n_chans=self.hidden_size, n_layers=5,
                odim=self.hidden_size, strides=[2, 2, 2, 1, 1], res=False, norm='none')
            self.ref_attn = MultiheadAttention(self.hidden_size, 4, encoder_decoder_attention=True)
            self.attn_mask = self.build_attn_mask().long()
        if hparams['decoder_type'] == 'conv':
            self.mel_decoder = ConvStacks(idim=self.hidden_size, n_chans=self.hidden_size,
                                          n_layers=hparams['dec_layers'], odim=self.c_out,
                                          dropout=hparams['dropout'], norm=hparams['norm_type'])
        elif hparams['decoder_type'] == 'wn':
            self.mel_decoder = nn.Sequential(
                WN(self.hidden_size), nn.Linear(self.hidden_size, self.c_out))
        # domain adv
        if hparams['pitch_domain_adv']:
            self.pitch_indep_proj = Linear(self.pitch_indep_proj_in, self.hidden_size)
            self.da_pitch_predictor = ConvStacks(idim=self.hidden_size, n_chans=self.hidden_size, odim=1)

    def forward(self, mel_input, h_content, pitch, energy=None, infer=False, **kwargs):
        ret = {}
        T = pitch.shape[1]
        encoded_embed = []
        pitch_indep_embed = []
        # Pitch embedding
        h_pitch = self.pitch_encoder(self.pitch_embed(pitch))
        encoded_embed.append(h_pitch)
        # ASR content embedding
        h_content = self.upsample_layer(h_content.transpose(1, 2)).transpose(1, 2)[:, :T]
        encoded_embed.append(h_content)
        pitch_indep_embed.append(h_content)
        if hparams['use_energy']:
            energy = torch.clamp(energy * 256 // 4, max=255).long()
            energy_embed = self.energy_embed(energy)
            encoded_embed.append(energy_embed)
            pitch_indep_embed.append(energy_embed)
        # Mel content embedding
        if hparams['mel_content_encoder']:
            h_mel_content = self.mel_content_encoder(mel_input)
            encoded_embed.append(h_mel_content)
            pitch_indep_embed.append(h_mel_content)

        # Ref embedding
        h_ref = self.ref_encoder(mel_input)[:, None, :].repeat(1, T, 1)
        # Decode
        encoded_embed.append(h_ref)
        ret['dec_inputs'] = dec_inputs = self.encoded_embed_proj(torch.cat(encoded_embed, -1))
        if hparams['ref_attn']:
            h_mel_content = self.ref_attn_kv_encoder(mel_input).transpose(0, 1)
            q_len, kv_len = mel_input.shape[1], h_mel_content.shape[0]
            attn_mask = self.attn_mask.to(mel_input.device)[:q_len, :kv_len] * -1e9
            attn, _ = self.ref_attn(dec_inputs.transpose(0, 1), h_mel_content, h_mel_content,
                                    attn_mask=attn_mask)
            dec_inputs = dec_inputs + attn.transpose(0, 1)

        nonpadding = pitch > 0
        ret['mel_out'] = self.run_decoder(dec_inputs, nonpadding, ret, infer=infer, **kwargs)
        # Domain Adv
        if hparams['pitch_domain_adv']:
            pitch_indep_embed = self.pitch_indep_proj(torch.cat(pitch_indep_embed, -1))
            ret['f0_pred'] = self.da_pitch_predictor(pitch_indep_embed)[:, :, 0]
        return ret

    def out2mel(self, out):
        return out

    def run_decoder(self, x, nonpadding, ret, infer, **kwargs):
        mel_out = self.mel_decoder(x)
        return mel_out

    def build_attn_mask(self, max_len=3000):
        q_len = max_len
        k_len = max_len // 8
        t = torch.arange(0, q_len)[:, None] - 8 * torch.arange(0, k_len)[None, :]  # y-4x
        return (t < 32) & (t > -32)
