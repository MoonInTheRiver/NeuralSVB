import matplotlib
matplotlib.use('Agg')

from tasks.tts.dataset_utils import FastSpeechDataset
import torch
from utils.pitch_utils import norm_interp_f0, denorm_f0, f0_to_coarse
import utils


class FastSingingDataset(FastSpeechDataset):
    def __getitem__(self, index):
        sample = super(FastSingingDataset, self).__getitem__(index)
        del sample['txt_token']
        item = self._get_item(index)
        hparams = self.hparams
        max_frames = hparams['max_frames']
        prof_spec = torch.Tensor(item['prof_mel'])[:max_frames]
        max_frames = prof_spec.shape[0] // hparams['frames_multiple'] * hparams['frames_multiple']
        prof_spec = prof_spec[:max_frames]
        prof_energy = (prof_spec.exp() ** 2).sum(-1).sqrt()
        if 'prof_mel2ph' in item:
            prof_mel2ph = torch.LongTensor(item['prof_mel2ph'])[:max_frames]
        else:
            prof_mel2ph = None

        if 'prof_f0' in item:
            if hparams.get('normalize_pitch', False):
                prof_f0 = item["prof_f0"]
                if len(prof_f0 > 0) > 0 and prof_f0[prof_f0 > 0].std() > 0:
                    prof_f0[prof_f0 > 0] = (prof_f0[prof_f0 > 0] - prof_f0[prof_f0 > 0].mean()) / prof_f0[prof_f0 > 0].std() * hparams['f0_std'] + \
                                 hparams['f0_mean']
                    prof_f0[prof_f0 > 0] = prof_f0[prof_f0 > 0].clip(min=60, max=900)
                prof_pitch = f0_to_coarse(prof_f0)
                prof_pitch = torch.LongTensor(prof_pitch[:max_frames])
            else:
                prof_pitch = torch.LongTensor(item.get("prof_pitch"))[:max_frames] if "prof_pitch" in item else None
            prof_f0, prof_uv = norm_interp_f0(item["prof_f0"][:max_frames], hparams)
            prof_uv = torch.FloatTensor(prof_uv)
            prof_f0 = torch.FloatTensor(prof_f0)
        else:
            prof_f0 = prof_uv = torch.zeros_like(prof_mel2ph)
            prof_pitch = None

        sample['prof_mel'] = prof_spec
        sample['prof_energy'] = prof_energy
        sample['prof_pitch'] = prof_pitch
        sample['prof_f0'] = prof_f0
        sample['prof_uv'] = prof_uv
        sample['prof_mel2ph'] = prof_mel2ph
        sample["prof_mel_nonpadding"] = prof_spec.abs().sum(-1) > 0
        return sample

    def collater(self, samples):
        batch = super(FastSingingDataset, self).collater(samples)
        batch['prof_f0'] = utils.collate_1d([s['prof_f0'] for s in samples], 0.0)
        batch['prof_pitch'] = utils.collate_1d([s['prof_pitch'] for s in samples]) if samples[0]['prof_pitch'] is not None else None
        batch['prof_uv'] = utils.collate_1d([s['prof_uv'] for s in samples])
        batch['prof_energy'] = utils.collate_1d([s['prof_energy'] for s in samples], 0.0)
        batch['prof_mel2ph'] = utils.collate_1d([s['prof_mel2ph'] for s in samples], 0.0) \
            if samples[0]['prof_mel2ph'] is not None else None
        batch['prof_mels'] = utils.collate_2d([s['prof_mel'] for s in samples], 0.0)
        batch['prof_mel_lengths'] = torch.LongTensor([s['prof_mel'].shape[0] for s in samples])
        return batch


class MultiSpkEmbDataset(FastSingingDataset):
    def __getitem__(self, index):
        sample = super(MultiSpkEmbDataset, self).__getitem__(index)
        item = self._get_item(index)
        a2p_f0_alignment = torch.LongTensor(item['a2p_f0_alignment'])[:sample['prof_pitch'].shape[0]].clamp(max=sample['pitch'].shape[0]-1)  #  最大值不超过amateur的长度 不然索引炸了.
        p2a_f0_alignment = torch.LongTensor(item['p2a_f0_alignment'])[:sample['pitch'].shape[0]].clamp(max=sample['prof_pitch'].shape[0]-1)  #  最大值不超过prof的长度 不然索引炸了.
        assert a2p_f0_alignment.shape == sample['prof_pitch'].shape, ('a2p F0 alignment with unmatched shape: ', a2p_f0_alignment.shape, sample['prof_pitch'].shape)
        assert p2a_f0_alignment.shape == sample['pitch'].shape, ('p2a F0 alignment with unmatched shape: ', p2a_f0_alignment.shape, sample['pitch'].shape)
        sample['a2p_f0_alignment'] = a2p_f0_alignment
        sample['p2a_f0_alignment'] = p2a_f0_alignment
        # add spk emb
        multi_spk_emb = torch.FloatTensor(item['multi_spk_emb'])
        sample['multi_spk_emb'] = multi_spk_emb
        return sample

    def collater(self, samples):
        batch = super(MultiSpkEmbDataset, self).collater(samples)
        batch['a2p_f0_alignment'] = utils.collate_1d([s['a2p_f0_alignment'] for s in samples])
        batch['p2a_f0_alignment'] = utils.collate_1d([s['p2a_f0_alignment'] for s in samples])
        # add spk emb
        batch['multi_spk_emb'] = utils.collate_2d([s['multi_spk_emb'] for s in samples])
        return batch