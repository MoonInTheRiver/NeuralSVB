from data_gen.tts.base_binarizer import BaseBinarizer
import re
import json
import os
from copy import deepcopy
import logging
from utils.hparams import hparams


def split_train_test_set(item_names):
    item_names = deepcopy(item_names)
    test_item_names = [x for x in item_names if any([ts in x for ts in hparams['test_prefixes']])]
    train_item_names = [x for x in item_names if x not in set(test_item_names)]
    logging.info("train {}".format(len(train_item_names)))
    logging.info("test {}".format(len(test_item_names)))
    return train_item_names, test_item_names


class SingingBinarizer(BaseBinarizer):
    def load_meta_data(self):
        super().load_meta_data()
        new_item_names = []
        n_utt_ds = {k: 0 for k in hparams['datasets']}
        for item_name in self.item_names:
            for dataset in hparams['datasets']:
                if len(re.findall(rf'{dataset}', item_name)) > 0:
                    new_item_names.append(item_name)
                    n_utt_ds[dataset] += 1
                    break
        self.item_names = new_item_names
        self._train_item_names, self._test_item_names = split_train_test_set(self.item_names)

    @property
    def train_item_names(self):
        return self._train_item_names

    @property
    def valid_item_names(self):
        return self._test_item_names
    @property
    def test_item_names(self):
        return self._test_item_names

    def process(self):
        self.load_meta_data()
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)
        self.spk_map = self.build_spk_map()
        print("| spk_map: ", self.spk_map)
        spk_map_fn = f"{hparams['binary_data_dir']}/spk_map.json"
        json.dump(self.spk_map, open(spk_map_fn, 'w'))

        self.process_data('valid')
        self.process_data('test')
        self.process_data('train')


if __name__ == "__main__":
    SingingBinarizer().process()
