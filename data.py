import sys
import os
import numpy as np
import requests
import random
import urllib
import torch
from torch.autograd import Variable
from transformer import subsequent_mask


class Batch:
    "Object for holding a batch of data with mask during training"
    def __init__(self, src, tgt, pad_idx, device):
        self.src = src.to(device)
        self.src_mask = (src != pad_idx).unsqueeze(-2).to(device)

        if tgt is not None:
            self.tgt = tgt[:, :-1].to(device)
            self.tgt_y = tgt[:, 1:].to(device)
            self.tgt_mask = self.make_std_mask(self.tgt, pad_idx).to(device)
            self.ntokens = (self.tgt_y != pad_idx).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad_idx):
        "Create a mask to hide padding and future words"
        tgt_mask = (tgt != pad_idx).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


DATA_CONFIG = {
    "wmt14": {
        "source_lang": "en",
        "target_lang": "de",
        "url": "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/",
        "files": [
            "dict.en-de", "newstest2012.de", "newstest2012.en",
            "newstest2013.de", "newstest2013.en", "newstest2014.de",
            "newstest2014.en", "newstest2015.de", "newstest2015.en",
            "train.align", "train.de", "train.en",
            "vocab.50K.de", "vocab.50K.en",
        ],
        "train": "train",
        "test": ["newstest2012", "newstest2013", "newstest2014", "newstest2015"],
        "vocab": "vocab.50K",
    }
}

PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3

class WMT14:
    def __init__(self, name: str, data_dir: str = ".data/"):
        assert name in DATA_CONFIG
        self.name = name
        self.config = DATA_CONFIG[name]
        self.src_lang = self.config["source_lang"]
        self.tgt_lang = self.config["target_lang"]
        self.data_dir = os.path.join(os.path.dirname(__file__), data_dir, name)
        os.makedirs(self.data_dir, exist_ok=True)

        self.src_word2idx = None
        self.src_idx2word = None
        self.tgt_word2idx = None
        self.tgt_idx2word = None

    def _download_files(self):
        for file in self.config["files"]:
            download_url = urllib.parse.urljoin(self.config["url"], file)
            filepath = os.path.join(self.data_dir, file)

            if not os.path.exists(filepath):
                data = requests.get(download_url)
                # Save file data to local copy
                with open(filepath, 'wb')as fp:
                    fp.write(data.content)

                print('Successfully downloaded {}'.format(file))

        print("-"*50)
        print("Downloaded files: ")
        for f in os.listdir(self.data_dir):
            print(f)

    def _load_vocab_file(self, filename):
        # first three words in vocab:
        # <unk> : unknown word
        # <s> : start of sentence
        # </s> : end of sentence
        vocab_file = os.path.join(self.data_dir, filename)
        words = list(map(lambda w: w.strip().lower(), open(vocab_file)))
        words.insert(0, '<pad>')
        words = words[:4] + list(set(words[4:]))
        word2id = {word: i for i, word in enumerate(words)}
        id2word = words

        return word2id, id2word

    def load_vocab(self):
        prefix = self.config['vocab']
        self.src_word2idx, self.src_idx2word = self._load_vocab_file(
            prefix + '.' + self.src_lang
        )
        self.tgt_word2idx, self.tgt_idx2word = self._load_vocab_file(
            prefix + '.' + self.tgt_lang
        )
        print(f"{self.src_lang} vocabulary size: {len(self.src_word2idx)}")
        print(f"{self.tgt_lang} vocabulary size: {len(self.tgt_word2idx)}")

    def _sentence_pair_iterator(self, file1, file2, seq_len):
        # discard longer sentences, pad shorter sentences

        def line_count(filename):
            num_lines = int(os.popen(f"wc -l {filename}").read().strip().split()[0])
            return num_lines

        def parse_line(line, word2idx):
            line = line.strip().lower().split()
            word_idxs = [word2idx.get(w, UNK_ID) for w in line]
            word_idxs = [SOS_ID] + word_idxs + [EOS_ID]
            word_idxs += [PAD_ID] * max(0, seq_len - len(word_idxs))
            return word_idxs

        print("Number of lines in {}: {}".format(file1, line_count(file1)))
        assert line_count(file1) == line_count(file2)
        line_pairs = list(zip(open(file1), open(file2)))
        random.shuffle(line_pairs)

        for l1, l2 in line_pairs:
            sentence_1 = parse_line(l1, self.src_word2idx)
            sentence_2 = parse_line(l2, self.tgt_word2idx)
            if len(sentence_1) == len(sentence_2) == seq_len:
                yield sentence_1, sentence_2


    def data_generator(self, batch_size, seq_len, 
                       device, data_type='train', 
                       file_prefix=None, epoch=None):
        # yield a pair of sentences (source, target)
        # each sentence is a list of idxs
        assert data_type in ['train', 'test']
        if self.src_idx2word is None:
            self.load_vocab()

        if file_prefix is None:
            prefixes = self.config[data_type]
            if not isinstance(prefixes, list):
                prefixes = [prefixes]
        else:
            prefixes = [file_prefix]

        batch_src, batch_tgt = [], []
        ep = 0
        while epoch is None or ep < epoch:
            for prefix in prefixes:
                for idxs_src, idxs_tgt in self._sentence_pair_iterator(
                    os.path.join(self.data_dir, prefix + '.' + self.src_lang),
                    os.path.join(self.data_dir, prefix + '.' + self.tgt_lang),
                    seq_len
                ):
                    batch_src.append(idxs_src)
                    batch_tgt.append(idxs_tgt)

                    if len(batch_src) == batch_size:
                        src = torch.from_numpy(np.array(batch_src).copy())
                        tgt = torch.from_numpy(np.array(batch_tgt).copy())
                        src = Variable(src, requires_grad=False)
                        tgt = Variable(tgt, requires_grad=False)

                        yield Batch(src, tgt, PAD_ID, device)

                        batch_src = []
                        batch_tgt = []

            ep += 1

        if len(batch_src) > 0:
            src = torch.from_numpy(np.array(batch_src).copy())
            tgt = torch.from_numpy(np.array(batch_tgt).copy())
            src = Variable(src, requires_grad=False)
            tgt = Variable(tgt, requires_grad=False)
            
            yield Batch(src, tgt, PAD_ID, device)


def decode_sentence(idx2word, sentence: list) -> str:
    sen_l = [idx2word[w] for w in sentence]
    # don't remove <unk>
    sen_l = [w for w in sen_l if w not in ['<s>', '</s>', '<pad>']]
    sen_l = ' '.join(sen_l)
    return sen_l


def main():
    wmt = WMT14("wmt14")
    wmt._download_files()
    data_gen = wmt.data_generator(4, 25)

    for src, tgt in data_gen:
        for i in range(src.shape[0]):
            sen1 = decode_sentence(wmt.src_idx2word, src[i])
            print("source: {}".format(sen1))
            sen2 = decode_sentence(wmt.tgt_idx2word, tgt[i])
            print("target: {}".format(sen2))

        break


if __name__ == "__main__":
    main()
