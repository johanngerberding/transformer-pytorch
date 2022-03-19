import os
import torch
from torch.autograd import Variable
import pickle

from transformer import build_model, subsequent_mask
from data import WMT14, PAD_ID, UNK_ID, SOS_ID, EOS_ID, Batch, decode_sentence
from config import get_cfg_defaults


# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')

def encode_sentence(cfg, dataset, sentence: str):
    sentence = sentence.strip().lower().split()
    word_idxs = [dataset.src_word2idx.get(w, UNK_ID) for w in sentence]
    word_idxs = word_idxs + [EOS_ID]
    word_idxs += [PAD_ID] * max(0, cfg.DATASET.MAX_SEQ_LEN - len(word_idxs))
    return word_idxs


def greedy_decode(model, src, src_mask, max_len, start_symbol, dataset):
    encoded = model.encode(src, src_mask)

    src_ = src.detach().cpu().numpy().tolist()[0]
    source = decode_sentence(dataset.src_idx2word, src_)
    print(f"Source sentence: {source}")

    tgts = torch.ones(1,1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            encoded, src_mask, Variable(tgts),
            Variable(subsequent_mask(tgts.size(1)).type_as(src.data)))

        prob = model.generator(out[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        tgts = torch.cat([tgts,
                          torch.ones(1,1).type_as(src.data).fill_(next_word)], dim=1)

    return tgts


def main():
    exp_root = "/home/johann/sonstiges/transformer-pytorch/exps/2022-03-19_64_3_100"

    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join(exp_root, 'config.yaml'))
    cfg.freeze()

    with open(os.path.join(exp_root, 'dataset.file'), 'rb') as fp:
        dataset = pickle.load(fp)

    # dataset = WMT14('wmt14')
    # dataset.load_vocab()

    print(f'dataset directory: {dataset.data_dir}')
    src_vocab_size = len(dataset.src_idx2word)
    tgt_vocab_size = len(dataset.tgt_idx2word)
    print(dataset.src_idx2word[:15])
    print(dataset.tgt_idx2word[:15])

    print(dataset.src_lang)
    print(dataset.tgt_lang)

    model_checkpoint = os.path.join(exp_root, "en-de-model-iter-0000010000.pt")
    model = torch.load(model_checkpoint)
    model.eval()
    model.to(DEVICE)

    data_gen = dataset.data_generator(1,
                                      cfg.DATASET.MAX_SEQ_LEN,
                                      DEVICE,
                                      data_type='train')

    count = 0
    for batch in data_gen:
        count += 1
        with torch.no_grad():
            out = greedy_decode(
                model, batch.src, batch.src_mask,
                cfg.DATASET.MAX_SEQ_LEN, SOS_ID, dataset)

        tgt = batch.tgt
        #print(tgt)
        gt = decode_sentence(dataset.tgt_idx2word, tgt.detach().cpu().numpy().tolist()[0])
        print(f"GT: {gt}")
        out = out.detach().cpu().numpy().tolist()[0]
        translation = decode_sentence(dataset.tgt_idx2word, out)
        print(translation)
        print("--"*30)
        if count == 5:
            break

    """    test_sentence = "Hi my name is Johann and I am a Scientific Resercher from Germany."
    encoded_sentence = encode_sentence(cfg, dataset, test_sentence)
    print("Encoded Test sentence: {}".format(encoded_sentence))
    encoded_sentence = torch.tensor(encoded_sentence, dtype=torch.long).to(DEVICE)
    encoded_sentence = Variable(encoded_sentence, requires_grad=False)
    encoded_sentence = encoded_sentence.unsqueeze(0)
    print(encoded_sentence.size())

    batch = Batch(encoded_sentence, None, PAD_ID, DEVICE)
    print(batch.src)
    print(batch.src_mask)

    print(out.size())
    print(out)

    out = out.detach().cpu().numpy()
    out = list(out[0])
    print(out)
    print(translation)
    """



if __name__ == '__main__':
    main()
