import os
import torch
from torch.autograd import Variable

from transformer import build_model, subsequent_mask
from data import WMT14, PAD_ID, UNK_ID, SOS_ID, EOS_ID, Batch, decode_sentence
from config import get_cfg_defaults


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def encode_sentence(cfg, dataset, sentence: str):
    sentence = sentence.strip().lower().split()
    word_idxs = [dataset.src_word2idx.get(w, UNK_ID) for w in sentence]
    word_idxs = word_idxs + [EOS_ID]
    word_idxs += [PAD_ID] * max(0, cfg.DATASET.MAX_SEQ_LEN - len(word_idxs))
    return word_idxs


def greedy_decode(model, src, src_mask, max_len):
    encoded = model.encode(src, src_mask)
    tgts = torch.ones(1,1).fill_(SOS_ID).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(encoded, src_mask, Variable(tgts), Variable(subsequent_mask(tgts.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])

        if i == 0:
            print(prob.size())
            print(prob[:25])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word[0]
        tgts = torch.cat([tgts, torch.ones(1,1).type_as(src.data).fill_(next_word)], dim=1)

    return tgts


def main():
    cfg = get_cfg_defaults()
    cfg.freeze()
    print(cfg)
    dataset = WMT14('wmt14')
    dataset.load_vocab()
    print(f'dataset directory: {dataset.data_dir}')
    src_vocab_size = len(dataset.src_idx2word)
    tgt_vocab_size = len(dataset.tgt_idx2word)
    model_checkpoint = '/home/johann/sonstiges/transformer-pytorch/checkpoint-630000.pth'
    model = build_model(
        src_vocab_size, tgt_vocab_size,
        N=cfg.MODEL.NUM_LAYERS,
        d_model=cfg.MODEL.D_MODEL,
        d_ff=cfg.MODEL.FFN_DIM,
        h=cfg.MODEL.ATTN_HEADS,
        dropout=cfg.MODEL.DROPOUT,
    )
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    model.to(DEVICE)

    test_sentence = "Hi my name is Johann and I am a Scientific Resercher from Germany."
    encoded_sentence = encode_sentence(cfg, dataset, test_sentence)
    print("Encoded Test sentence: {}".format(encoded_sentence))
    encoded_sentence = torch.tensor(encoded_sentence, dtype=torch.long).to(DEVICE)
    encoded_sentence = Variable(encoded_sentence, requires_grad=False)
    encoded_sentence = encoded_sentence.unsqueeze(0)
    print(encoded_sentence.size())

    batch = Batch(encoded_sentence, None, PAD_ID, DEVICE)
    print(batch.src)
    print(batch.src_mask)

    with torch.no_grad():
        out = greedy_decode(model, batch.src, batch.src_mask, cfg.DATASET.MAX_SEQ_LEN)
    print(out.size())
    print(out)

    out = out.detach().cpu().numpy()
    out = list(out[0])
    print(out)
    translation = decode_sentence(dataset.tgt_idx2word, out)
    print(translation)



if __name__ == '__main__':
    main()
