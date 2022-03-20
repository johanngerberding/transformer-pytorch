import os
import torch
from torch.autograd import Variable
import pickle
import argparse

from transformer import subsequent_mask
from data import  PAD_ID, SOS_ID, decode_sentence, encode_sentence
from config import get_cfg_defaults


# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')

def greedy_decode(model, src, src_mask, max_len, start_symbol, dataset):
    encoded = model.encode(src, src_mask)

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
    parser = argparse.ArgumentParser("Translate an english to a german sentence.")
    parser.add_argument(
        '--model',
        type=str,
        help='Path to model checkpoint.',
        default="/home/johann/sonstiges/transformer-pytorch/exps/2022-03-19_64_3_100/en-de-model-iter-0000240000.pt",
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config.yaml.',
        default="/home/johann/sonstiges/transformer-pytorch/exps/2022-03-19_64_3_100/config.yaml",
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Path to dataset.file.',
        default="/home/johann/sonstiges/transformer-pytorch/exps/2022-03-19_64_3_100/dataset.file",
    )
    parser.add_argument(
        '--src',
        type=str,
        help='English sentence to translate. If None, 5 train examples are translated.',
    )
    parser.add_argument(
        '--attn',
        type=bool,
        help='Create attention plots.',
        default=True,
    )

    args = parser.parse_args()


    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    with open(args.dataset, 'rb') as fp:
        dataset = pickle.load(fp)

    print(f'dataset directory: {dataset.data_dir}')
    src_vocab_size = len(dataset.src_idx2word)
    tgt_vocab_size = len(dataset.tgt_idx2word)

    model_checkpoint = os.path.join(args.model)
    model = torch.load(model_checkpoint)
    model.eval()
    model.to(DEVICE)

    if args.src:
        encoded_src = encode_sentence(cfg, dataset, args.src)
        encoded_src = torch.tensor(encoded_src).long().unsqueeze(0).to(DEVICE)
        encoded_src_mask = (encoded_src != PAD_ID).unsqueeze(-2).to(DEVICE)

        with torch.no_grad():
            out = greedy_decode(
                model, encoded_src,
                encoded_src_mask, cfg.DATASET.MAX_SEQ_LEN,
                SOS_ID, dataset,
            )

        out = out.detach().cpu().numpy().tolist()[0]
        trans = decode_sentence(dataset.tgt_idx2word, out)

        print("="*75)
        print(f"Source:\t{args.src}")
        print("-"*75)
        print(f"Translation:\t{trans}")


    else:
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
            gt = decode_sentence(dataset.tgt_idx2word, tgt.detach().cpu().numpy().tolist()[0])
            out = out.detach().cpu().numpy().tolist()[0]
            translation = decode_sentence(dataset.tgt_idx2word, out)

            src = batch.src.detach().cpu().numpy().tolist()[0]
            source = decode_sentence(dataset.src_idx2word, src)

            print("="*100)
            print(f"SRC:\t{source}")
            print("-"*100)
            print(f"GT:\t{gt}")
            print("-"*100)
            print(f"PRED:\t{translation}")

            if count == 5:
                break


if __name__ == '__main__':
    main()
