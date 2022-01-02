import torch
import torch.nn as nn
from torch.autograd import Variable
from data import WMT14
from transformer import Transformer, subsequent_mask


def create_padding_mask():
    pass 


def create_autoregressive_mask():
    pass 


def label_smoothing():
    pass 



def main():
    batch_size = 16
    max_seq_len = 100

    dataset = WMT14('wmt14')
    dataset.load_vocab()

    data_gen = dataset.data_generator(batch_size, max_seq_len)
    src_vocab_size = len(dataset.src_idx2word)
    tgt_vocab_size = len(dataset.tgt_idx2word)

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        enc_layers=6,
        dec_layers=6,
        d_model=512,
        dff=2048,
        num_heads=8,
        max_seq_len=100,
        enc_dp=0.1,
        dec_dp=0.1,
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001,
        betas=(0.9, 0.98)
    )
    loss_fn = nn.KLDivLoss(size_average=False)

    model.train()
    for src, tgt in data_gen:
        src = torch.tensor(src)
        tgt = torch.tensor(tgt)
        # remove leading <s> to keep seq len consistent
        src = src[:, 1:]
        
        src_mask = (src != 0).unsqueeze(-2)

        tgt_y = tgt[:, 1:]
        tgt = tgt[:, :-1]

        tgt_mask = (tgt != 0).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))

        optimizer.zero_grad()
        out = model(src, tgt, src_mask, tgt_mask)
        print(out.size())
        print(tgt_y.size())
        
        test_out = out.view()
        test_tgt_y = tgt_y.view()
        test_loss = loss_fn()

        loss = loss_fn(out, tgt_y)
        print(loss)
        loss.backward()

        optimizer.step()



if __name__ == "__main__":
    main()
