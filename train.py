import os
from datetime import date
import argparse
import shutil
import time
import glob
import logging
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from data import WMT14, Batch, decode_sentence, SOS_ID
from transformer import subsequent_mask, build_model
from config import get_cfg_defaults

global max_src_in_batch, max_tgt_in_batch


class LabelSmoothing(nn.Module):
    def __init__(
            self,
            size: int,
            padding_idx: int,
            smoothing: float = 0.0
    ):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, pred, target):
        assert pred.size(1) == self.size
        true_dist = pred.data.clone()  # copy data
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(pred, Variable(true_dist, requires_grad=False))


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(),
                                    lr=0, betas=(0.9, 0.98), eps=1e-9))


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding"
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def example_data_gen(V: int, batch: int, nbatches: int):
    "Generate random data for a src-tgt copy task"
    for i in range(nbatches):
        data = torch.randint(1, V, (batch, 10))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)


def run_epoch(data_iter, model: nn.Module, loss_compute,
              epoch: int, writer, experiment_dir: str, cfg, dataset) -> float:
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt,
                            batch.src_mask, batch.tgt_mask)

        loss = loss_compute(out, batch.tgt_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 100 == 0:
            elapsed = time.time() - start
            print("Epoch step: {} loss: {} tokens per second: {}"
                  .format(i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0

        if i % 1000 == 0:
            if loss_compute == None:
                writer.add_scalar("loss/val", loss.item(), epoch * i)
            else:
                writer.add_scalar("loss/train", loss.item(), epoch * i)

        if i % 10000 == 0:
            if loss_compute is not None:
                torch.save(model, os.path.join(experiment_dir,
                                        "en-de-model-iter-{}.pt".format(str(epoch * i).zfill(10))))
                #torch.save(model.state_dict(),
                       # os.path.join(experiment_dir,
                                        # "iter_{}.pth".format(str(epoch * i).zfill(10))))

            # print a few translation examples
            for j in range(10):
                with torch.no_grad():
                    out = greedy_decode(
                        model, batch.src[j].unsqueeze(0), batch.src_mask[j].unsqueeze(0),
                        cfg.DATASET.MAX_SEQ_LEN, SOS_ID)
                # out -> batch_size, max_len
                out = out.detach().cpu().numpy().tolist()
                trans = decode_sentence(dataset.tgt_idx2word, out[0])
                source = decode_sentence(dataset.src_idx2word,
                                         batch.src[j].detach().cpu().numpy().tolist())
                logging.info("=="*50)
                logging.info(f"SRC:\t{source}")
                logging.info("--"*50)
                logging.info(f"TRANS:\t{trans}")

    return total_loss / total_tokens


class LossCompute:
    "A simple loss compute and train function"
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return loss.item() * norm


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data)
                        .fill_(next_word)], dim=1)

    return ys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help="Path to customized experiment.yaml",
                        type=str,
                        default=None)
    parser.add_argument('--out',
                        help="Path to output directory",
                        type=str,
                        default=None)
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    if args.config:
        cfg.merge_from_file(args.config)
    cfg.freeze()
    print(cfg)

    if args.out is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        today = date.today().strftime("%Y-%m-%d")
        experiment_dir = "exps/{}_{}_{}_{}".format(today,
                                                        str(cfg.TRAIN.BATCH_SIZE),
                                                        str(cfg.TRAIN.N_EPOCHS),
                                                        str(cfg.DATASET.MAX_SEQ_LEN))

        experiment_dir = os.path.join(dir_path, experiment_dir)
        if os.path.isdir(experiment_dir):
            shutil.rmtree(experiment_dir)
        os.makedirs(experiment_dir, exist_ok=False)
    else:
        os.makedirs(args.out, exist_ok=False)
        experiment_dir = args.out

    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as fp:
        cfg.dump()

    logging.basicConfig(filename=os.path.join(experiment_dir, 'train.log'),
                        level=logging.DEBUG)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training device: {device}")

    writer = SummaryWriter()
    dataset = WMT14('wmt14')
    if len(os.listdir(dataset.data_dir)) <= 0:
        dataset._download_files()

    dataset.load_vocab()

    with open(os.path.join(experiment_dir, 'dataset.file'), 'wb') as fp:
        pickle.dump(dataset, fp, pickle.HIGHEST_PROTOCOL)

    data_gen = dataset.data_generator(cfg.TRAIN.BATCH_SIZE,
                                      cfg.DATASET.MAX_SEQ_LEN,
                                      device,
                                      data_type='train')
    data_gen_val = dataset.data_generator(cfg.VAL.BATCH_SIZE,
                                          cfg.DATASET.MAX_SEQ_LEN,
                                          device,
                                          data_type='test')

    src_vocab_size = len(dataset.src_idx2word)
    tgt_vocab_size = len(dataset.tgt_idx2word)

    criterion = LabelSmoothing(size=tgt_vocab_size,
                               padding_idx=0, smoothing=cfg.TRAIN.SMOOTHING)
    criterion.to(device)
    model = build_model(src_vocab_size,
                        tgt_vocab_size,
                        N=cfg.MODEL.NUM_LAYERS,
                        d_model=cfg.MODEL.D_MODEL,
                        d_ff=cfg.MODEL.FFN_DIM,
                        h=cfg.MODEL.ATTN_HEADS,
                        dropout=cfg.MODEL.DROPOUT)
    model.to(device)
    model_opt = NoamOpt(model.src_embed[0].d_model,
                        cfg.OPTIMIZER.FACTOR,
                        cfg.OPTIMIZER.WARMUP,
                        torch.optim.Adam(model.parameters(),
                                         lr=cfg.OPTIMIZER.INITIAL_LR,
                                         betas=(cfg.OPTIMIZER.BETA_1,
                                                cfg.OPTIMIZER.BETA_2),
                                         eps=cfg.OPTIMIZER.EPS))

    for epoch in range(1, cfg.TRAIN.N_EPOCHS + 1):
        model.train()
        loss_train = run_epoch(data_gen, model,
                  LossCompute(model.generator, criterion, model_opt),
                  epoch, writer, experiment_dir, cfg, dataset)

        writer.add_scalar('loss/train/epoch', loss_train, epoch)
        model.eval()
        with torch.no_grad():
            loss_val = run_epoch(data_gen_val, model,
                            LossCompute(model.generator, criterion, None),
                            epoch, writer, experiment_dir, cfg, dataset)

        writer.add_scalar('loss/val/epoch', loss_val, epoch)

        torch.save(model, os.path.join(experiment_dir,
                                "en-de-model-epoch-{}.pt".format(str(epoch).zfill(3))))

        # torch.save(model.state_dict(),
                   # os.path.join(experiment_dir,
                                # "epoch_{}.pth".format(str(epoch).zfill(3))))

    torch.save(model, os.path.join(experiment_dir, "en-de-model-final.pt"))



if __name__ == "__main__":
    main()
