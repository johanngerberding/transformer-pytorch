import os
from datetime import date
import time
import glob
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from data import WMT14, Batch, decode_sentence
from transformer import subsequent_mask, build_model

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


def run_epoch(data_iter, model: nn.Module, loss_compute) -> float:
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
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch step: {} loss: {} tokens per second: {}"
                  .format(i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0

    return total_loss / total_tokens


class SimpleLossCompute:
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
    batch_size = 8
    max_seq_len = 100
    epochs = 10
    warmup_steps = 2000
    initial_lr = 0
    beta_1 = 0.9
    beta_2 = 0.98

    dir_path = os.path.dirname(os.path.realpath(__file__))
    today = date.today().strftime("%Y-%m-%d")
    experiment_dir = "experiments/{}_{}_{}_{}".format(today,
                                                      str(batch_size),
                                                      str(epochs),
                                                      str(max_seq_len))

    experiment_dir = os.path.join(dir_path, experiment_dir)
    os.makedirs(experiment_dir, exist_ok=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training device: {device}")

    writer = SummaryWriter()
    dataset = WMT14('wmt14')
    dataset.load_vocab()
    data_gen = dataset.data_generator(batch_size,
                                      max_seq_len,
                                      data_type='train')
    data_gen_val = dataset.data_generator(batch_size,
                                          max_seq_len,
                                          data_type='test')

    src_vocab_size = len(dataset.src_idx2word)
    tgt_vocab_size = len(dataset.tgt_idx2word)

    criterion = LabelSmoothing(size=tgt_vocab_size,
                               padding_idx=0, smoothing=0.1)
    model = build_model(src_vocab_size, tgt_vocab_size, N=6)
    model.to(device)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, warmup_steps,
                        torch.optim.Adam(model.parameters(), lr=initial_lr,
                                         betas=(beta_1, beta_2), eps=1e-9))
    model_opt.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        loss_train = run_epoch(data_gen, model,
                  SimpleLossCompute(model.generator, criterion, model_opt))

        writer.add_scalar('loss/train', loss_train, epoch)
        model.eval()
        loss_val = run_epoch(data_gen_val, model,
                        SimpleLossCompute(model.generator, criterion, None))

        writer.add_scalar('loss/val', loss_val, epoch)

        torch.save(model.state_dict(), os.path.join(experiment_dir, "epoch_{}.pth".format(str(epoch).zfill(3))))
        checkpoints = glob.glob(experiment_dir + "/*.pth")
        checkpoints.sort()
        del_checkpoints = checkpoints[:-3]
        for ckpt in del_checkpoints:
            os.remove(ckpt)

    torch.save(model.state_dict(), os.path.join(experiment_dir, "final.pth"))



if __name__ == "__main__":
    main()
