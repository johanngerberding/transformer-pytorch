import gradio as gr
import os
import torch
import pickle

from config import get_cfg_defaults
from transformer import subsequent_mask
from translate import greedy_decode
from data import encode_sentence, decode_sentence, PAD_ID, SOS_ID

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

EXPS_ROOT = "/home/johann/sonstiges/transformer-pytorch/exps/2022-03-19_64_3_100"
MODEL_PATH = os.path.join(EXPS_ROOT, "en-de-model-iter-0000240000.pt")
CONFIG_PATH = os.path.join(EXPS_ROOT, "config.yaml")
DATASET_PATH = os.path.join(EXPS_ROOT, "dataset.file")

cfg = get_cfg_defaults()
cfg.merge_from_file(CONFIG_PATH)
cfg.freeze()

with open(DATASET_PATH, 'rb') as fp:
    dataset = pickle.load(fp)

model = torch.load(MODEL_PATH)
model.eval()
model.to(DEVICE)


def predict(sentence: str):
    enc_sen = encode_sentence(cfg, dataset, sentence)
    enc_sen = torch.tensor(enc_sen).long().unsqueeze(0).to(DEVICE)
    enc_sen_mask = (enc_sen != PAD_ID).unsqueeze(-2).to(DEVICE)

    with torch.no_grad():
        out = greedy_decode(
            model, enc_sen, enc_sen_mask,
            cfg.DATASET.MAX_SEQ_LEN, SOS_ID, dataset)

    out = out.detach().cpu().numpy().tolist()[0]
    translation = decode_sentence(dataset.tgt_idx2word, out)

    return translation



gr.Interface(
    fn=predict,
    inputs=gr.inputs.Textbox(lines=5, label="English sentence"),
    outputs=gr.outputs.Textbox(label="German translation"),
    title="Transformer en-de-Translation Demo"
).launch()
