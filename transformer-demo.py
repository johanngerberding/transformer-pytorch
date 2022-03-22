import gradio as gr
import os
import torch
import pickle

from config import get_cfg_defaults
from transformer import subsequent_mask
from translate import greedy_decode
from data import encode_sentence, decode_sentence, PAD_ID, SOS_ID
from utils import plot_attn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPS_ROOT = "/home/johann/sonstiges/transformer-pytorch/exps/2022-03-19_64_3_100"
MODEL_PATH = os.path.join(EXPS_ROOT, "en-de-model-iter-0000520000.pt")
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


def predict(sentence: str, layer, head, part):
    enc_sen = encode_sentence(cfg, dataset, sentence)
    enc_sen = torch.tensor(enc_sen).long().unsqueeze(0).to(DEVICE)
    enc_sen_mask = (enc_sen != PAD_ID).unsqueeze(-2).to(DEVICE)

    with torch.no_grad():
        out = greedy_decode(
            model, enc_sen, enc_sen_mask,
            cfg.DATASET.MAX_SEQ_LEN, SOS_ID, dataset)

    out = out.detach().cpu().numpy().tolist()[0]
    translation = decode_sentence(dataset.tgt_idx2word, out)
    model.cpu()
    attn = plot_attn(sentence, translation, model, layer, head, part)

    return translation, attn



gr.Interface(
    fn=predict,
    inputs=[gr.inputs.Textbox(lines=5, label="English sentence"),
            gr.inputs.Slider(minimum=1, maximum=cfg.MODEL.NUM_LAYERS, step=1, label="Attention Layer"),
            gr.inputs.Slider(minimum=1, maximum=cfg.MODEL.ATTN_HEADS, step=1, label="Attention Head"),
            gr.inputs.Radio(["encoder", "decoder", "decoder-src"], default="decoder-src", label="Transformer Part")],
    outputs=[gr.outputs.Textbox(label="German translation"),
             gr.outputs.Image(label="Attention map")],
    title="Transformer en-de-Translation Demo"
).launch(share=True)
