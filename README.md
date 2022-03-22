# Attention Is All You Need

This repository contains my implementation of the (Vanilla) Transformer ([Paper](https://arxiv.org/pdf/1706.03762.pdf)) model from 2017 using PyTorch.

## Training

You can train your Transformer model from scratch after creating an environment based on the `requirements.txt`. The configurations (`config.py`) are from the original paper, you can change them if necessary. The WMT14 dataset will be downloaded automatically. If you want to add your own dataset you have to modify `data.py`.

Training with standard config:
```
python3 -m venv .env
source .env/bin/activate
# install pytorch (I worked with 1.9) before other requirements
pip install -r requirements.txt
python train.py
```

## Inference 

You can download a pretrained english to german translation model, the corresponding config file and the dataset file [here](). This model was trained on a single GPU for 520,000 iterations.  You can use the `translate.py` to translate one sentence you provide like

```
python translate model.pt cfg.yaml dataset.file --src "your english sentence"
```

If you are a visual person, you might prefer the `transformer-demo.py`. This script starts a simple gradio application for translation.

## ToDos

* implement BLEU score evaluation

## Model architecture 

The visualization down below shows the overall architecture of the Transformer model. The model from the original Paper consists of 6 Encoder and Decoder Blocks.  

<img src="assets/transformer_model_architecture.png" width="50%"/>

The Scaled-Dot-Product-Attention is the core building block of the Transformer model architecture.

<img src="assets/attention.png" width="80%"/>

If you are interested in the details of how this model actually works, I recommend checking out the references down below.

## References

* [Attention? Attention!](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)
* [Pytorch Tutorial](https://pytorch.org/tutorials/beginner/translation_transformer.html)
* [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
* [Annotated PyTorch Paper Implementations](https://nn.labml.ai/)