# Attention Is All You Need

* [Paper](https://arxiv.org/pdf/1706.03762.pdf)


## Model architecture 

<img src="assets/transformer_model_architecture.jpg" width="50%"/>

### Components

<img src="assets/attention.jpg" width="80%"/>


## ToDos

* code for loading the data, I think I will use WMT14 EN-DE
* implement the model:
    - positional encoding
    - scaled dot product attention
    - (masked) multi-head attention
    - feed forward layer
    - add & norm layer

## References

* [Attention? Attention!](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)
* [Pytorch Tutorial](https://pytorch.org/tutorials/beginner/translation_transformer.html)
* [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
* [Annotated PyTorch Paper Implementations](https://nn.labml.ai/)