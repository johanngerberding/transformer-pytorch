import matplotlib.pyplot as plt
import seaborn as sns

def draw(data, x, y, ax):
    sns.heatmap(data, xticklabels=x,
                square=True, yticklabels=y,
                vmin=0.0, vmax=1.0,
                cbar=False, ax=ax)


def plot_encoder_attn(src: str, model, step=2, num_heads=4):
    """Plot encoder attention maps."""
    sent = src.split()
    for layer in range(1, len(model.encoder.layers), step):
        fig, axs = plt.subplots(1,4, figsize=(20, 6))
        fig.suptitle(f"Encoder Layer {layer+1}", fontsize=16)
        fig.tight_layout()
        for h in range(num_heads):
            draw(model.encoder.layers[layer].self_attn.attn[0, h].data[:len(sent), :len(sent)],
                sent, sent if h ==0 else [], ax=axs[h])
            axs[h].set_title(f"Head {h}")
        plt.show()
        fig.savefig(f"encoder_attn_{layer+1}.jpg")


def plot_decoder_attn(src: str, tgt: str, model, step=2, num_heads=4):
    """Plot decoder attention maps."""
    sent = src.split()
    tgt_sent = tgt.split()

    for layer in range(1, len(model.decoder.layers), step):
        fig, axs = plt.subplots(1, num_heads, figsize=(20, 6))
        fig.suptitle(f'Decoder Attention Layer {layer+1}', fontsize=16)
        fig.tight_layout()
        for h in range(num_heads):
            draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(tgt_sent)],
                tgt_sent, tgt_sent if h ==0 else [], ax=axs[h])
            axs[h].set_title(f"Head {h}")
        plt.show()
        fig.savefig(f"decoder_attn_{layer+1}.jpg")

        fig, axs = plt.subplots(1, num_heads, figsize=(20, 6))
        fig.suptitle(f'Decoder Src-Attention Layer {layer+1}', fontsize=16)
        fig.tight_layout()
        for h in range(num_heads):
            draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(sent)],
                sent, tgt_sent if h ==0 else [], ax=axs[h])
            axs[h].set_title(f"Head {h}")
        plt.show()
        fig.savefig(f"decoder_src_attn_{layer+1}.jpg")


def plot_attn(src: str, tgt: str, model, layer: int, head: int):
    pass
