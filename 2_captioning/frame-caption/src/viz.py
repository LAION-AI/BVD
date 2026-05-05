import matplotlib.pyplot as plt
import textwrap


def wrap_text(text, max_len=50):
    if len(text) > max_len:
        text = textwrap.fill(text, max_len)
    return text


def plot_grid(
    imgs, rows, cols, captions, orig_captions=None, figsize=(20, 12), fig_path=None
):
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    for i in range(rows):
        for j in range(cols):
            if orig_captions:
                orig_cap = orig_captions[i * cols + j]
                cap = captions[i * cols + j]
                caption = (
                    f"Orig: {wrap_text(orig_cap, 30)}\n\nGen: {wrap_text(cap, 30)}"
                )
            else:
                caption = wrap_text(captions[i * cols + j], 30)

            axs[i, j].imshow(imgs[i * cols + j])
            axs[i, j].set_title(caption, fontsize=8)
            axs[i, j].axis("off")
    if fig_path:
        plt.savefig(fig_path)
