# %%
import torch as th
from dictionary_learning.dictionary import CrossCoder

path = "models/1740989131_provocative-harrier/1740989131_provocative-harrier_16k5e-2/3001344_toks.pt"
model = CrossCoder.from_pretrained(path)
# %%
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj=path,
    path_in_repo="model.pt",
    repo_id="Butanium/crosscoder-Qwen2.5-0.5B-Instruct-and-Base-16k5e-2-3M-toks",
    repo_type="model",
)
# %%
import tempfile
import json
import os

with tempfile.TemporaryDirectory() as directory:
    path = os.path.join(directory, "config.json")
    json_dict = {
        "model_type": "crosscoder",
        "model_0": "Qwen2.5-0.5B",
        "model_1": "Qwen2.5-0.5B-Instruct",
        "activation_dim": model.decoder.weight.shape[2],
        "dict_size": model.decoder.weight.shape[1],
        "num_layers": 2,
        "mu": 5e-2,
        "learning_rate": 1e-4,
        "batch_size": 1024,
    }
    with open(path, "w") as f:
        json.dump(json_dict, f)
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo="config.json",
        repo_id="Butanium/crosscoder-Qwen2.5-0.5B-Instruct-and-Base-16k5e-2-3M-toks",
        repo_type="model",
    )
# %%
model.decoder.weight.shape
# %%
from dictionary_learning.dictionary import CrossCoder

hf_model = CrossCoder.from_pretrained(
    "Butanium/crosscoder-Qwen2.5-0.5B-Instruct-and-Base-16k5e-2-3M-toks", from_hub=True
)
# %%
from torch.testing import assert_close

for param, hf_param in zip(model.parameters(), hf_model.parameters()):
    assert_close(param, hf_param)
# %%

# %%


# %%
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams["text.usetex"] = True
plt.rcParams.update({"font.size": 20})
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


@th.no_grad()
def plot_norm_hist(crosscoder, name, fig=None, ax=None):
    norms = crosscoder.decoder.weight.norm(dim=2)
    rel_norms = 0.5 * ((norms[1] - norms[0]) / th.maximum(norms[0], norms[1]) + 1)
    values = rel_norms.detach().cpu().numpy()

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 4.0))

    hist, bins, _ = ax.hist(
        values, bins=100, color="lightgray", label="Other", log=True
    )
    ax.hist(values, bins=bins, color="lightgray", log=True)  # Base gray histogram
    ax.hist(
        values[((values >= 0.4) & (values < 0.6))],
        bins=bins,
        color="C1",
        label="Shared",
        log=True,
    )
    ax.hist(
        values[((values >= 0.9))], bins=bins, color="C0", label="Chat-only", log=True
    )
    ax.hist(
        values[(values <= 0.1)],
        bins=bins,
        color="limegreen",
        label="Base-only",
        log=True,
    )
    ax.set_xticks([0, 0.1, 0.4, 0.5, 0.6, 0.9, 1])
    ax.axvline(x=0.1, color="green", linestyle="--", alpha=0.5)
    ax.axvline(x=0.4, color="C1", linestyle="--", alpha=0.5)
    ax.axvline(x=0.6, color="C1", linestyle="--", alpha=0.5)
    ax.axvline(x=0.9, color="C0", linestyle="--", alpha=0.5)
    ax.set_xlabel("Relative Norm Difference")
    ax.set_ylabel("Latents")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left")
    ax.set_title(name)
    fig.tight_layout()
    return fig, ax


# %%
tokens = [1000448, 2000896, 3001344]
widths = ["16k", "32k"]
L1_penalties = ["5e-2", "3e-2"]
from itertools import product

all_settings = list(product(tokens, widths, L1_penalties))

fig, axes = plt.subplots(3, 4, figsize=(30, 16))
from tqdm import tqdm

for (tokens, width, L1_penalty), ax in tqdm(
    zip(all_settings, axes.flatten()), total=len(all_settings)
):
    path = f"models/1740989131_provocative-harrier/1740989131_provocative-harrier_{width}{L1_penalty}/{tokens}_toks.pt"
    model = CrossCoder.from_pretrained(path)
    tok_string = f"{tokens // 1e6}M tokens"
    plot_norm_hist(model, f"{width} {L1_penalty} {tok_string}", ax=ax, fig=fig)
plt.savefig("norm_hist.png", dpi=300)
# %%
