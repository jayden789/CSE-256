# run_experiments.py
#
# Runs the hyperparameter / ablation experiments reported in the writeup and saves results to results/.
# Usage: python run_experiments.py [--part 1a|1b|2|all] [--seeds 3]

import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main import set_seed, dan_loaders, experiment
from DANmodels import DAN, word_encoder, build_train_vocab, load_glove, random_embedding
from bpe import train_bpe


BASE = dict(emb_dim=300, freeze=False, hidden_size=300, num_layers=2, dropout=0.3, word_dropout=0.0,
            epochs=15, lr=0.001, batch_size=32)

_glove_cache = {}


def glove(dim):
    if dim not in _glove_cache:
        _glove_cache[dim] = load_glove(dim)
    return _glove_cache[dim]


def run_config(cfg, seed):
    """
    Trains one DAN configuration and returns its per-epoch curves and timing.
    cfg['kind'] is 'glove', 'random', or 'bpe'.
    """
    set_seed(seed)
    c = SimpleNamespace(**{**BASE, **cfg})
    if c.kind == "glove":
        we = glove(c.emb_dim)
        encode = word_encoder(we.word_indexer)
        embedding = we.get_initialized_embedding_layer(frozen=c.freeze)
        vocab = len(we.word_indexer)
    elif c.kind == "random":
        indexer = build_train_vocab("data/train.txt")
        encode = word_encoder(indexer)
        embedding = random_embedding(len(indexer), c.emb_dim)
        vocab = len(indexer)
    else:
        tok = train_bpe("data/train.txt", c.vocab_size)
        encode = tok.encode
        embedding = random_embedding(tok.vocab_size(), c.emb_dim)
        vocab = tok.vocab_size()

    train_loader, dev_loader = dan_loaders(encode, c.batch_size)
    tokens_per_sent = float(np.mean([len(x) for x in train_loader.dataset.indices]))
    model = DAN(embedding, hidden_size=c.hidden_size, num_layers=c.num_layers, dropout=c.dropout,
                word_dropout=c.word_dropout)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    start = time.time()
    train_acc, dev_acc = experiment(model, train_loader, dev_loader, epochs=c.epochs, lr=c.lr,
                                    print_every=10 ** 9)
    return dict(train_acc=train_acc, dev_acc=dev_acc, seconds=time.time() - start, vocab=vocab,
                tokens_per_sent=tokens_per_sent, params=n_params)


EXPERIMENTS = {
    "1a": [
        ("GloVe 300d (base)", dict(kind="glove")),
        ("GloVe 50d", dict(kind="glove", emb_dim=50)),
        ("GloVe 300d, frozen", dict(kind="glove", freeze=True)),
        ("GloVe 50d, frozen", dict(kind="glove", emb_dim=50, freeze=True)),
        ("1 hidden layer", dict(kind="glove", num_layers=1)),
        ("3 hidden layers", dict(kind="glove", num_layers=3)),
        ("hidden size 100", dict(kind="glove", hidden_size=100)),
        ("no dropout", dict(kind="glove", dropout=0.0)),
        ("dropout 0.5", dict(kind="glove", dropout=0.5)),
        ("word dropout 0.3", dict(kind="glove", word_dropout=0.3)),
        ("lr 1e-4", dict(kind="glove", lr=0.0001)),
    ],
    "1b": [
        ("GloVe 300d", dict(kind="glove")),
        ("Random 300d", dict(kind="random")),
        ("GloVe 50d", dict(kind="glove", emb_dim=50)),
        ("Random 50d", dict(kind="random", emb_dim=50)),
    ],
    "2": [("Word-level, random 300d", dict(kind="random"))] + [
        (f"BPE vocab {v}", dict(kind="bpe", vocab_size=v)) for v in [500, 1000, 2000, 5000, 10000, 15000]
    ],
}


def summarize(name, runs):
    best = [max(r["dev_acc"]) for r in runs]
    final = [r["dev_acc"][-1] for r in runs]
    train = [r["train_acc"][-1] for r in runs]
    best_epoch = [int(np.argmax(r["dev_acc"])) + 1 for r in runs]
    return dict(name=name, best_dev=float(np.mean(best)), best_dev_std=float(np.std(best)),
                final_dev=float(np.mean(final)), final_dev_std=float(np.std(final)),
                final_train=float(np.mean(train)), best_epoch=float(np.mean(best_epoch)),
                seconds=float(np.mean([r["seconds"] for r in runs])), vocab=runs[0]["vocab"],
                tokens_per_sent=runs[0]["tokens_per_sent"], params=runs[0]["params"],
                mean_dev_curve=np.mean([r["dev_acc"] for r in runs], 0).tolist(),
                mean_train_curve=np.mean([r["train_acc"] for r in runs], 0).tolist())


def plot_part(part, rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for row in rows:
        epochs = range(1, len(row["mean_dev_curve"]) + 1)
        axes[0].plot(epochs, row["mean_train_curve"], label=row["name"])
        axes[1].plot(epochs, row["mean_dev_curve"], label=row["name"])
    for ax, title in zip(axes, ["Train accuracy", "Dev accuracy"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.grid()
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"results/part{part}_curves.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default="all", choices=["1a", "1b", "2", "all"])
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    os.makedirs("results", exist_ok=True)

    parts = ["1a", "1b", "2"] if args.part == "all" else [args.part]
    for part in parts:
        rows = []
        print(f"\n=== Part {part} ({args.seeds} seeds, mean ± std) ===")
        print(f"{'config':28s} {'best dev':>14s} {'final dev':>14s} {'train':>6s} {'best ep':>7s} "
              f"{'sec':>6s} {'vocab':>6s} {'tok/sent':>8s}")
        for name, cfg in EXPERIMENTS[part]:
            runs = [run_config(cfg, seed) for seed in range(args.seeds)]
            row = summarize(name, runs)
            rows.append(row)
            print(f"{name:28s} {row['best_dev']:.3f} ± {row['best_dev_std']:.3f}  "
                  f"{row['final_dev']:.3f} ± {row['final_dev_std']:.3f} {row['final_train']:6.3f} "
                  f"{row['best_epoch']:7.1f} {row['seconds']:6.1f} {row['vocab']:6d} {row['tokens_per_sent']:8.1f}",
                  flush=True)
        with open(f"results/part{part}.json", "w") as f:
            json.dump(rows, f, indent=1)
        plot_part(part, rows)


if __name__ == "__main__":
    main()
