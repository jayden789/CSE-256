# CSE 256 PA2: Transformer Blocks + Dissecting a Pretrained LLM

## Setup

Python 3 with `torch`, `numpy`, `matplotlib` and `nltk` (Parts 1–2), plus `transformers` and `jupyter` (Part 3):

```
pip install torch numpy matplotlib nltk transformers jupyter
```

`main.py` downloads NLTK's `punkt_tab` tokenizer data on first run. If that fails with an SSL certificate error
(common with python.org Python on macOS), run `pip install certifi` and then
`SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())") python -c "import nltk; nltk.download('punkt_tab')"`.

Run everything from this directory (`PA2_code/`); the code reads `speechesdataset/` relative to it.

## Parts 1 and 2

| Command | What it does |
|---|---|
| `python main.py` | Runs Part 1 then Part 2 with the default hyperparameters in `main.py` |
| `python main.py part1` | Part 1: Transformer encoder + feedforward classifier, trained jointly for 15 epochs |
| `python main.py part2` | Part 2: GPT-like decoder pretrained for 500 iterations on next-word prediction |

Each part prints its parameter count, the per-epoch test accuracy (Part 1) or the train and test perplexities every
100 iterations plus the final perplexity over the full test sets (Part 2), and runs the attention sanity checks (every attention row must sum to 1). Outputs:

- `results/part1.json`, `results/part2.json`: all numbers printed during training
- `figures/encoder_attention_s*.png`, `figures/decoder_attention_s*.png`: labeled attention maps (every layer and
  head) for two sentences
- `figures/*_attention_map_*.png`: the per-map plots from the provided `utilities.py` sanity check

Both runs take about 35 seconds each on a laptop CPU. Results are reproducible (seed 42, and the tokenizer
vocabulary is sorted so word ids do not change between runs).

## Part 3

Part 3 is the executed notebook `part3.ipynb` (with all outputs). To re-run it:

```
jupyter nbconvert --to notebook --execute --inplace part3.ipynb
```

It downloads six `EleutherAI/pythia-160m-deduped` checkpoints (about 4 GB) on first run and takes about 15 minutes
on a laptop CPU.

## Files

- `transformer.py`: attention head, multi-head attention, feedforward, Transformer block, learned token + position
  embeddings, `TransformerEncoder`, `EncoderClassifier` (mean pooling + one-hidden-layer classifier) and
  `TransformerDecoder`. Everything is built from `nn.Linear`/`nn.Embedding`/`nn.LayerNorm`; no library Transformer
  layers are used.
- `main.py`: hyperparameters, data loading, training and evaluation loops for Parts 1 and 2, and attention plots.
- `part3.ipynb`: Part 3 (logit lens and layer transplants on Pythia).
- `dataset.py`, `utilities.py`, `tokenizer.py`: provided starter code. Small changes: `utilities.sanity_check` takes
  an optional filename prefix and closes its figures; the tokenizer sorts its vocabulary for reproducible word ids.
  In `main.py`, the starter's `compute_perplexity` referenced an undefined `total_loss`; that line was removed and
  the function now runs under `torch.no_grad()`.
