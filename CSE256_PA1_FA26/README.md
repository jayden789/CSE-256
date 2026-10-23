# CSE 256 PA1: DAN, BPE, Skip-Gram

## Setup

Requires Python 3 with `torch`, `numpy`, `scikit-learn`, and `matplotlib`:

```
pip install torch numpy scikit-learn matplotlib
```

Run all commands from this directory (the code reads `data/` relative to it).

## Running

| Part | Command | What it does |
|---|---|---|
| Baseline | `python main.py --model BOW` | Bag-of-words 2- and 3-layer networks (provided) |
| 1a | `python main.py --model DAN` | DAN with fine-tuned 300d GloVe embeddings |
| 1b | `python main.py --model DAN --random_init` | DAN with randomly initialized embeddings |
| 2 | `python main.py --model SUBWORDDAN` | DAN over BPE subwords (vocab size 10000, random embeddings) |

Each run prints train/dev loss and accuracy per epoch, the training time, and the final and best dev accuracy,
and saves a train/dev accuracy plot (e.g. `dan_glove300_accuracy.png`).

### Options

| Flag | Default | Applies to | Meaning |
|---|---|---|---|
| `--emb_dim {50,300}` | 300 | DAN | GloVe embedding size (or random embedding size with `--random_init`) |
| `--random_init` | off | DAN | Random embeddings learned from scratch; vocabulary built from `train.txt` |
| `--freeze` | off | DAN | Keep GloVe embeddings fixed |
| `--vocab_size N` | 10000 | SUBWORDDAN | Target BPE vocabulary size (stops early once no pair occurs twice, ~13.4k) |
| `--sub_emb_dim N` | 300 | SUBWORDDAN | Subword embedding size |
| `--hidden_size N` | 300 | DAN, SUBWORDDAN | Hidden layer size |
| `--num_layers N` | 2 | DAN, SUBWORDDAN | Number of hidden layers |
| `--dropout P` | 0.3 | DAN, SUBWORDDAN | Dropout on the averaged embedding and after each hidden layer |
| `--word_dropout P` | 0.0 | DAN, SUBWORDDAN | Drop each token from the average with probability P during training |
| `--epochs N` | 100 BOW / 15 otherwise | all | Training epochs |
| `--lr X` | 1e-4 BOW and GloVe DAN / 1e-3 otherwise | all | Adam learning rate |
| `--batch_size N` | 16 BOW / 32 otherwise | all | Batch size |
| `--seed N` | 0 | all | Random seed |

Examples:

```
python main.py --model DAN --emb_dim 50
python main.py --model SUBWORDDAN --vocab_size 2000
```

### Experiments in the report

```
python run_experiments.py --part all --seeds 3
```

Runs every configuration in the report over 3 seeds (about 25 minutes on a laptop CPU) and writes tables to
`results/part*.json` and plots to `results/part*_curves.png`. Use `--part 1a`, `1b`, or `2` to run one part.

## Files

- `DANmodels.py`: DAN model (masked averaging of embeddings + feedforward network), dataset, and padding collate function
- `bpe.py`: Byte Pair Encoding trainer and tokenizer
- `main.py`: training/evaluation entry point for all models
- `run_experiments.py`: hyperparameter and ablation sweeps for the report
- `BOWmodels.py`, `sentiment_data.py`, `utils.py`: provided starter code (unchanged)
