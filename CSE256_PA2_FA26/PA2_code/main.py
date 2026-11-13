import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import sys
import json
import time
import warnings
import nltk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import TransformerEncoder, EncoderClassifier, TransformerDecoder
from utilities import Utilities


seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input
## size of 64, hidden size of 100 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

SENTENCES = ["We must never give up, and we will never give in.",
             "Our troops are fighting terrorists in Iraq so that we do not have to face them here at home."]

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data.
    """

    texts = []
    files = os.listdir(directory)
    for filename in files:
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
            losses.append(loss.item())
            if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def attention_maps(model, tokenizer, sentence):
    model.eval()
    ids = tokenizer.encode(sentence)[:block_size]
    with torch.no_grad():
        _, maps = model(torch.tensor([ids], dtype=torch.long, device=device))
    model.train()
    return [tokenizer.itos[i] for i in ids], maps


def sanity_check(model, tokenizer, prefix):
    for k, sentence in enumerate(SENTENCES, 1):
        print(f"\nSanity check on: {sentence!r}")
        Utilities(tokenizer, model).sanity_check(sentence, block_size, prefix=f"figures/{prefix}_s{k}_")
        words, maps = attention_maps(model, tokenizer, sentence)
        worst = max((m[0].sum(-1) - 1).abs().max().item() for m in maps)
        print(f"Max |row sum - 1| over {len(maps)} attention maps: {worst:.2e}")

        fig, axes = plt.subplots(n_head, n_layer, figsize=(3.2 * n_layer, 3.4 * n_head))
        for idx, attn in enumerate(maps):
            layer, head = divmod(idx, n_head)
            ax = axes[head][layer]
            ax.imshow(attn[0].cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
            ax.set_title(f"layer {layer + 1}, head {head + 1}", fontsize=9)
            ax.set_xticks(range(len(words)))
            ax.set_yticks(range(len(words)))
            ax.set_xticklabels(words, rotation=90, fontsize=6)
            ax.set_yticklabels(words, fontsize=6)
        fig.tight_layout()
        outfile = f"figures/{prefix}_attention_s{k}.png"
        fig.savefig(outfile, dpi=150)
        plt.close(fig)
        print(f"Labeled attention maps saved to {outfile}")


def part1(tokenizer):
    torch.manual_seed(seed)
    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=False)

    encoder = TransformerEncoder(tokenizer.vocab_size, n_embd, n_head, n_layer, block_size)
    model = EncoderClassifier(encoder, n_input, n_hidden, n_output).to(device)
    print(f"\nPart 1: encoder parameters {count_parameters(encoder):,}, classifier parameters "
          f"{count_parameters(model.classifier):,}, total {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    history = []
    start = time.time()

     # for the classification  task, you will train for a fixed number of epochs like this:

    for epoch in range(epochs_CLS):
        total_loss = 0.0
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)

            # CLS training code here
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        row = dict(epoch=epoch + 1, train_loss=total_loss / len(train_CLS_loader),
                   train_acc=compute_classifier_accuracy(model, train_CLS_loader),
                   test_acc=compute_classifier_accuracy(model, test_CLS_loader))
        history.append(row)
        print(f"Epoch {row['epoch']:2d}: train loss {row['train_loss']:.4f}, train accuracy {row['train_acc']:.2f}%, "
              f"test accuracy {row['test_acc']:.2f}%")
    print(f"Final test accuracy: {history[-1]['test_acc']:.2f}% (trained in {time.time() - start:.1f}s)")

    epochs = [r['epoch'] for r in history]
    plt.figure(figsize=(6, 3.6))
    plt.plot(epochs, [r['train_acc'] for r in history], marker="o", label="train")
    plt.plot(epochs, [r['test_acc'] for r in history], marker="o", label="test")
    plt.xlabel("epoch")
    plt.ylabel("accuracy (%)")
    plt.xticks(epochs)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/part1_accuracy.png", dpi=150)
    plt.close()

    sanity_check(encoder, tokenizer, "encoder")
    with open("results/part1.json", "w") as f:
        json.dump(dict(encoder_params=count_parameters(encoder), classifier_params=count_parameters(model.classifier),
                       total_params=count_parameters(model), history=history), f, indent=1)


def part2(tokenizer):
    torch.manual_seed(seed)
    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    test_LM_loaders = {}
    for name in ["obama", "wbush", "hbush"]:
        with open(f"speechesdataset/test_LM_{name}.txt", 'r', encoding='utf-8') as f:
            test_LM_loaders[name] = DataLoader(LanguageModelingDataset(tokenizer, f.read(), block_size),
                                               batch_size=batch_size, shuffle=False)

    model = TransformerDecoder(tokenizer.vocab_size, n_embd, n_head, n_layer, block_size).to(device)
    print(f"\nPart 2: decoder parameters {count_parameters(model):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    start = time.time()

    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        # LM training code here
        loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % eval_interval == 0:
            row = dict(iter=i + 1, train=compute_perplexity(model, train_LM_loader, eval_iters))
            for name, loader in test_LM_loaders.items():
                row[name] = compute_perplexity(model, loader, eval_iters)
            history.append(row)
            print(f"Iter {row['iter']:3d}: train perplexity {row['train']:.2f}, obama {row['obama']:.2f}, "
                  f"wbush {row['wbush']:.2f}, hbush {row['hbush']:.2f}")
    final = history[-1]
    print(f"Final perplexity: train {final['train']:.2f}, obama {final['obama']:.2f}, wbush {final['wbush']:.2f}, "
          f"hbush {final['hbush']:.2f} (trained in {time.time() - start:.1f}s)")
    full_test = {name: compute_perplexity(model, loader, eval_iters=len(loader)) for name, loader in test_LM_loaders.items()}
    print(f"Final perplexity over the full test sets: obama {full_test['obama']:.2f}, wbush {full_test['wbush']:.2f}, "
          f"hbush {full_test['hbush']:.2f}")

    iters = [r['iter'] for r in history]
    plt.figure(figsize=(6, 3.6))
    for key, label in [("train", "train"), ("obama", "Obama"), ("wbush", "W. Bush"), ("hbush", "H. W. Bush")]:
        plt.plot(iters, [r[key] for r in history], marker="o", label=label)
    plt.xlabel("iteration")
    plt.ylabel("perplexity")
    plt.xticks(iters)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/part2_perplexity.png", dpi=150)
    plt.close()

    sanity_check(model, tokenizer, "decoder")
    with open("results/part2.json", "w") as f:
        json.dump(dict(decoder_params=count_parameters(model), history=history, full_test=full_test), f, indent=1)


def main():
    parts = sys.argv[1:] or ["part1", "part2"]
    if any(p not in ("part1", "part2") for p in parts):
        sys.exit("Usage: python main.py [part1] [part2]")
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    if "part1" in parts:
        part1(tokenizer)
    if "part2" in parts:
        part2(tokenizer)




if __name__ == "__main__":
    main()
