# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import SentimentDatasetDAN, DAN, collate_dan, word_encoder, build_train_vocab, load_glove, random_embedding
from bpe import train_bpe


# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(data_loader):

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            eval_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader, epochs=100, lr=0.0001, print_every=10):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(epochs):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % print_every == print_every - 1:
            print(f'Epoch #{epoch + 1}: train loss {train_loss:.3f}, train accuracy {train_accuracy:.3f}, '
                  f'dev loss {test_loss:.3f}, dev accuracy {test_accuracy:.3f}')

    return all_train_accuracy, all_test_accuracy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dan_loaders(encode, batch_size):
    train_data = SentimentDatasetDAN("data/train.txt", encode)
    dev_data = SentimentDatasetDAN("data/dev.txt", encode)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_dan)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False, collate_fn=collate_dan)
    return train_loader, dev_loader


def run_dan(name, embedding, encode, args, plot_file):
    train_loader, dev_loader = dan_loaders(encode, args.batch_size)
    model = DAN(embedding, hidden_size=args.hidden_size, num_layers=args.num_layers,
                dropout=args.dropout, word_dropout=args.word_dropout)
    print(f'\n{name}:')
    start_time = time.time()
    train_acc, dev_acc = experiment(model, train_loader, dev_loader, epochs=args.epochs, lr=args.lr,
                                    print_every=args.print_every)
    elapsed = time.time() - start_time
    best_epoch = int(np.argmax(dev_acc))
    print(f'{name}: trained in {elapsed:.1f} seconds; final dev accuracy {dev_acc[-1]:.3f}, '
          f'best dev accuracy {dev_acc[best_epoch]:.3f} (epoch {best_epoch + 1})')

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_acc) + 1), train_acc, label='train')
    plt.plot(range(1, len(dev_acc) + 1), dev_acc, label='dev')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy for {name}')
    plt.legend()
    plt.grid()
    plt.savefig(plot_file)
    plt.close()
    print(f"Accuracy plot saved as {plot_file}")


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, choices=['BOW', 'DAN', 'SUBWORDDAN'],
                        help='Model type to train')
    parser.add_argument('--emb_dim', type=int, default=300, choices=[50, 300],
                        help='DAN: which GloVe embeddings to use')
    parser.add_argument('--random_init', action='store_true',
                        help='DAN: randomly initialize the embeddings instead of using GloVe (Part 1b)')
    parser.add_argument('--freeze', action='store_true', help='DAN: keep GloVe embeddings frozen')
    parser.add_argument('--vocab_size', type=int, default=10000, help='SUBWORDDAN: BPE subword vocabulary size')
    parser.add_argument('--sub_emb_dim', type=int, default=300, help='SUBWORDDAN: subword embedding size')
    parser.add_argument('--hidden_size', type=int, default=300, help='DAN: hidden layer size')
    parser.add_argument('--num_layers', type=int, default=2, help='DAN: number of hidden layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='DAN: dropout probability')
    parser.add_argument('--word_dropout', type=float, default=0.0,
                        help='DAN: probability of dropping each token from the average during training')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Training epochs (default: 100 for BOW, 15 for DAN/SUBWORDDAN)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Adam learning rate (default: 1e-4 for BOW and GloVe DAN, '
                             '1e-3 for random-init DAN and SUBWORDDAN)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (default: 16 for BOW, 32 for DAN)')
    parser.add_argument('--print_every', type=int, default=None,
                        help='Print losses/accuracies every N epochs (default: 10 for BOW, 1 for DAN)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    # Parse the command-line arguments
    args = parser.parse_args()
    set_seed(args.seed)

    is_bow = args.model == "BOW"
    if args.epochs is None:
        args.epochs = 100 if is_bow else 15
    if args.lr is None:
        glove_dan = args.model == "DAN" and not args.random_init
        args.lr = 0.0001 if is_bow or glove_dan else 0.001
    if args.batch_size is None:
        args.batch_size = 16 if is_bow else 32
    if args.print_every is None:
        args.print_every = 10 if is_bow else 1

    # Check if the model type is "BOW"
    if args.model == "BOW":
        # Load dataset
        start_time = time.time()

        train_data = SentimentDatasetBOW("data/train.txt")
        dev_data = SentimentDatasetBOW("data/dev.txt", vectorizer=train_data.vectorizer, train=False)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")

        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader,
                                                           test_loader, epochs=args.epochs, lr=args.lr,
                                                           print_every=args.print_every)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader,
                                                           test_loader, epochs=args.epochs, lr=args.lr,
                                                           print_every=args.print_every)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    elif args.model == "DAN":
        #TODO:  Train and evaluate your DAN
        start_time = time.time()
        if args.random_init:
            indexer = build_train_vocab("data/train.txt")
            embedding = random_embedding(len(indexer), args.emb_dim)
            name = f'DAN, random {args.emb_dim}d embeddings'
        else:
            word_embeddings = load_glove(args.emb_dim)
            indexer = word_embeddings.word_indexer
            embedding = word_embeddings.get_initialized_embedding_layer(frozen=args.freeze)
            name = f'DAN, GloVe {args.emb_dim}d embeddings' + (' (frozen)' if args.freeze else '')
        print(f"Data loaded in : {time.time() - start_time:.1f} seconds")

        suffix = f"dan_{'random' if args.random_init else 'glove'}{args.emb_dim}"
        run_dan(name, embedding, word_encoder(indexer), args, f'{suffix}_accuracy.png')

    elif args.model == "SUBWORDDAN":
        start_time = time.time()
        tokenizer = train_bpe("data/train.txt", args.vocab_size)
        print(f"BPE trained in {time.time() - start_time:.1f} seconds: {len(tokenizer.merges)} merges, "
              f"{tokenizer.vocab_size()} tokens including PAD and UNK")

        embedding = random_embedding(tokenizer.vocab_size(), args.sub_emb_dim)
        run_dan(f'Subword DAN, BPE vocab {args.vocab_size}', embedding, tokenizer.encode, args,
                f'subword_dan_{args.vocab_size}_accuracy.png')

if __name__ == "__main__":
    main()
