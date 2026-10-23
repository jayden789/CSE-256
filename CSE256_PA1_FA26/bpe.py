# bpe.py

import heapq
from collections import Counter, defaultdict
from sentiment_data import read_sentiment_examples
from utils import Indexer


END_OF_WORD = "</w>"


def merge_pair(symbols, pair, merged):
    """
    Replaces every non-overlapping occurrence of the adjacent pair (left to right) in symbols with merged.
    """
    out = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def word_to_symbols(word):
    """
    Splits a word into characters, attaching the end-of-word marker to the last one (as in Sennrich et al.),
    so that e.g. a suffix "er</w>" is distinguished from "er" inside a word.
    """
    chars = list(word)
    chars[-1] = chars[-1] + END_OF_WORD
    return chars


class BPETokenizer:
    """
    Byte Pair Encoding (Sennrich et al., 2016). Learns an ordered list of merges from a word-frequency table,
    then segments words by applying those merges in the order they were learned.

    The token indexer reserves index 0 for PAD and index 1 for UNK, matching the word-level DAN setup.
    """

    def __init__(self):
        self.merges = []      # list of (left, right) pairs, in the order learned
        self.ranks = {}       # pair -> position in self.merges
        self.indexer = Indexer()
        self.cache = {}

    def train(self, word_counts, vocab_size):
        """
        :param word_counts: Counter mapping word -> frequency in the training data
        :param vocab_size: target number of subword types (not counting PAD and UNK). Training stops early if
        no pair occurs at least twice.
        """
        words = [word_to_symbols(w) for w in word_counts]
        freqs = [word_counts[w] for w in word_counts]

        vocab = set()
        pair_counts = Counter()
        where = defaultdict(set)  # pair -> ids of words that (may) contain it
        for wid, symbols in enumerate(words):
            vocab.update(symbols)
            for pair in zip(symbols, symbols[1:]):
                pair_counts[pair] += freqs[wid]
                where[pair].add(wid)

        # Max-heap of (-count, pair) with lazy deletion: stale entries are skipped when popped. Ties are broken by
        # the pair itself so training is deterministic.
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        while len(vocab) < vocab_size and heap:
            neg_count, best = heapq.heappop(heap)
            if pair_counts.get(best, 0) != -neg_count:
                continue  # stale entry
            if -neg_count < 2:
                break
            merged = best[0] + best[1]
            self.ranks[best] = len(self.merges)
            self.merges.append(best)
            vocab.add(merged)

            changed = set()
            for wid in where.pop(best):
                old = words[wid]
                new = merge_pair(old, best, merged)
                if len(new) == len(old):
                    continue
                f = freqs[wid]
                for pair in zip(old, old[1:]):
                    pair_counts[pair] -= f
                    changed.add(pair)
                for pair in zip(new, new[1:]):
                    pair_counts[pair] += f
                    where[pair].add(wid)
                    changed.add(pair)
                words[wid] = new

            pair_counts.pop(best, None)
            changed.discard(best)
            for pair in changed:
                c = pair_counts[pair]
                if c > 0:
                    heapq.heappush(heap, (-c, pair))
                else:
                    del pair_counts[pair]

        self.indexer = Indexer()
        self.indexer.add_and_get_index("PAD")
        self.indexer.add_and_get_index("UNK")
        for token in sorted(vocab):
            self.indexer.add_and_get_index(token)
        self.cache = {}
        return self

    def tokenize_word(self, word):
        """
        Segments one word by repeatedly applying the earliest-learned merge present in it.
        """
        if word in self.cache:
            return self.cache[word]
        symbols = word_to_symbols(word)
        while len(symbols) > 1:
            pairs = [p for p in zip(symbols, symbols[1:]) if p in self.ranks]
            if not pairs:
                break
            best = min(pairs, key=lambda p: self.ranks[p])
            symbols = merge_pair(symbols, best, best[0] + best[1])
        self.cache[word] = symbols
        return symbols

    def tokenize(self, words):
        return [tok for w in words for tok in self.tokenize_word(w)]

    def encode(self, words):
        """
        :return: subword token indices for a list of words; tokens not in the vocabulary (e.g. characters never
        seen in training) map to UNK
        """
        idxs = [self.indexer.index_of(t) for t in self.tokenize(words)]
        return [i if i != -1 else 1 for i in idxs]

    def vocab_size(self):
        return len(self.indexer)


def train_bpe(infile, vocab_size):
    word_counts = Counter(w for ex in read_sentiment_examples(infile) for w in ex.words)
    return BPETokenizer().train(word_counts, vocab_size)
