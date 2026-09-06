from collections import Counter, defaultdict
import re
from typing import Dict, Iterable, List, Tuple

from .core import BPETokenizer, Vocabulary


class BPETrainer:
    def __init__(self, vocab_size: int, special_tokens: Dict[str, str]):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.word_pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def _collect_word_freqs(self, texts: Iterable[str]) -> Counter:
        word_freqs = Counter()
        for text in texts:
            for match in self.word_pattern.finditer(text):
                word_freqs[match.group()] += 1
        return word_freqs

    def _merge_word(self, word: List[str], pair: Tuple[str, str], merged: str) -> List[str]:
        result = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                result.append(merged)
                i += 2
            else:
                result.append(word[i])
                i += 1
        return result

    def train(self, texts: Iterable[str]) -> BPETokenizer:
        vocabulary = Vocabulary.build(self.special_tokens)
        token_to_id = dict(vocabulary.token_to_id)
        id_to_token = dict(vocabulary.id_to_token)

        word_freqs = self._collect_word_freqs(texts)
        words = [list(word) + ["</w>"] for word in word_freqs]
        freqs = list(word_freqs.values())

        pair_counts = Counter()
        pair_to_words = defaultdict(set)
        for idx, (word, freq) in enumerate(zip(words, freqs)):
            for pair in zip(word[:-1], word[1:]):
                pair_counts[pair] += freq
                pair_to_words[pair].add(idx)

        merges: List[Tuple[str, str]] = []
        target = self.vocab_size - len(token_to_id)
        while len(merges) < target and pair_counts:
            best_pair = max(pair_counts, key=pair_counts.get)
            merged = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            token_to_id[merged] = len(token_to_id)
            id_to_token[len(id_to_token)] = merged

            for idx in list(pair_to_words[best_pair]):
                word = words[idx]
                if best_pair not in zip(word[:-1], word[1:]):
                    continue
                freq = freqs[idx]
                new_word = self._merge_word(word, best_pair, merged)
                for pair in zip(word[:-1], word[1:]):
                    pair_counts[pair] -= freq
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    pair_to_words[pair].discard(idx)
                for pair in zip(new_word[:-1], new_word[1:]):
                    pair_counts[pair] += freq
                    pair_to_words[pair].add(idx)
                words[idx] = new_word

            pair_counts.pop(best_pair, None)
            pair_to_words.pop(best_pair, None)

            if len(merges) % 500 == 0:
                print(f"BPE merges: {len(merges)}/{target}")

        return BPETokenizer(Vocabulary(token_to_id, id_to_token, vocabulary.special_tokens), merges)
