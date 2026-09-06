from collections import Counter
import re
from typing import Dict, Iterable, List, Tuple

from .core import BPETokenizer, Vocabulary


class BPETrainer:
    def __init__(self, vocab_size: int, special_tokens: Dict[str, str]):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.word_pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def _pair_counts(self, words: List[List[str]]) -> Counter:
        counts = Counter()
        for word in words:
            counts.update(zip(word[:-1], word[1:]))
        return counts

    def _merge_corpus(self, words: List[List[str]], pair: Tuple[str, str], merged: str) -> List[List[str]]:
        new_words = []
        for word in words:
            result = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    result.append(merged)
                    i += 2
                else:
                    result.append(word[i])
                    i += 1
            new_words.append(result)
        return new_words

    def train(self, texts: Iterable[str]) -> BPETokenizer:
        vocabulary = Vocabulary.build(self.special_tokens)
        token_to_id = dict(vocabulary.token_to_id)
        id_to_token = dict(vocabulary.id_to_token)

        words = []
        for text in texts:
            for match in self.word_pattern.finditer(text):
                words.append(list(match.group()) + ["</w>"])

        merges: List[Tuple[str, str]] = []
        while len(token_to_id) < self.vocab_size:
            counts = self._pair_counts(words)
            if not counts:
                break
            best_pair, _ = counts.most_common(1)[0]
            merges.append(best_pair)
            merged = best_pair[0] + best_pair[1]
            token_to_id[merged] = len(token_to_id)
            id_to_token[len(id_to_token)] = merged
            words = self._merge_corpus(words, best_pair, merged)

        return BPETokenizer(Vocabulary(token_to_id, id_to_token, vocabulary.special_tokens), merges)
