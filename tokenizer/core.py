import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import SPECIAL_TOKENS


@dataclass
class Vocabulary:
    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]
    special_tokens: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def build(cls, special_tokens: Optional[Dict[str, str]] = None) -> "Vocabulary":
        special_tokens = special_tokens or SPECIAL_TOKENS
        token_to_id = {token: idx for idx, token in enumerate(special_tokens.values())}
        id_to_token = {idx: token for token, idx in token_to_id.items()}
        special_token_ids = {name: token_to_id[token] for name, token in special_tokens.items()}
        return cls(token_to_id, id_to_token, special_token_ids)

    def token_id(self, name: str) -> int:
        return self.special_tokens[name]

    def size(self) -> int:
        return len(self.token_to_id)


class BPETokenizer:
    def __init__(self, vocabulary: Vocabulary, merges: List[Tuple[str, str]]):
        self.vocabulary = vocabulary
        self.merges = merges
        self.merge_rank = {merge: rank for rank, merge in enumerate(merges)}
        self.word_pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocabulary = Vocabulary(
            token_to_id={token: int(idx) for token, idx in data["token_to_id"].items()},
            id_to_token={int(idx): token for idx, token in data["id_to_token"].items()},
            special_tokens={name: int(idx) for name, idx in data["special_tokens"].items()},
        )
        merges = [tuple(merge) for merge in data["merges"]]
        return cls(vocabulary, merges)

    def save(self, path: str):
        data = {
            "token_to_id": self.vocabulary.token_to_id,
            "id_to_token": {str(idx): token for idx, token in self.vocabulary.id_to_token.items()},
            "special_tokens": self.vocabulary.special_tokens,
            "merges": [list(merge) for merge in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _merge_word(self, tokens: List[str]) -> List[str]:
        while len(tokens) >= 2:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            best_pair = min(pairs, key=lambda pair: self.merge_rank.get(pair, float("inf")))
            if best_pair not in self.merge_rank:
                break
            merged = best_pair[0] + best_pair[1]
            result = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    result.append(merged)
                    i += 2
                else:
                    result.append(tokens[i])
                    i += 1
            tokens = result
        return tokens

    def _encode_plain(self, chunk: str, unk_id: int) -> List[int]:
        if not chunk:
            return []
        ids = []
        for match in self.word_pattern.finditer(chunk):
            word_tokens = self._merge_word(list(match.group()) + ["</w>"])
            ids.extend(self.vocabulary.token_to_id.get(token, unk_id) for token in word_tokens)
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = [self.vocabulary.token_id("<bos>")] if add_bos else []
        unk_id = self.vocabulary.token_id("<unk>")

        specials = sorted(self.vocabulary.special_tokens.keys(), key=len, reverse=True)
        special_pattern = re.compile("|".join(re.escape(token) for token in specials)) if specials else None

        position = 0
        if special_pattern is not None:
            for match in special_pattern.finditer(text):
                ids.extend(self._encode_plain(text[position : match.start()], unk_id))
                ids.append(self.vocabulary.special_tokens[match.group()])
                position = match.end()
        ids.extend(self._encode_plain(text[position:], unk_id))

        if add_eos:
            ids.append(self.vocabulary.token_id("<eos>"))
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = [self.vocabulary.id_to_token.get(idx, "<unk>") for idx in ids]
        text = "".join(tokens).replace("</w>", " ")
        return text.strip()
