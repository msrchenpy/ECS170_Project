'''
Text utilities shared by the Stage 4 text classification and text generation code.

Both Stage 4 tasks start from raw text, so the tokenisation, the vocabulary and the batch
padding helpers are factored out into this module instead of being duplicated inside the
two data loaders.

The tokenizer keeps words (letters / digits / inner apostrophes) and a small set of
sentence delimiters.  Everything is lower-cased, because the sentiment of a movie review
and the wording of a joke do not depend on the capitalisation of the first word of a
sentence, while the size of the vocabulary does.
'''

import re
import unicodedata

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
PAD_INDEX = 0
UNK_INDEX = 1
BOS_INDEX = 2
EOS_INDEX = 3

# a word is a run of letters / digits that may contain inner apostrophes ("don't", "80's"
# is split into "80" and "s"), a delimiter is one of the punctuation marks that carry
# sentiment ("!", "?", ...) or that end a joke (".", "!")
_WORD_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)*|[!?.,;:]")


def clean_text(text):
    '''Normalise one raw document: drop the HTML line breaks of the IMDb pages, lower case'''
    text = str(text)
    text = text.replace('<br />', ' ').replace('<br/>', ' ').replace('<br>', ' ')
    text = unicodedata.normalize('NFKC', text)
    return text.lower()


def tokenize(text):
    '''Split one raw document into a list of tokens'''
    return _WORD_PATTERN.findall(clean_text(text))


class Vocabulary:
    '''
    A token <-> index mapping built from a collection of tokenised documents.

    The four special tokens always occupy the indices 0..3, so `PAD_INDEX` can be handed to
    `nn.Embedding(padding_idx=...)` and the padding never contributes to the loss.
    '''

    def __init__(self, min_freq=2, max_size=None, special_tokens=None):
        self.min_freq = int(min_freq)
        self.max_size = None if max_size is None else int(max_size)
        self.special_tokens = list(special_tokens if special_tokens else SPECIAL_TOKENS)
        self.token_to_index = {}
        self.index_to_token = []
        for token in self.special_tokens:
            self._add(token)

    # ------------------------------------------------------------------ building
    def _add(self, token):
        if token not in self.token_to_index:
            self.token_to_index[token] = len(self.index_to_token)
            self.index_to_token.append(token)

    def build(self, token_sequences):
        '''
        Add every token that occurs at least `min_freq` times.

        The tokens are ordered by decreasing frequency and then alphabetically, so two runs
        that read the same files build exactly the same vocabulary (the file order of the
        data set does not influence the result).
        '''
        counter = {}
        for tokens in token_sequences:
            for token in tokens:
                counter[token] = counter.get(token, 0) + 1

        candidates = [(token, count) for token, count in counter.items()
                      if count >= self.min_freq and token not in self.token_to_index]
        candidates.sort(key=lambda pair: (-pair[1], pair[0]))
        if self.max_size is not None:
            room = max(0, self.max_size - len(self.index_to_token))
            candidates = candidates[:room]
        for token, _ in candidates:
            self._add(token)
        return self

    # ------------------------------------------------------------------ encoding
    def encode(self, tokens, add_bos=False, add_eos=False):
        '''Convert a list of tokens into a list of indices, unknown tokens become <unk>'''
        indices = [self.token_to_index.get(token, UNK_INDEX) for token in tokens]
        if add_bos:
            indices = [BOS_INDEX] + indices
        if add_eos:
            indices = indices + [EOS_INDEX]
        return indices

    def decode(self, indices, skip_special=True):
        '''Convert a list of indices back into a list of tokens'''
        tokens = []
        for index in indices:
            index = int(index)
            if index < 0 or index >= len(self.index_to_token):
                continue
            token = self.index_to_token[index]
            if skip_special and token in self.special_tokens:
                continue
            tokens.append(token)
        return tokens

    def decode_text(self, indices):
        '''Convert a list of indices into a readable string (special tokens are dropped)'''
        return detokenize(self.decode(indices))

    def __len__(self):
        return len(self.index_to_token)

    # ------------------------------------------------------------------ persistence
    def to_dict(self):
        return {'min_freq': self.min_freq,
                'max_size': self.max_size,
                'special_tokens': list(self.special_tokens),
                'index_to_token': list(self.index_to_token)}

    @classmethod
    def from_dict(cls, state):
        vocabulary = cls(min_freq=state['min_freq'], max_size=state['max_size'],
                         special_tokens=state['special_tokens'])
        for token in state['index_to_token']:
            vocabulary._add(token)
        return vocabulary


def detokenize(tokens):
    '''Join tokens into a string, without a space before a punctuation mark'''
    text = ''
    for token in tokens:
        if not text:
            text = token
        elif token in '.,;:!?':
            text += token
        else:
            text += ' ' + token
    return text


def pad_sequences(sequences, pad_value=PAD_INDEX):
    '''Pad a list of index lists to the length of the longest one (right padding)'''
    import torch
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.int64)
    return torch.nn.utils.rnn.pad_sequence([torch.as_tensor(sequence, dtype=torch.int64)
                                            for sequence in sequences],
                                           batch_first=True, padding_value=pad_value), lengths
