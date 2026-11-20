'''
Concrete dataset class that loads the Stage 4 joke data set used for text generation.

The archive is one CSV file (`text_generation/data`, 1,622 rows) with two columns, `ID` and
`Joke`; every row is one single-line joke.  The task of Stage 4-4 is to train a character /
word level language model that continues a joke from its first three words.

The 1,622 jokes are split into 90% training and 10% validation jokes with a fixed seed.  The
validation split is used to follow the learning curves and to compute the reported
perplexity; the generation examples in the report are produced by the model trained on the
90% training split only.
'''

import csv
import os
import random

import numpy as np

from local_code.base_class.dataset import dataset
from local_code.stage_4_code.Text_Utils import BOS_INDEX, EOS_INDEX, Vocabulary, tokenize

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Dataset_Loader_Generation(dataset):
    dataset_name = 'Short Jokes'
    dataset_description = '1,622 one-line jokes, used as a word level text generation corpus'
    dataset_source_folder_path = r'D:/dl/Compressed/stage_4_dataa/text_generation'
    dataset_source_file_name = 'data'

    # ---- tokenisation settings -----------------------------------------------------
    max_length = 64          # tokens per joke, including <bos> and <eos>
    min_freq = 1             # every word of the corpus is kept, the vocabulary is small
    max_vocab_size = None
    validation_ratio = 0.1
    split_seed = 1
    use_cache = True
    cache_folder = os.path.join(_PROJECT_ROOT, 'result', 'stage_4_result', 'cache')
    cache_file_stem = 'jokes_tokens'

    def __init__(self, dName=None, dDescription=None):
        dataset.__init__(self, dName, dDescription)
        if not self.dataset_name:
            self.dataset_name = 'Stage 4 text generation (jokes)'
        if not self.dataset_descrition:
            self.dataset_descrition = ('1622 short jokes, one language modelling sample each')
        self.vocabulary = None
        self.statistics = {}
        self.prompt_examples = []

    def _cache_path(self):
        return os.path.join(self.cache_folder, self.cache_file_stem + '_len'
                            + str(self.max_length) + '_minfreq' + str(self.min_freq) + '.pkl')

    def _read_jokes(self):
        '''Return the list of raw joke strings, in the order of the CSV file'''
        path = os.path.join(self.dataset_source_folder_path, self.dataset_source_file_name)
        jokes = []
        with open(path, 'r', encoding='utf-8', errors='ignore', newline='') as file:
            for row in csv.DictReader(file):
                joke = (row.get('Joke') or '').strip()
                if joke:
                    jokes.append(joke)
        return jokes

    def load(self):
        if getattr(self, 'data', None) is not None:
            return self.data
        print('loading the joke data set from ' + self.dataset_source_folder_path + ' ...')
        state = None
        if self.use_cache and os.path.exists(self._cache_path()):
            import pickle
            with open(self._cache_path(), 'rb') as file:
                state = pickle.load(file)
            print('  loaded the cached token indices from ' + self._cache_path())

        if state is None:
            jokes = self._read_jokes()
            print('  read ' + str(len(jokes)) + ' jokes')

            token_lists = [tokenize(joke) for joke in jokes]
            # a joke that is longer than the limit loses its last words, the <eos> token is
            # appended afterwards so that the model always learns where a joke stops
            token_lists = [tokens[:self.max_length - 2] for tokens in token_lists]

            self.vocabulary = Vocabulary(min_freq=self.min_freq, max_size=self.max_vocab_size)
            self.vocabulary.build(token_lists)
            print('  vocabulary size: ' + str(len(self.vocabulary)))

            sequences = [[BOS_INDEX] + self.vocabulary.encode(tokens) + [EOS_INDEX]
                         for tokens in token_lists]

            indices = list(range(len(jokes)))
            random.Random(self.split_seed).shuffle(indices)
            n_validation = int(round(len(jokes) * self.validation_ratio))
            validation_indices = sorted(indices[:n_validation])
            training_indices = sorted(indices[n_validation:])

            state = {
                'vocabulary': self.vocabulary.to_dict(),
                'jokes': jokes,
                'sequences': sequences,
                'train_index': training_indices,
                'validation_index': validation_indices,
            }
            if self.use_cache:
                import pickle
                if not os.path.exists(self.cache_folder):
                    os.makedirs(self.cache_folder)
                with open(self._cache_path(), 'wb') as file:
                    pickle.dump(state, file)
                print('  cached the token indices in ' + self._cache_path())
        else:
            self.vocabulary = Vocabulary.from_dict(state['vocabulary'])

        jokes = state['jokes']
        sequences = state['sequences']
        self.data = {
            'train': {'X': [sequences[index] for index in state['train_index']],
                      'text': [jokes[index] for index in state['train_index']]},
            'test': {'X': [sequences[index] for index in state['validation_index']],
                     'text': [jokes[index] for index in state['validation_index']]},
        }

        train_lengths = np.asarray([len(sequence) - 2 for sequence in self.data['train']['X']])
        self.statistics = {
            'n_jokes': int(len(jokes)),
            'n_train': int(len(self.data['train']['X'])),
            'n_validation': int(len(self.data['test']['X'])),
            'vocab_size': int(len(self.vocabulary)),
            'mean_tokens_per_joke': float(train_lengths.mean()),
            'max_tokens_per_joke': int(train_lengths.max()),
        }
        # three-word prompts taken from real jokes, used as the "in the data set" examples
        self.prompt_examples = [' '.join(joke.split()[:3]) for joke in self.data['train']['text'][:5]]
        return self.data

    def encode_prompt(self, prompt):
        '''Turn a raw prompt string into the list of indices the model is conditioned on'''
        return self.vocabulary.encode(tokenize(prompt))

    def print_dataset_information(self):
        dataset.print_dataset_information(self)
        print('dataset statistics:')
        for key in sorted(self.statistics):
            print('  ' + str(key) + ': ' + str(self.statistics[key]))
