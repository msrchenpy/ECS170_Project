'''
Concrete dataset class that loads the Stage 4 IMDb movie review sentiment data set.

The archive contains 25,000 training and 25,000 testing reviews; every split has a `pos`
and a `neg` folder with one text file per review, named `<id>_<rating>.txt`, where the
rating is the number of stars (1..10) the review was given on IMDb.  Following the data
set description only the polarity is used as the label (neg: rating <= 4, pos: rating >= 7),
the reviews with a neutral rating are not part of the archive.

Reading 50,000 small text files and building the vocabulary takes a while, so the parsed
token indices are cached in a pickle file under `result/stage_4_result/cache/`.  The cache
is keyed by the tokenisation settings, so changing `max_length`, `min_freq` or
`max_vocab_size` simply triggers a rebuild instead of silently reusing stale numbers.
'''

import os
import pickle
import random
import re

import numpy as np

from local_code.base_class.dataset import dataset
from local_code.stage_4_code.Text_Utils import Vocabulary, tokenize

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Dataset_Loader_Classification(dataset):
    dataset_name = 'IMDb Movie Review Sentiment'
    dataset_description = ('25,000 training and 25,000 testing movie reviews, labelled with '
                           'their binary sentiment polarity')
    dataset_source_folder_path = r'D:/dl/Compressed/stage_4_dataa/text_classification'
    dataset_source_file_name = ''

    # ---- tokenisation settings -----------------------------------------------------
    max_length = 300          # reviews are truncated to this many tokens (mean review ~230)
    min_freq = 2              # a token must appear at least twice in the training set
    max_vocab_size = 30000
    use_cache = True
    cache_folder = os.path.join(_PROJECT_ROOT, 'result', 'stage_4_result', 'cache')
    cache_file_stem = 'imdb_tokens'

    label_names = ['neg', 'pos']

    def __init__(self, dName=None, dDescription=None):
        dataset.__init__(self, dName, dDescription)
        if not self.dataset_name:
            self.dataset_name = 'Stage 4 text classification (IMDb reviews)'
        if not self.dataset_descrition:
            self.dataset_descrition = ('25000 training and 25000 testing movie reviews, '
                                       'binary sentiment')
        self.vocabulary = None
        self.statistics = {}

    # ------------------------------------------------------------------ reading
    @staticmethod
    def _rating_from_file_name(file_name):
        match = re.search(r'_(\d+)\.txt$', file_name)
        return int(match.group(1)) if match else -1

    def _read_split(self, split):
        '''Read one split, return the token lists, the labels and the star ratings'''
        token_sequences = []
        labels = []
        ratings = []
        for label_index, label_name in enumerate(self.label_names):
            folder = os.path.join(self.dataset_source_folder_path, split, label_name)
            file_names = sorted([name for name in os.listdir(folder) if name.endswith('.txt')],
                                key=lambda name: int(name.split('_')[0]))
            for file_name in file_names:
                with open(os.path.join(folder, file_name), 'r', encoding='utf-8',
                          errors='ignore') as file:
                    raw_text = file.read()
                token_sequences.append(tokenize(raw_text)[:self.max_length])
                labels.append(label_index)
                ratings.append(self._rating_from_file_name(file_name))
        return token_sequences, np.asarray(labels, dtype=np.int64), np.asarray(ratings, dtype=np.int64)

    # ------------------------------------------------------------------ cache
    def _cache_path(self):
        file_name = (self.cache_file_stem + '_len' + str(self.max_length)
                     + '_minfreq' + str(self.min_freq) + '_vocab' + str(self.max_vocab_size)
                     + '.pkl')
        return os.path.join(self.cache_folder, file_name)

    def _load_cache(self):
        if not self.use_cache:
            return None
        path = self._cache_path()
        if not os.path.exists(path):
            return None
        with open(path, 'rb') as file:
            state = pickle.load(file)
        print('  loaded the cached token indices from ' + path)
        return state

    def _save_cache(self, state):
        if not self.use_cache:
            return
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)
        path = self._cache_path()
        with open(path, 'wb') as file:
            pickle.dump(state, file)
        print('  cached the token indices in ' + path)

    # ------------------------------------------------------------------ loading
    def load(self):
        if getattr(self, 'data', None) is not None:
            return self.data
        print('loading the IMDb sentiment data set from ' + self.dataset_source_folder_path + ' ...')
        state = self._load_cache()

        if state is None:
            train_tokens, train_labels, train_ratings = self._read_split('train')
            test_tokens, test_labels, test_ratings = self._read_split('test')
            print('  read ' + str(len(train_tokens)) + ' training and ' + str(len(test_tokens))
                  + ' testing reviews')

            # the vocabulary is built from the *training* split only, the testing reviews may
            # contain words the model has never seen (those become <unk>)
            self.vocabulary = Vocabulary(min_freq=self.min_freq, max_size=self.max_vocab_size)
            self.vocabulary.build(train_tokens)
            print('  vocabulary size: ' + str(len(self.vocabulary)))

            state = {'vocabulary': self.vocabulary.to_dict(),
                     'train': {'X': [self.vocabulary.encode(tokens) for tokens in train_tokens],
                               'y': train_labels,
                               'rating': train_ratings,
                               'length': np.asarray([len(tokens) for tokens in train_tokens],
                                                    dtype=np.int64)},
                     'test': {'X': [self.vocabulary.encode(tokens) for tokens in test_tokens],
                              'y': test_labels,
                              'rating': test_ratings,
                              'length': np.asarray([len(tokens) for tokens in test_tokens],
                                                   dtype=np.int64)}}
            self._save_cache(state)
        else:
            self.vocabulary = Vocabulary.from_dict(state['vocabulary'])

        self.data = {'train': state['train'], 'test': state['test']}
        self._compute_statistics()
        return self.data

    def _compute_statistics(self):
        '''Numbers quoted in the data set description of the report'''
        train_length = np.asarray(self.data['train']['length'])
        test_length = np.asarray(self.data['test']['length'])
        self.statistics = {
            'n_train': int(len(self.data['train']['y'])),
            'n_test': int(len(self.data['test']['y'])),
            'n_train_neg': int((self.data['train']['y'] == 0).sum()),
            'n_train_pos': int((self.data['train']['y'] == 1).sum()),
            'n_test_neg': int((self.data['test']['y'] == 0).sum()),
            'n_test_pos': int((self.data['test']['y'] == 1).sum()),
            'vocab_size': int(len(self.vocabulary)),
            'mean_length_train': float(train_length.mean()),
            'median_length_train': float(np.median(train_length)),
            'mean_length_test': float(test_length.mean()),
            'truncated_train': int((train_length >= self.max_length).sum()),
            'truncated_test': int((test_length >= self.max_length).sum()),
            'total_tokens_train': int(train_length.sum()),
        }

    # ------------------------------------------------------------------ helpers
    def split_train_validation(self, validation_ratio=0.1, seed=1):
        '''
        Split the training reviews into a training and a validation subset.

        The validation subset is used to follow the learning process and to pick the number
        of epochs; the official testing set is only touched once, for the final numbers.
        '''
        n_train = len(self.data['train']['y'])
        indices = list(range(n_train))
        random.Random(seed).shuffle(indices)
        n_validation = int(round(n_train * validation_ratio))
        validation_indices = np.asarray(sorted(indices[:n_validation]), dtype=np.int64)
        training_indices = np.asarray(sorted(indices[n_validation:]), dtype=np.int64)
        return training_indices, validation_indices

    @staticmethod
    def subset(split, indices):
        '''Build a sub split (dict with X / y) from a list of indices'''
        indices = np.asarray(indices, dtype=np.int64)
        return {'X': [split['X'][int(index)] for index in indices],
                'y': split['y'][indices],
                'length': split['length'][indices] if 'length' in split else None}

    def print_dataset_information(self):
        dataset.print_dataset_information(self)
        print('dataset statistics:')
        for key in sorted(self.statistics):
            print('  ' + str(key) + ': ' + str(self.statistics[key]))
