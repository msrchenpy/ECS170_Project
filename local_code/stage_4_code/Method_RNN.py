'''
Concrete method class for the Stage 4-2 / 4-3 text classification task.

The model is an embedding layer followed by a recurrent encoder and a linear classifier:

    tokens -> Embedding(vocab, E) -> Dropout -> RNN / GRU / LSTM (H, L layers)
           -> pooling over the time steps -> Dropout -> Linear(H', 2) -> logits

`rnn_type` selects the recurrent unit ('rnn', 'gru' or 'lstm'), every other setting of the
network is shared, so the ablation study of Stage 4-5 changes exactly one thing at a time.

Three design choices are worth pointing out:

  * The reviews have different lengths, so every mini batch is padded to the longest review
    of that batch and the real lengths are passed to the encoder, which keeps the state of the
    padding positions frozen.  The padding therefore never enters the recurrence and never
    influences the gradient.
  * `pooling` decides how the sequence of hidden states is turned into one vector.  'last'
    uses the final state (the classic many-to-one RNN), 'mean' and 'max' pool over all time
    steps with a mask that hides the padding.
  * The testing set is never used to pick the number of epochs: the training split is
    further divided into a training and a validation subset, the validation accuracy drives
    the early stopping (the validation loss breaks ties), and the official testing set is
    evaluated once at the end.
'''

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from local_code.base_class.method import method
from local_code.stage_4_code.Recurrent_Network import Recurrent_Encoder, check_rnn_type
from local_code.stage_4_code.Text_Utils import Vocabulary, pad_sequences, tokenize


class _IndexDataset(Dataset):
    '''A minimal dataset that returns (token indices, label) pairs'''

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], int(self.y[index])


def _collate_batch(batch):
    '''Pad the reviews of one mini batch to the longest review of that batch'''
    sequences = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.int64)
    padded, lengths = pad_sequences(sequences)
    return padded, lengths, labels


class Method_RNN(method, nn.Module):
    # ---- architecture --------------------------------------------------------------
    rnn_type = 'rnn'            # 'rnn' | 'gru' | 'lstm'
    embedding_dim = 128
    hidden_size = 128
    num_layers = 1
    bidirectional = False
    pooling = 'last'            # 'last' | 'mean' | 'max'
    embedding_dropout = 0.2
    dropout = 0.3
    freeze_embedding = False

    # ---- vocabulary / labels (set by the script before the model is built) ----------
    vocab_size = None
    n_class = 2
    pad_index = 0

    # ---- optimisation --------------------------------------------------------------
    learning_rate = 1e-3
    weight_decay = 0.0
    batch_size = 64
    max_epoch = 8
    grad_clip = 1.0
    early_stop_patience = 3
    seed = 1
    verbose = True

    # ---- recorded training process -------------------------------------------------
    history = None
    best_epoch = None

    def __init__(self, mName=None, mDescription=None, vocab_size=None, n_class=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        if vocab_size is not None:
            self.vocab_size = int(vocab_size)
        if n_class is not None:
            self.n_class = int(n_class)
        if self.vocab_size is None:
            raise ValueError('the vocabulary size must be set before the model is built')

        self.embedding = nn.Embedding(int(self.vocab_size), int(self.embedding_dim),
                                      padding_idx=int(self.pad_index))
        self.embedding.weight.requires_grad_(not bool(self.freeze_embedding))
        self.recurrent = Recurrent_Encoder(self.rnn_type, self.embedding_dim, self.hidden_size,
                                           num_layers=self.num_layers,
                                           bidirectional=self.bidirectional,
                                           dropout=self.dropout)
        self.classifier = nn.Linear(self.recurrent.output_size(), int(self.n_class))
        self.loss_function = nn.CrossEntropyLoss()

        self.history = {'epoch': [], 'train_loss': [], 'train_accuracy': [],
                        'val_loss': [], 'val_accuracy': [], 'epoch_time': []}
        self.best_epoch = None

        # put the parameters of the network on the GPU (when one is available)
        self.device_request = None
        self._device = torch.device('cpu')
        self.device = None

    # ------------------------------------------------------------------ device
    def _resolve_device(self):
        requested = self.device_request
        if isinstance(requested, str) and requested:
            return torch.device(requested)
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self.device_request = value
        self._device = self._resolve_device()
        self.to(self._device)

    def device_summary(self):
        name = ''
        if self.device.type == 'cuda':
            try:
                name = torch.cuda.get_device_name(self.device)
            except Exception:
                name = ''
        return str(self.device) + (' (' + name + ')' if name else '')

    def count_parameters(self):
        return int(sum(parameter.numel() for parameter in self.parameters()
                       if parameter.requires_grad))

    # ------------------------------------------------------------------ forward
    def forward(self, x, lengths=None):
        '''
        Forward propagation.

        x       : [batch, time] token indices, right padded with `pad_index`
        lengths : [batch] number of real tokens of every review (inferred from x if omitted)
        '''
        if lengths is None:
            lengths = (x != int(self.pad_index)).sum(dim=1)
        lengths = torch.clamp(lengths, min=1)

        embedded = self.embedding(x)
        embedded = nn.functional.dropout(embedded, p=float(self.embedding_dropout),
                                         training=self.training)

        output, _ = self.recurrent(embedded, lengths)

        pooling = str(self.pooling).lower()
        if pooling == 'last':
            # the state of the last real token of every review (the backward direction of a
            # bidirectional layer contributes its state at the same position)
            index = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, output.size(-1))
            features = output.gather(1, index).squeeze(1)
        elif pooling in ('mean', 'max'):
            mask = (x != int(self.pad_index)).unsqueeze(-1)
            if pooling == 'mean':
                features = (output * mask).sum(dim=1) / lengths.unsqueeze(1).to(output.dtype)
            else:
                features = output.masked_fill(~mask, -1e9).max(dim=1)[0]
        else:
            raise ValueError('unsupported pooling: ' + str(self.pooling))

        features = nn.functional.dropout(features, p=float(self.dropout), training=self.training)
        return self.classifier(features)

    # ------------------------------------------------------------------ training
    def _evaluate_split(self, X, y):
        '''Loss and accuracy of one split, without updating any parameter'''
        if X is None or len(X) == 0:
            return float('nan'), float('nan')
        self.eval()
        loader = DataLoader(_IndexDataset(X, y), batch_size=max(self.batch_size, 256),
                            shuffle=False, collate_fn=_collate_batch)
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch_X, lengths, batch_y in loader:
                batch_X = batch_X.to(self.device)
                lengths = lengths.to(self.device)
                batch_y = batch_y.to(self.device)
                logits = self.forward(batch_X, lengths)
                total_loss += self.loss_function(logits, batch_y).item() * batch_X.shape[0]
                correct += (logits.argmax(dim=1) == batch_y).sum().item()
                total += batch_X.shape[0]
        return total_loss / total, correct / total

    def train_model(self, X, y, X_val=None, y_val=None):
        '''Mini-batch training with Adam, gradient clipping and early stopping'''
        torch.manual_seed(int(self.seed))
        np.random.seed(int(self.seed))

        optimizer = torch.optim.Adam([parameter for parameter in self.parameters()
                                      if parameter.requires_grad],
                                     lr=float(self.learning_rate),
                                     weight_decay=float(self.weight_decay))
        loader = DataLoader(_IndexDataset(X, y), batch_size=int(self.batch_size),
                            shuffle=True, collate_fn=_collate_batch, drop_last=False)

        best_val_accuracy = -float('inf')
        best_val_loss = float('inf')
        best_state = None
        patience_count = 0

        for epoch in range(1, int(self.max_epoch) + 1):
            epoch_start = time.time()
            nn.Module.train(self)
            total_loss, correct, total = 0.0, 0, 0
            for batch_X, lengths, batch_y in loader:
                batch_X = batch_X.to(self.device)
                lengths = lengths.to(self.device)
                batch_y = batch_y.to(self.device)

                logits = self.forward(batch_X, lengths)
                loss = self.loss_function(logits, batch_y)

                # check here for the gradient init doc:
                # https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                optimizer.zero_grad()
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.parameters(), float(self.grad_clip))
                optimizer.step()

                total_loss += loss.item() * batch_X.shape[0]
                correct += (logits.argmax(dim=1) == batch_y).sum().item()
                total += batch_X.shape[0]

            train_loss, train_accuracy = total_loss / total, correct / total
            val_loss, val_accuracy = self._evaluate_split(X_val, y_val)

            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_accuracy)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            self.history['epoch_time'].append(time.time() - epoch_start)

            if self.verbose:
                print('Epoch: ' + str(epoch) + '/' + str(self.max_epoch)
                      + ', train loss: ' + str(round(train_loss, 4))
                      + ', train accuracy: ' + str(round(train_accuracy, 4))
                      + ', validation loss: ' + str(round(val_loss, 4))
                      + ', validation accuracy: ' + str(round(val_accuracy, 4))
                      + ', ' + str(round(self.history['epoch_time'][-1], 1)) + 's')

            if X_val is not None and self.early_stop_patience and self.early_stop_patience > 0:
                # this is a classification task, so the checkpoint that is kept is the one with
                # the best validation ACCURACY; the validation loss only breaks ties, because
                # it keeps moving after the accuracy has flattened out
                improved = (val_accuracy > best_val_accuracy + 1e-6) or (
                    abs(val_accuracy - best_val_accuracy) <= 1e-6 and val_loss < best_val_loss - 1e-6)
                if improved:
                    best_val_accuracy = val_accuracy
                    best_val_loss = val_loss
                    best_state = {key: value.detach().clone()
                                  for key, value in self.state_dict().items()}
                    self.best_epoch = epoch
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= int(self.early_stop_patience):
                        print('early stopping at epoch ' + str(epoch) + ', restoring epoch '
                              + str(self.best_epoch))
                        break

        if X_val is not None and best_state is not None:
            self.load_state_dict(best_state)
        elif X_val is not None and self.best_epoch is None:
            self.best_epoch = int(np.argmax(self.history['val_accuracy'])) + 1

    # The template calls the training entry point `train`, while torch's nn.Module also has
    # a `train(mode)` function (nn.Module.eval() calls self.train(False)).  This method
    # accepts both call styles, exactly like the Stage 2 method does.
    def train(self, X=None, y=None):
        if isinstance(X, bool):
            return nn.Module.train(self, X)
        if X is None:
            raise TypeError('train() expects the training data, use train_model() instead')
        return self.train_model(X, y)

    # ------------------------------------------------------------------ inference
    def test(self, X):
        '''Predict the labels of a list of token index sequences'''
        return self.predict_proba(X).argmax(dim=1)

    def predict_proba(self, X):
        '''Return the class probability distribution of every review'''
        self.eval()
        loader = DataLoader(_IndexDataset(X, [0] * len(X)), batch_size=max(self.batch_size, 256),
                            shuffle=False, collate_fn=_collate_batch)
        probabilities = []
        with torch.no_grad():
            for batch_X, lengths, _ in loader:
                logits = self.forward(batch_X.to(self.device), lengths.to(self.device))
                probabilities.append(torch.softmax(logits, dim=1).cpu())
        return torch.cat(probabilities, dim=0)

    def predict_text(self, texts, vocabulary=None):
        '''Classify raw review strings, used by the qualitative examples of the report'''
        vocabulary = vocabulary or self.vocabulary
        if vocabulary is None:
            raise ValueError('a vocabulary is needed to tokenise raw text')
        indices = [vocabulary.encode(tokenize(text))[:self.max_length] or [1] for text in texts]
        probabilities = self.predict_proba(indices)
        return probabilities.argmax(dim=1).numpy(), probabilities.numpy()

    def config_summary(self):
        return {'task': 'text classification',
                'rnn_type': check_rnn_type(self.rnn_type),
                'embedding_dim': int(self.embedding_dim),
                'hidden_size': int(self.hidden_size),
                'num_layers': int(self.num_layers),
                'bidirectional': bool(self.bidirectional),
                'pooling': str(self.pooling),
                'dropout': float(self.dropout),
                'embedding_dropout': float(self.embedding_dropout),
                'learning_rate': float(self.learning_rate),
                'weight_decay': float(self.weight_decay),
                'batch_size': int(self.batch_size),
                'max_epoch': int(self.max_epoch),
                'early_stop_patience': int(self.early_stop_patience),
                'trainable_parameters': self.count_parameters(),
                'device': self.device_summary()}

    def run(self):
        print('method running...')
        print('training device: ' + self.device_summary() + ', torch ' + torch.__version__)
        self.vocabulary = self.data.get('vocabulary')

        train_split = self.data['train']
        validation_split = self.data.get('validation')
        test_split = self.data['test']

        print('--start training...')
        start_time = time.time()
        self.train_model(train_split['X'], train_split['y'],
                         validation_split['X'] if validation_split else None,
                         validation_split['y'] if validation_split else None)
        self.method_training_time = time.time() - start_time

        print('--start testing...')
        test_start_time = time.time()
        pred_y = self.test(test_split['X'])
        pred_proba = self.predict_proba(test_split['X'])
        self.method_testing_time = time.time() - test_start_time
        test_loss, test_accuracy = self._evaluate_split(test_split['X'], test_split['y'])

        print('training time: ' + str(round(self.method_training_time, 2)) + 's')
        print('testing accuracy: ' + str(round(test_accuracy, 4)))
        return {'pred_y': pred_y.numpy(),
                'true_y': np.asarray(test_split['y']),
                'pred_proba': pred_proba.numpy(),
                'history': self.history,
                'config': self.config_summary(),
                'training_time': self.method_training_time,
                'testing_time': self.method_testing_time,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'best_epoch': self.best_epoch}
