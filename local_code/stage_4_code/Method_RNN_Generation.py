'''
Concrete method class for the Stage 4-4 text generation task.

The model is a word level autoregressive language model:

    previous words -> Embedding(vocab, E) -> RNN / GRU / LSTM -> Linear(H', vocab) -> P(next word)

Training maximises the probability of every next word of every training joke, i.e. it
minimises the cross entropy between the predicted distribution and the observed word.  A
joke is wrapped into `<bos> ... <eos>`, so the model learns both how a joke starts and where
it ends; generation stops when `<eos>` is sampled.

The same class is used for the RNN, the GRU and the LSTM ablation: `rnn_type` is the only
setting that changes.  Generation uses temperature scaling plus top-k (and an optional
repetition penalty), because greedy decoding of a small language model tends to fall into
short loops ("the the the ...").
'''

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from local_code.base_class.method import method
from local_code.stage_4_code.Recurrent_Network import Recurrent_Encoder, check_rnn_type
from local_code.stage_4_code.Text_Utils import (BOS_INDEX, EOS_INDEX, Vocabulary, detokenize,
                                                pad_sequences, tokenize)


class _SequenceDataset(Dataset):
    '''Returns (input indices, target indices) pairs: the sequence shifted by one position'''

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        return list(sequence[:-1]), list(sequence[1:])


def _collate_batch(batch, pad_index):
    '''Pad the inputs with <pad>, the targets with the ignored index (-100)'''
    inputs = [item[0] for item in batch]
    targets = [[index if index != pad_index else -100 for index in item[1]] for item in batch]
    padded_inputs, _ = pad_sequences(inputs, pad_value=pad_index)
    padded_targets, _ = pad_sequences(targets, pad_value=-100)
    return padded_inputs, padded_targets


class Method_RNN_Generation(method, nn.Module):
    # ---- architecture --------------------------------------------------------------
    rnn_type = 'lstm'           # 'rnn' | 'gru' | 'lstm'
    embedding_dim = 128
    hidden_size = 256
    num_layers = 1
    dropout = 0.3
    tie_weights = True          # share the embedding and the output matrix (small corpora)

    # ---- vocabulary (set by the script before the model is built) -------------------
    vocab_size = None
    pad_index = 0

    # ---- optimisation --------------------------------------------------------------
    learning_rate = 3e-3
    weight_decay = 0.0
    batch_size = 64
    max_epoch = 30
    grad_clip = 5.0
    early_stop_patience = 6     # stop after this many epochs without a better validation loss
    seed = 1
    verbose = True

    # ---- recorded training process -------------------------------------------------
    history = None
    best_epoch = None

    def __init__(self, mName=None, mDescription=None, vocab_size=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        if vocab_size is not None:
            self.vocab_size = int(vocab_size)
        if self.vocab_size is None:
            raise ValueError('the vocabulary size must be set before the model is built')

        self.embedding = nn.Embedding(int(self.vocab_size), int(self.embedding_dim),
                                      padding_idx=int(self.pad_index))
        self.recurrent = Recurrent_Encoder(self.rnn_type, self.embedding_dim, self.hidden_size,
                                           num_layers=self.num_layers, dropout=self.dropout)
        # weight tying is only possible when the hidden state has the size of an embedding
        self.tie_weights = bool(self.tie_weights) and int(self.embedding_dim) == int(self.hidden_size)
        output_size = int(self.embedding_dim) if self.tie_weights else int(self.hidden_size)
        self.decoder = nn.Linear(output_size, int(self.vocab_size))
        if self.tie_weights:
            # the same matrix maps words to vectors and vectors back to word scores, which
            # roughly halves the number of parameters of a small language model
            self.decoder.weight = self.embedding.weight

        # -100 is the value PyTorch's cross entropy ignores, the padded target positions use it
        self.loss_function = nn.CrossEntropyLoss(ignore_index=-100)

        self.history = {'epoch': [], 'train_loss': [], 'train_perplexity': [],
                        'val_loss': [], 'val_perplexity': [],
                        'train_token_accuracy': [], 'val_token_accuracy': [],
                        'epoch_time': []}

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
    def forward(self, x, hidden=None):
        '''Return the logits of the next word for every position of the input sequence'''
        embedded = nn.functional.dropout(self.embedding(x), p=float(self.dropout),
                                         training=self.training)
        output, hidden = self.recurrent(embedded, state=hidden)
        output = nn.functional.dropout(output, p=float(self.dropout), training=self.training)
        return self.decoder(output), hidden

    # ------------------------------------------------------------------ training
    def _evaluate_split(self, sequences):
        '''Average loss, perplexity and next-word accuracy of one split'''
        if sequences is None or len(sequences) == 0:
            return float('nan'), float('nan'), float('nan')
        self.eval()
        loader = DataLoader(_SequenceDataset(sequences), batch_size=128, shuffle=False,
                            collate_fn=lambda batch: _collate_batch(batch, self.pad_index))
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                logits, _ = self.forward(batch_X)
                total_loss += self.loss_function(logits.reshape(-1, logits.size(-1)),
                                                 batch_y.reshape(-1)).item() * batch_X.shape[0]
                mask = batch_y.reshape(-1) != -100
                correct += (logits.reshape(-1, logits.size(-1)).argmax(dim=1)[mask]
                            == batch_y.reshape(-1)[mask]).sum().item()
                total += batch_X.shape[0]
        mean_loss = total_loss / total
        return mean_loss, float(np.exp(mean_loss)), correct / max(total, 1)

    def train_model(self, train_sequences, validation_sequences=None):
        '''Mini-batch training with Adam, gradient clipping and a fixed number of epochs'''
        torch.manual_seed(int(self.seed))
        np.random.seed(int(self.seed))

        optimizer = torch.optim.Adam(self.parameters(), lr=float(self.learning_rate),
                                     weight_decay=float(self.weight_decay))
        loader = DataLoader(_SequenceDataset(train_sequences), batch_size=int(self.batch_size),
                            shuffle=True,
                            collate_fn=lambda batch: _collate_batch(batch, self.pad_index))

        best_val_loss = float('inf')
        best_state = None
        patience_count = 0
        patience_limit = int(self.early_stop_patience or 0)

        for epoch in range(1, int(self.max_epoch) + 1):
            epoch_start = time.time()
            nn.Module.train(self)
            total_loss, correct, total = 0.0, 0, 0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                logits, _ = self.forward(batch_X)
                loss = self.loss_function(logits.reshape(-1, logits.size(-1)),
                                          batch_y.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.parameters(), float(self.grad_clip))
                optimizer.step()

                total_loss += loss.item() * batch_X.shape[0]
                mask = batch_y.reshape(-1) != -100
                correct += (logits.reshape(-1, logits.size(-1)).argmax(dim=1)[mask]
                            == batch_y.reshape(-1)[mask]).sum().item()
                total += batch_X.shape[0]

            train_loss = total_loss / total
            val_loss, val_perplexity, val_accuracy = self._evaluate_split(validation_sequences)

            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_perplexity'].append(float(np.exp(train_loss)))
            self.history['train_token_accuracy'].append(correct / max(total, 1))
            self.history['val_loss'].append(val_loss)
            self.history['val_perplexity'].append(val_perplexity)
            self.history['val_token_accuracy'].append(val_accuracy)
            self.history['epoch_time'].append(time.time() - epoch_start)

            if self.verbose:
                print('Epoch: ' + str(epoch) + '/' + str(self.max_epoch)
                      + ', train loss: ' + str(round(train_loss, 4))
                      + ', train perplexity: ' + str(round(float(np.exp(train_loss)), 2))
                      + ', validation loss: ' + str(round(val_loss, 4))
                      + ', validation perplexity: ' + str(round(float(val_perplexity), 2)))

            # the corpus is small, so the model starts to memorise the training jokes after a
            # few epochs; keep the weights of the epoch with the best validation loss and stop
            # when the validation loss has not improved for `early_stop_patience` epochs
            if validation_sequences is not None and patience_limit > 0:
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = float(val_loss)
                    best_state = {key: value.detach().clone()
                                  for key, value in self.state_dict().items()}
                    self.best_epoch = int(epoch)
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= patience_limit:
                        print('early stopping at epoch ' + str(epoch) + ', restoring epoch '
                              + str(self.best_epoch))
                        break

        if best_state is not None:
            self.load_state_dict(best_state)
        elif validation_sequences is not None:
            self.best_epoch = int(np.argmin(self.history['val_loss'])) + 1

    def train(self, X=None, y=None):
        if isinstance(X, bool):
            return nn.Module.train(self, X)
        if X is None:
            raise TypeError('train() expects the training sequences, use train_model() instead')
        return self.train_model(X)

    # ------------------------------------------------------------------ scoring
    def score_texts(self, texts):
        '''Perplexity and next-word accuracy of raw strings, used to evaluate the generator'''
        self.eval()
        sequences = [[BOS_INDEX] + self.vocabulary.encode(tokenize(text)) + [EOS_INDEX]
                     for text in texts]
        total_loss, correct, total = 0.0, 0, 0
        loader = DataLoader(_SequenceDataset(sequences), batch_size=64, shuffle=False,
                            collate_fn=lambda batch: _collate_batch(batch, self.pad_index))
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                logits, _ = self.forward(batch_X)
                total_loss += self.loss_function(logits.reshape(-1, logits.size(-1)),
                                                 batch_y.reshape(-1)).item() * batch_X.shape[0]
                mask = batch_y.reshape(-1) != -100
                correct += (logits.reshape(-1, logits.size(-1)).argmax(dim=1)[mask]
                            == batch_y.reshape(-1)[mask]).sum().item()
                total += batch_X.shape[0]
        mean_loss = total_loss / total
        return {'loss': mean_loss, 'perplexity': float(np.exp(mean_loss)),
                'token_accuracy': correct / max(total, 1)}

    # ------------------------------------------------------------------ generation
    def _apply_repetition_penalty(self, logits, previous_ids, penalty):
        if not penalty or penalty <= 1.0 or not previous_ids:
            return logits
        unique = torch.tensor(sorted(set(previous_ids)), device=logits.device)
        values = logits[0, unique]
        values = torch.where(values > 0, values / float(penalty), values * float(penalty))
        logits = logits.clone()
        logits[0, unique] = values
        return logits

    def _sample(self, logits, temperature, top_k, generator):
        '''Turn one score vector into one sampled word index'''
        logits = logits / max(float(temperature), 1e-6)
        if top_k is not None and 0 < int(top_k) < logits.size(-1):
            values, indices = torch.topk(logits, int(top_k), dim=-1)
            filtered = torch.full_like(logits, float('-inf'))
            filtered.scatter_(1, indices, values)
            logits = filtered
        probabilities = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probabilities, num_samples=1, generator=generator).item())

    @torch.no_grad()
    def generate(self, prompt='', max_new_tokens=40, temperature=1.0, top_k=None,
                 repetition_penalty=1.0, seed=None):
        '''
        Continue a prompt word by word.

        Returns a dictionary with the full text, the generated continuation and the list of
        generated tokens, so the caller can show both the prompt and what the model added.
        '''
        self.eval()
        generator = None
        if seed is not None:
            # the generator must live on the same device as the logits it samples from
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))

        prompt_tokens = tokenize(prompt)
        if self.vocabulary is None:
            raise ValueError('a vocabulary is needed to generate text')
        prompt_ids = self.vocabulary.encode(prompt_tokens)

        input_ids = [BOS_INDEX] + prompt_ids
        x = torch.tensor([input_ids], dtype=torch.int64, device=self.device)
        logits, hidden = self.forward(x)

        generated_ids = []
        for _ in range(int(max_new_tokens)):
            step_logits = self._apply_repetition_penalty(logits[:, -1], prompt_ids + generated_ids,
                                                         repetition_penalty)
            next_id = self._sample(step_logits, temperature, top_k, generator)
            generated_ids.append(next_id)
            if next_id == EOS_INDEX:
                break
            step_input = torch.tensor([[next_id]], dtype=torch.int64, device=self.device)
            logits, hidden = self.forward(step_input, hidden)

        continuation = detokenize(self.vocabulary.decode(generated_ids))
        return {'prompt': prompt,
                'continuation': continuation,
                'text': (detokenize(prompt_tokens) + ' ' + continuation).strip(),
                'tokens': self.vocabulary.decode(generated_ids),
                'n_tokens': len(generated_ids)}

    def generate_batch(self, prompts, **kwargs):
        return [self.generate(prompt, **kwargs) for prompt in prompts]

    # ------------------------------------------------------------------ reporting
    def config_summary(self):
        return {'task': 'text generation',
                'rnn_type': check_rnn_type(self.rnn_type),
                'embedding_dim': int(self.embedding_dim),
                'hidden_size': int(self.hidden_size),
                'num_layers': int(self.num_layers),
                'dropout': float(self.dropout),
                'tie_weights': bool(self.tie_weights),
                'learning_rate': float(self.learning_rate),
                'weight_decay': float(self.weight_decay),
                'batch_size': int(self.batch_size),
                'max_epoch': int(self.max_epoch),
                'trainable_parameters': self.count_parameters(),
                'device': self.device_summary()}

    def run(self):
        print('method running...')
        print('training device: ' + self.device_summary() + ', torch ' + torch.__version__)
        self.vocabulary = self.data.get('vocabulary')

        print('--start training...')
        start_time = time.time()
        self.train_model(self.data['train']['X'], self.data['test']['X'])
        self.method_training_time = time.time() - start_time

        print('--start evaluating...')
        test_start_time = time.time()
        metrics = self.score_texts(self.data['test']['text'])
        self.method_testing_time = time.time() - test_start_time
        print('training time: ' + str(round(self.method_training_time, 2)) + 's')
        print('validation perplexity: ' + str(round(metrics['perplexity'], 2)))

        return {'history': self.history,
                'config': self.config_summary(),
                'training_time': self.method_training_time,
                'testing_time': self.method_testing_time,
                'best_epoch': self.best_epoch,
                'n_epochs_run': len(self.history['epoch']),
                'validation_metrics': metrics}
