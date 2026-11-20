'''
Concrete Evaluate class for the text generation task of Stage 4-4.

A language model is judged on two different things:

  * how well it models unseen text, measured by the perplexity and the next-word accuracy on
    the validation jokes (a perplexity of p means that the model is, on average, as unsure
    as if it had to choose between p equally likely words), and
  * how good the text it actually writes is.  Since there is no reference continuation for an
    invented prompt, the generated jokes are compared with the *training corpus*:

      - exact match     : the generated joke is one of the training jokes, word for word,
      - nearest overlap : word level Jaccard / longest common subsequence ratio against the
                          closest training joke, which shows whether the output is a plausible
                          joke rather than random words,
      - verbatim span   : the generated text contains a run of 8 words that also occurs in a
                          training joke (a direct sign of memorisation),
      - distinct-1 / 2  : diversity of the generated vocabulary and bigrams.

For the prompts that are the first three words of a training joke the reference continuation
is known, so the token level F1 and the LCS ratio against that continuation are reported too;
that number directly answers "is the generation correct?" for the in-data-set prompts.
'''

import numpy as np

from local_code.base_class.evaluate import evaluate
from local_code.stage_4_code.Text_Utils import clean_text, tokenize


def _jaccard(first, second):
    first, second = set(first), set(second)
    if not first or not second:
        return 0.0
    return len(first & second) / float(len(first | second))


def _lcs_length(first, second):
    '''Length of the longest common subsequence of two token lists'''
    if not first or not second:
        return 0
    previous = [0] * (len(second) + 1)
    for token in first:
        current = [0]
        for index, other in enumerate(second, start=1):
            if token == other:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[index - 1]))
        previous = current
    return previous[-1]


def _lcs_ratio(first, second):
    if not first or not second:
        return 0.0
    return _lcs_length(first, second) / float(max(len(first), len(second)))


def _token_f1(prediction, reference):
    prediction, reference = list(prediction), list(reference)
    if not prediction or not reference:
        return 0.0
    common = _lcs_length(prediction, reference)
    if common == 0:
        return 0.0
    precision = common / float(len(prediction))
    recall = common / float(len(reference))
    return 2 * precision * recall / (precision + recall)


def _has_common_span(tokens, corpus_token_sets, span=8):
    '''Does any `span` long run of `tokens` appear in one of the training jokes?'''
    if len(tokens) < span:
        return False
    for start in range(len(tokens) - span + 1):
        if tuple(tokens[start:start + span]) in corpus_token_sets:
            return True
    return False


def _distinct_n(token_lists, n):
    total, unique = 0, set()
    for tokens in token_lists:
        for start in range(len(tokens) - n + 1):
            unique.add(tuple(tokens[start:start + n]))
            total += 1
    return len(unique) / float(total) if total else 0.0


class Evaluate_Generation(evaluate):
    data = None

    def evaluate(self):
        print('evaluating the generated text...')
        samples = self.data.get('samples', [])
        train_texts = self.data.get('train_texts', [])
        validation_metrics = self.data.get('validation_metrics', {})
        max_new_tokens = int(self.data.get('max_new_tokens', 0))

        train_tokens = [tokenize(text) for text in train_texts]
        train_token_sets = {tuple(tokens[start:start + 8])
                            for tokens in train_tokens
                            for start in range(max(0, len(tokens) - 8 + 1))}
        normalised_train = {clean_text(text).strip() for text in train_texts}

        generated = [tokenize(sample['continuation']) for sample in samples]
        full_texts = [sample['text'] for sample in samples]

        exact_matches = sum(1 for text in full_texts
                            if clean_text(text).strip() in normalised_train)
        nearest_jaccard, nearest_lcs = [], []
        for tokens in generated:
            if not tokens:
                nearest_jaccard.append(0.0)
                nearest_lcs.append(0.0)
                continue
            nearest_jaccard.append(max(_jaccard(tokens, reference) for reference in train_tokens))
            nearest_lcs.append(max(_lcs_ratio(tokens, reference) for reference in train_tokens))

        verbatim = sum(1 for tokens in generated if _has_common_span(tokens, train_token_sets))
        finished = sum(1 for sample in samples
                       if 0 < int(sample.get('n_tokens', 0)) < max_new_tokens)

        # prompts that are the beginning of a training joke have a known continuation
        reference_by_prompt = {}
        for text in train_texts:
            tokens = tokenize(text)
            if len(tokens) > 3:
                reference_by_prompt.setdefault(' '.join(tokens[:3]), tokens[3:])
        reference_f1, reference_lcs, n_reference = [], [], 0
        for sample, tokens in zip(samples, generated):
            reference = reference_by_prompt.get(' '.join(tokenize(sample['prompt'])))
            if reference is None:
                continue
            n_reference += 1
            reference_f1.append(_token_f1(tokens, reference))
            reference_lcs.append(_lcs_ratio(tokens, reference))

        metrics = {
            'perplexity': float(validation_metrics.get('perplexity', float('nan'))),
            'token_accuracy': float(validation_metrics.get('token_accuracy', float('nan'))),
            'n_samples': int(len(samples)),
            'mean_generated_tokens': float(np.mean([len(tokens) for tokens in generated]))
            if generated else 0.0,
            'exact_match_rate': exact_matches / float(len(samples)) if samples else 0.0,
            'nearest_jaccard_mean': float(np.mean(nearest_jaccard)) if nearest_jaccard else 0.0,
            'nearest_lcs_mean': float(np.mean(nearest_lcs)) if nearest_lcs else 0.0,
            'verbatim_span_rate': verbatim / float(len(samples)) if samples else 0.0,
            'finished_rate': finished / float(len(samples)) if samples else 0.0,
            'distinct_1': _distinct_n(generated, 1),
            'distinct_2': _distinct_n(generated, 2),
            'prompt_reference_f1': float(np.mean(reference_f1)) if reference_f1 else float('nan'),
            'prompt_reference_lcs': float(np.mean(reference_lcs)) if reference_lcs else float('nan'),
            'n_prompt_reference': n_reference,
        }
        return metrics

    def print_metrics(self, metrics, title='generation result'):
        print('************ ' + str(title) + ' ************')
        for key in ['perplexity', 'token_accuracy', 'mean_generated_tokens', 'finished_rate',
                    'exact_match_rate', 'nearest_jaccard_mean', 'nearest_lcs_mean',
                    'verbatim_span_rate', 'distinct_1', 'distinct_2', 'prompt_reference_f1',
                    'prompt_reference_lcs']:
            print(str(key).ljust(22) + ': ' + str(round(metrics[key], 4)))
