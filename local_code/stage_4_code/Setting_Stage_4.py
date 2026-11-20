'''
Concrete SettingModule classes for the two Stage 4 tasks.

`Setting_Classification` trains one recurrent classifier on the IMDb reviews and evaluates it
on the official testing set, while `Setting_Generation` trains one word level language model
on the jokes and evaluates both its perplexity and the jokes it writes for a list of
three-word prompts.  Both settings save the complete run (history, configuration, metrics and
generated text) so that the report can be rebuilt from the saved results without training
again.
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
from local_code.stage_4_code.Evaluate_Generation import Evaluate_Generation
from local_code.stage_4_code.Evaluate_Metrics import Evaluate_Metrics


class Setting_Classification(setting):
    validation_ratio = 0.1
    split_seed = 1
    limit_train = 0          # keep only this many reviews per split (0 = every review)

    def load_run_save_evaluate(self):
        loaded_data = self.dataset.load()
        print('training reviews: ' + str(len(loaded_data['train']['y']))
              + ', testing reviews: ' + str(len(loaded_data['test']['y'])))

        indices = self.dataset.split_train_validation(self.validation_ratio, self.split_seed)
        self.method.data = {
            'train': self.dataset.subset(loaded_data['train'], indices[0]),
            'validation': self.dataset.subset(loaded_data['train'], indices[1]),
            'test': loaded_data['test'],
            'vocabulary': self.dataset.vocabulary,
        }
        print('training / validation reviews: '
              + str(len(self.method.data['train']['y'])) + ' / '
              + str(len(self.method.data['validation']['y'])))

        if self.limit_train and self.limit_train > 0:
            for key in ('train', 'validation'):
                split = self.method.data[key]
                self.method.data[key] = {'X': split['X'][:self.limit_train],
                                         'y': split['y'][:self.limit_train]}
            print('the training and validation splits were limited to '
                  + str(self.limit_train) + ' reviews')

        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.fold_count = None
        self.result.save()

        self.evaluate.data = learned_result
        metrics = self.evaluate.evaluate()
        self.evaluate.print_metrics(metrics, title=str(self.method.rnn_type) + ' classifier')
        return {'result': learned_result, 'metrics': metrics}


class Setting_Generation(setting):
    prompts = []
    max_new_tokens = 40
    temperature = 0.9
    top_k = 30
    repetition_penalty = 1.1
    sample_seed = 7

    def load_run_save_evaluate(self):
        loaded_data = self.dataset.load()
        print('training jokes: ' + str(len(loaded_data['train']['X']))
              + ', validation jokes: ' + str(len(loaded_data['test']['X'])))

        self.method.data = {'train': loaded_data['train'],
                            'test': loaded_data['test'],
                            'vocabulary': self.dataset.vocabulary}
        learned_result = self.method.run()

        # write a joke for every probe prompt with the same sampling settings for all models
        samples = [self.method.generate(prompt, max_new_tokens=self.max_new_tokens,
                                        temperature=self.temperature, top_k=self.top_k,
                                        repetition_penalty=self.repetition_penalty,
                                        seed=self.sample_seed + index)
                   for index, prompt in enumerate(self.prompts)]
        learned_result['samples'] = samples

        self.evaluate.data = {'samples': samples,
                              'train_texts': loaded_data['train']['text'],
                              'validation_metrics': learned_result['validation_metrics'],
                              'max_new_tokens': self.max_new_tokens}
        metrics = self.evaluate.evaluate()
        self.evaluate.print_metrics(metrics, title=str(self.method.rnn_type) + ' generator')
        learned_result['generation_metrics'] = metrics

        self.result.data = learned_result
        self.result.fold_count = None
        self.result.save()
        return {'result': learned_result, 'metrics': metrics}
