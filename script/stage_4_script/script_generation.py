'''
Stage 4-4 / 4-5: train the recurrent joke generators and sample jokes from three-word prompts.

Every configuration is a word level language model trained on the 1,622 jokes of the
text-generation data set.  After training, each model continues a list of three-word prompts:
the first prompts are the first three words of real jokes (so the reference continuation is
known), the later ones are invented by us and do not appear in the corpus.

Usage
    python -m script.stage_4_script.script_generation
    python -m script.stage_4_script.script_generation --models gru --epochs 2
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import argparse
import json
import os
import sys
import time

import numpy as np

from local_code.stage_4_code.Dataset_Loader_Generation import Dataset_Loader_Generation
from local_code.stage_4_code.Evaluate_Generation import Evaluate_Generation
from local_code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_Stage_4 import Setting_Generation

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# the three recurrent units compared in Stage 4-5, plus one deeper LSTM.
# The learning rate and the dropout were chosen on the validation split (see the report).
CONFIGURATIONS = [
    {'tag': 'rnn', 'rnn_type': 'rnn', 'embedding_dim': 128, 'hidden_size': 256,
     'num_layers': 1, 'dropout': 0.5, 'learning_rate': 3e-3},
    {'tag': 'gru', 'rnn_type': 'gru', 'embedding_dim': 128, 'hidden_size': 256,
     'num_layers': 1, 'dropout': 0.5, 'learning_rate': 3e-3},
    {'tag': 'lstm', 'rnn_type': 'lstm', 'embedding_dim': 128, 'hidden_size': 256,
     'num_layers': 1, 'dropout': 0.5, 'learning_rate': 3e-3},
    {'tag': 'lstm_2layer', 'rnn_type': 'lstm', 'embedding_dim': 128, 'hidden_size': 256,
     'num_layers': 2, 'dropout': 0.5, 'learning_rate': 3e-3},
]

# prompts invented by us: they are grammatical three-word starts that are not a joke opening
INVENTED_PROMPTS = ['my cat thinks', 'the moon said', 'grandma finally told']


def build_method(method_class, configuration, **kwargs):
    '''Build a generator whose class attributes are the settings of one configuration'''
    attributes = {key: value for key, value in configuration.items() if key != 'tag'}
    configured_class = type('Configured' + method_class.__name__, (method_class,), attributes)
    return configured_class(**kwargs)


def jsonable(value):
    '''Convert numpy values into plain Python values so that json can dump them'''
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def pick_data_prompts(jokes, count=4):
    '''
    Take three-word openings of real jokes.

    The jokes of the data set are sorted by their identifier, so the first ones are not
    representative of the whole corpus; several positions are sampled instead and a joke is
    only used when its opening words are distinctive enough to be a fair prompt.
    '''
    prompts, references = [], []
    positions = [0, 37, 149, 311, 528, 742, 905, 1130, 1288, 1465]
    for position in positions:
        if position >= len(jokes) or len(prompts) >= count:
            continue
        joke = jokes[position]
        words = joke.split()
        if len(words) < 12:
            continue
        opening = words[:3]
        # only keep clean openings: three plain lowercase-able words, no punctuation
        if not all(word.replace("'", '').isalpha() for word in opening):
            continue
        prompts.append(' '.join(opening))
        references.append(joke)
    return prompts, references


def main(argv=None):
    parser = argparse.ArgumentParser(description='Stage 4 text generation experiments')
    parser.add_argument('--models', default='rnn,gru,lstm,lstm_2layer')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-new-tokens', type=int, default=40)
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--top-k', type=int, default=30)
    parser.add_argument('--repetition-penalty', type=float, default=1.1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--tag-prefix', default='')
    parser.add_argument('--output-folder',
                        default=os.path.join(_PROJECT_ROOT, 'result', 'stage_4_result',
                                             'generation'))
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    wanted = [name.strip() for name in args.models.split(',') if name.strip()]
    configurations = [config for config in CONFIGURATIONS if config['tag'] in wanted]
    if not configurations:
        raise SystemExit('no configuration matches --models ' + args.models)
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    data_loader = Dataset_Loader_Generation()
    data_loader.load()
    data_loader.print_dataset_information()

    data_prompts, references = pick_data_prompts(data_loader.data['train']['text'])
    prompts = data_prompts + INVENTED_PROMPTS
    print('\nprobe prompts:')
    for prompt, reference in zip(data_prompts, references):
        print('  [in data] ' + prompt + '  ->  ' + reference)
    for prompt in INVENTED_PROMPTS:
        print('  [invented] ' + prompt)

    summary = []
    for configuration in configurations:
        tag = str(args.tag_prefix) + configuration['tag']
        print('\n' + '=' * 78)
        print('running configuration ' + tag + ': ' + str(configuration))
        print('=' * 78)

        model = build_method(Method_RNN_Generation, configuration,
                             mName='Method_RNN_Generation_' + tag,
                             mDescription='recurrent joke generator',
                             vocab_size=len(data_loader.vocabulary))
        model.max_epoch = int(args.epochs)
        model.batch_size = int(args.batch_size)
        model.device = args.device
        model.seed = 1

        result_saver = Result_Saver(rName='generation_' + tag, rType='pickle')
        result_saver.result_destination_folder_path = args.output_folder
        result_saver.result_destination_file_name = tag + '_result.pkl'

        setting = Setting_Generation(sName='Setting_Generation_' + tag,
                                     sDescription='train / validation / sampling')
        setting.prepare(data_loader, model, result_saver, Evaluate_Generation())
        setting.prompts = prompts
        setting.max_new_tokens = int(args.max_new_tokens)
        setting.temperature = float(args.temperature)
        setting.top_k = int(args.top_k)
        setting.repetition_penalty = float(args.repetition_penalty)

        start_time = time.time()
        outcome = setting.load_run_save_evaluate()
        wall_time = time.time() - start_time

        learned = outcome['result']
        metrics = outcome['metrics']
        samples = learned['samples']

        print('\ngenerated jokes of configuration ' + tag + ':')
        for sample, prompt in zip(samples, prompts):
            mark = 'in data ' if prompt in data_prompts else 'invented'
            print('  [' + mark + '] ' + sample['text'])

        row = {'tag': tag,
               'config': jsonable(learned['config']),
               'metrics': jsonable(metrics),
               'validation_metrics': jsonable(learned['validation_metrics']),
               'training_time': float(learned['training_time']),
               'wall_time': float(wall_time),
               'samples': jsonable(samples),
               'prompts': list(prompts),
               'data_prompts': list(data_prompts),
               'reference_jokes': list(references),
               'history': jsonable(learned['history'])}
        summary.append(row)
        with open(os.path.join(args.output_folder, 'summary' + args.tag_prefix + '.json'), 'w',
                  encoding='utf-8') as file:
            json.dump(summary, file, indent=2)
        print('--> ' + tag + ': validation perplexity '
              + str(round(learned['validation_metrics']['perplexity'], 2))
              + ', ' + str(round(learned['training_time'], 1)) + 's')

    print('\n' + '=' * 78)
    print('generation summary')
    print('=' * 78)
    print('%-14s %10s %10s %12s %12s %10s' % ('configuration', 'val PPL', 'token acc',
                                              'nearest Jaccard', 'verbatim span', 'time(s)'))
    for row in summary:
        print('%-14s %10.2f %10.4f %12.4f %12.4f %10.1f'
              % (row['tag'], row['validation_metrics']['perplexity'],
                 row['validation_metrics']['token_accuracy'],
                 row['metrics']['nearest_jaccard_mean'],
                 row['metrics']['verbatim_span_rate'], row['training_time']))
    return summary


if __name__ == '__main__':
    main()
