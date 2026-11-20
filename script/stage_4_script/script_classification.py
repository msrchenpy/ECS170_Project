'''
Stage 4-2 / 4-3 / 4-5: train the recurrent text classifiers on the IMDb sentiment data.

The script runs several configurations of `Method_RNN` that differ only in the recurrent unit
(or in one architecture setting for the extra ablation study), evaluates every one of them on
the official testing set and stores the complete run under `result/stage_4_result/`.

Usage
    python -m script.stage_4_script.script_classification
    python -m script.stage_4_script.script_classification --models rnn --epochs 1 --limit-train 400
'''

import argparse
import json
import os
import sys
import time

import numpy as np

from local_code.stage_4_code.Dataset_Loader_Classification import Dataset_Loader_Classification
from local_code.stage_4_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_4_code.Method_RNN import Method_RNN
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_Stage_4 import Setting_Classification

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# the three recurrent units compared in Stage 4-5, plus three extra architecture variants
CONFIGURATIONS = [
    {'tag': 'rnn', 'rnn_type': 'rnn', 'hidden_size': 128, 'embedding_dim': 128,
     'num_layers': 1, 'bidirectional': False, 'pooling': 'last', 'dropout': 0.3},
    {'tag': 'gru', 'rnn_type': 'gru', 'hidden_size': 128, 'embedding_dim': 128,
     'num_layers': 1, 'bidirectional': False, 'pooling': 'last', 'dropout': 0.3},
    {'tag': 'lstm', 'rnn_type': 'lstm', 'hidden_size': 128, 'embedding_dim': 128,
     'num_layers': 1, 'bidirectional': False, 'pooling': 'last', 'dropout': 0.3},
    {'tag': 'gru_bidirectional', 'rnn_type': 'gru', 'hidden_size': 128, 'embedding_dim': 128,
     'num_layers': 1, 'bidirectional': True, 'pooling': 'last', 'dropout': 0.3},
    {'tag': 'lstm_2layer', 'rnn_type': 'lstm', 'hidden_size': 128, 'embedding_dim': 128,
     'num_layers': 2, 'bidirectional': False, 'pooling': 'last', 'dropout': 0.3},
    {'tag': 'gru_meanpool', 'rnn_type': 'gru', 'hidden_size': 128, 'embedding_dim': 128,
     'num_layers': 1, 'bidirectional': False, 'pooling': 'mean', 'dropout': 0.3},
]


def build_method(method_class, configuration, **kwargs):
    '''
    Build a `Method_RNN` whose class attributes are the settings of one configuration.

    The template reads the architecture settings as class attributes before the network is
    created, so a tiny subclass with the configuration as class attributes is the cleanest
    way to instantiate one run per configuration without touching a global.
    '''
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


def main(argv=None):
    parser = argparse.ArgumentParser(description='Stage 4 text classification experiments')
    parser.add_argument('--models', default='rnn,gru,lstm,gru_bidirectional,lstm_2layer,'
                                            'gru_meanpool')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--limit-train', type=int, default=0,
                        help='keep only this many training reviews (0 = all)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--tag-prefix', default='')
    parser.add_argument('--output-folder',
                        default=os.path.join(_PROJECT_ROOT, 'result', 'stage_4_result',
                                             'classification'))
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    wanted = [name.strip() for name in args.models.split(',') if name.strip()]
    configurations = [config for config in CONFIGURATIONS if config['tag'] in wanted]
    if not configurations:
        raise SystemExit('no configuration matches --models ' + args.models)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    data_loader = Dataset_Loader_Classification()
    data_loader.load()

    summary = []
    for configuration in configurations:
        tag = str(args.tag_prefix) + configuration['tag']
        print('\n' + '=' * 78)
        print('running configuration ' + tag + ': ' + str(configuration))
        print('=' * 78)

        model = build_method(Method_RNN, configuration, mName='Method_RNN_' + tag,
                             mDescription='recurrent text classifier',
                             vocab_size=len(data_loader.vocabulary), n_class=2)
        model.max_epoch = int(args.epochs)
        model.batch_size = int(args.batch_size)
        model.device = args.device
        model.seed = 1

        result_saver = Result_Saver(rName='classification_' + tag, rType='pickle')
        result_saver.result_destination_folder_path = args.output_folder
        result_saver.result_destination_file_name = tag + '_result.pkl'

        setting = Setting_Classification(sName='Setting_Classification_' + tag,
                                         sDescription='train / validation / test')
        setting.prepare(data_loader, model, result_saver, Evaluate_Metrics())
        setting.limit_train = int(args.limit_train)

        start_time = time.time()
        outcome = setting.load_run_save_evaluate()
        wall_time = time.time() - start_time

        metrics = outcome['metrics']
        learned = outcome['result']
        row = {'tag': tag,
               'config': jsonable(learned['config']),
               'metrics': jsonable({key: value for key, value in metrics.items()}),
               'training_time': float(learned['training_time']),
               'wall_time': float(wall_time),
               'best_epoch': learned['best_epoch'],
               'test_loss': float(learned['test_loss']),
               'test_accuracy': float(learned['test_accuracy']),
               'n_epochs_run': len(learned['history']['epoch']),
               'history': jsonable(learned['history'])}
        summary.append(row)
        # write the summary after every configuration so that a long run keeps the results
        # of the configurations that already finished
        with open(os.path.join(args.output_folder, 'summary' + args.tag_prefix + '.json'), 'w',
                  encoding='utf-8') as file:
            json.dump(summary, file, indent=2)
        print('--> ' + tag + ': accuracy ' + str(round(metrics['accuracy'], 4))
              + ', macro F1 ' + str(round(metrics['f1_macro'], 4))
              + ', ' + str(round(learned['training_time'], 1)) + 's')

    summary_path = os.path.join(args.output_folder, 'summary' + args.tag_prefix + '.json')
    with open(summary_path, 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)
    print('\nsummary written to ' + summary_path)

    print('\n' + '=' * 78)
    print('classification summary')
    print('=' * 78)
    print('%-20s %8s %8s %8s %8s %10s %8s' % ('configuration', 'acc', 'prec', 'rec', 'f1',
                                               'params', 'time(s)'))
    for row in summary:
        print('%-20s %8.4f %8.4f %8.4f %8.4f %10d %8.1f'
              % (row['tag'], row['metrics']['accuracy'], row['metrics']['precision'],
                 row['metrics']['recall'], row['metrics']['f1'],
                 row['config']['trainable_parameters'], row['training_time']))
    return summary


if __name__ == '__main__':
    main()
