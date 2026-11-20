'''
Build every figure of the Stage 4 report from the saved experiment summaries.

The training scripts store one `summary.json` per experiment folder (classification and
generation).  This script reads those files and only draws: nothing is trained again, so the
figures can be rebuilt or restyled without spending GPU time.

Usage
    python -m script.stage_4_script.make_figures
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import argparse
import json
import os

from local_code.stage_4_code.Plot_Figure import architecture_diagram, bar_chart
from local_code.stage_4_code.Plot_Learning_Curve import save_learning_curves, stack_figures

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RESULT_ROOT = os.path.join(_PROJECT_ROOT, 'result', 'stage_4_result')

# the order in which the models appear in the figures
UNIT_ORDER = ['rnn', 'gru', 'lstm']


def load_summary(path):
    if not os.path.exists(path):
        print('  missing ' + path + ', skipped')
        return []
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def to_histories(rows):
    return {row['tag']: row['history'] for row in rows}


def figure_classification(output_folder, summaries):
    rows = [row for row in summaries if row['tag'] in UNIT_ORDER]
    if len(rows) < 2:
        print('  not enough classification runs for the learning curves')
        return
    histories = to_histories(rows)
    loss_png = save_learning_curves(histories, output_folder,
                                    file_stem='classification_loss',
                                    train_key='train_loss', val_key='val_loss',
                                    y_label='loss',
                                    title='Text classification: cross entropy loss per epoch',
                                    panel_order=[tag for tag in UNIT_ORDER if tag in histories])
    accuracy_png = save_learning_curves(histories, output_folder,
                                        file_stem='classification_accuracy',
                                        train_key='train_accuracy', val_key='val_accuracy',
                                        y_label='accuracy', y_min=0.4, y_max=1.0,
                                        title='Text classification: accuracy per epoch',
                                        panel_order=[tag for tag in UNIT_ORDER
                                                     if tag in histories])
    stack_figures([loss_png, accuracy_png],
                  os.path.join(output_folder, 'fig_2_classification_curves.png'), gap=26)


def figure_generation(output_folder, summaries):
    rows = [row for row in summaries if row['tag'] in UNIT_ORDER]
    if len(rows) < 2:
        print('  not enough generation runs for the learning curves')
        return
    histories = to_histories(rows)
    order = [tag for tag in UNIT_ORDER if tag in histories]
    loss_png = save_learning_curves(histories, output_folder,
                                    file_stem='generation_loss',
                                    train_key='train_loss', val_key='val_loss',
                                    y_label='loss',
                                    title='Text generation: next word loss per epoch',
                                    panel_order=order)
    perplexity_png = save_learning_curves(histories, output_folder,
                                          file_stem='generation_perplexity',
                                          train_key='train_perplexity',
                                          val_key='val_perplexity',
                                          y_label='perplexity',
                                          title='Text generation: perplexity per epoch',
                                          panel_order=order)
    stack_figures([loss_png, perplexity_png],
                  os.path.join(output_folder, 'fig_3_generation_curves.png'), gap=26)


def figure_comparison(output_folder, classification_rows, generation_rows):
    '''Bar chart of the testing metrics of every trained classifier'''
    if classification_rows:
        categories = [row['tag'] for row in classification_rows]
        accuracy = [row['metrics']['accuracy'] for row in classification_rows]
        f1 = [row['metrics']['f1_macro'] for row in classification_rows]
        bar_chart(categories,
                  [{'label': 'accuracy', 'values': accuracy, 'color': (31, 119, 180)},
                   {'label': 'macro F1', 'values': f1, 'color': (255, 127, 14)}],
                  os.path.join(output_folder, 'fig_4_classification_comparison.png'),
                  title='Text classification: testing set performance', y_label='score',
                  figure_size=(1100, 520), value_format='%.3f')

    if generation_rows:
        categories = [row['tag'] for row in generation_rows]
        perplexity = [row['validation_metrics']['perplexity'] for row in generation_rows]
        overlap = [row['metrics']['nearest_lcs_mean'] for row in generation_rows]
        first = bar_chart(categories,
                          [{'label': 'validation perplexity', 'values': perplexity,
                            'color': (44, 160, 44)}],
                          os.path.join(output_folder, '_generation_perplexity_bar.png'),
                          title='Text generation: validation perplexity', y_label='perplexity',
                          figure_size=(1100, 380), value_format='%.1f')
        second = bar_chart(categories,
                           [{'label': 'longest common subsequence with nearest joke',
                             'values': overlap, 'color': (148, 103, 189)}],
                           os.path.join(output_folder, '_generation_overlap_bar.png'),
                           title='Text generation: similarity to the closest training joke',
                           y_label='LCS ratio', figure_size=(1100, 380),
                           value_format='%.3f')
        stack_figures([first, second],
                      os.path.join(output_folder, 'fig_5_generation_comparison.png'))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Draw the Stage 4 report figures')
    parser.add_argument('--result-root', default=_RESULT_ROOT)
    parser.add_argument('--output-folder',
                        default=os.path.join(_RESULT_ROOT, 'figures'))
    args = parser.parse_args(argv)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    print('drawing the architecture diagram...')
    architecture_diagram(os.path.join(args.output_folder, 'fig_1_architecture.png'))

    classification = load_summary(os.path.join(args.result_root, 'classification',
                                               'summary.json'))
    ablation = load_summary(os.path.join(args.result_root, 'ablation', 'summary.json'))
    generation = load_summary(os.path.join(args.result_root, 'generation', 'summary.json'))

    print('drawing the classification learning curves...')
    figure_classification(args.output_folder, classification)

    print('drawing the generation learning curves...')
    figure_generation(args.output_folder, generation)

    print('drawing the comparison bars...')
    figure_comparison(args.output_folder, classification + ablation, generation)

    print('figures written to ' + args.output_folder)


if __name__ == '__main__':
    main()
