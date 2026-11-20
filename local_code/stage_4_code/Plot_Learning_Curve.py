'''
Drawing of the Stage 4 learning curves.

The per-epoch numbers recorded in the `history` dictionary of the two methods are written to
a CSV file (so they can be re-plotted or inspected later) and drawn into one PNG per figure.
Every figure has one panel per model, and every panel shows the training curve and the
validation curve of the same quantity: the gap between the two is what tells over-fitting
from under-fitting, which is exactly what the report has to discuss.
'''

import csv
import os

from local_code.stage_4_code.Plot_Figure import line_chart


def save_history_csv(history, csv_path):
    '''Write the per epoch numbers of one run to a CSV file'''
    folder = os.path.dirname(csv_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    keys = list(history.keys())
    length = max([len(history[key]) for key in keys] + [0])
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(keys)
        for index in range(length):
            writer.writerow([history[key][index] if index < len(history[key]) else ''
                             for key in keys])
    return csv_path


def save_learning_curves(histories, output_folder, file_stem, train_key, val_key, y_label,
                         title=None, panel_order=None, y_min=None, y_max=None,
                         figure_size=(1260, 430)):
    '''
    Draw one panel per model.

    histories : dict {model name: history dict}
    train_key : history key of the training series ('train_loss', ...)
    val_key   : history key of the validation series ('val_loss', ...)
    '''
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    names = list(panel_order) if panel_order else list(histories.keys())
    panels = []
    for name in names:
        history = histories[name]
        panels.append({'title': name.upper() if len(name) <= 4 else name,
                       'y_label': y_label,
                       'y_min': y_min, 'y_max': y_max,
                       'series': [{'label': 'training', 'values': history[train_key],
                                   'color': (31, 119, 180)},
                                  {'label': 'validation', 'values': history[val_key],
                                   'color': (214, 39, 40), 'dashed': True}]})
        save_history_csv(history, os.path.join(output_folder,
                                               file_stem + '_' + name + '_history.csv'))
    png_path = os.path.join(output_folder, file_stem + '.png')
    line_chart(panels, png_path, title=title, figure_size=figure_size)
    print('  figure saved to ' + png_path)
    return png_path


def stack_figures(paths, output_path, gap=16, background=(255, 255, 255)):
    '''Stack several PNG figures vertically into one image (report figure with two rows)'''
    from PIL import Image
    images = [Image.open(path).convert('RGB') for path in paths]
    width = max(image.width for image in images)
    height = sum(image.height for image in images) + gap * (len(images) - 1)
    canvas = Image.new('RGB', (width, height), background)
    y = 0
    for image in images:
        canvas.paste(image, ((width - image.width) // 2, y))
        y += image.height + gap
    folder = os.path.dirname(output_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    canvas.save(output_path)
    print('  combined figure saved to ' + output_path)
    return output_path
