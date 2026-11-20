'''
Small drawing library used by the Stage 4 figures.

matplotlib is not part of the environment of the machine that produced the results, so the
figures are drawn with Pillow, which is already a dependency of the project (it is used by
Stage 2 as well).  Three primitives are provided:

  * `line_chart`        : one or more panels of line series sharing the same x axis,
  * `bar_chart`         : grouped bars with the value printed above every bar,
  * `architecture_diagram` : the block diagram of the two Stage 4 models.

Everything is drawn on a white background with a light grid, so the figures can be pasted
directly into the report without any post-processing.
'''

import math
import os

from PIL import Image, ImageDraw, ImageFont

PALETTE = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (255, 127, 14),
           (148, 103, 189), (140, 86, 75), (23, 190, 207), (227, 119, 194)]


# --------------------------------------------------------------------------------------
# text helpers
# --------------------------------------------------------------------------------------
def _font(size, bold=False):
    names = (['arialbd.ttf', 'segoeuib.ttf', 'DejaVuSans-Bold.ttf'] if bold
             else ['arial.ttf', 'segoeui.ttf', 'DejaVuSans.ttf'])
    for name in names:
        for folder in ('C:/Windows/Fonts', os.environ.get('WINDIR', 'C:/Windows') + '/Fonts', ''):
            path = os.path.join(folder, name) if folder else name
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def _text(draw, x, y, text, font, fill=(0, 0, 0), anchor='la'):
    '''Draw one line of text; anchor: first letter l/m/r, second letter a (top)'''
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    except Exception:
        left, top, right, bottom = 0, 0, draw.textsize(text, font=font)[0], font.size
    width, height = right - left, bottom - top
    if anchor[0] == 'm':
        x -= width / 2.0
    elif anchor[0] == 'r':
        x -= width
    if anchor[1] == 'm':
        y -= height / 2.0
    elif anchor[1] == 'b':
        y -= height
    draw.text((x - left, y - top), text, font=font, fill=fill)


def _nice_step(span, target=6):
    if span <= 0:
        return 1.0
    raw = span / float(target)
    exponent = int(math.floor(math.log10(raw)))
    fraction = raw / (10.0 ** exponent)
    for candidate in (1.0, 2.0, 2.5, 5.0, 10.0):
        if fraction <= candidate + 1e-9:
            return candidate * (10.0 ** exponent)
    return 10.0 * (10.0 ** exponent)


def _tick_format(step):
    if step >= 1:
        return '%.0f' if abs(step - round(step)) < 1e-9 else '%.1f'
    return '%.' + str(max(0, -int(math.floor(math.log10(step) + 1e-9)))) + 'f'


def _vertical_text(image, text, font, center, fill=(60, 60, 60)):
    '''Draw `text` rotated by 90 degrees, centred on `center`, directly on `image`.'''
    text = str(text)
    if not text:
        return
    bbox = font.getbbox(text)
    patch_width = max(bbox[2] - bbox[0], 1) + 4
    patch_height = max(bbox[3] - bbox[1], 1) + 6
    patch = Image.new('RGBA', (patch_width, patch_height), (255, 255, 255, 0))
    patch_draw = ImageDraw.Draw(patch)
    patch_draw.text((2 - bbox[0], 3 - bbox[1]), text, font=font, fill=fill)
    rotated = patch.rotate(90, expand=True)
    position = (int(round(center[0] - rotated.width / 2.0)),
                int(round(center[1] - rotated.height / 2.0)))
    image.paste(rotated, position, rotated)


# --------------------------------------------------------------------------------------
# line chart
# --------------------------------------------------------------------------------------
def line_chart(panels, png_path, title=None, x_label='epoch', figure_size=(1260, 520)):
    '''
    Draw one or more line-chart panels side by side.

    panels: list of dicts with the keys
        title    : panel title,
        series   : list of dicts {label, values, color (optional), dashed (optional)},
        y_label  : y axis label,
        y_min/y_max : optional fixed axis limits.
    '''
    width, height = figure_size
    image = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font_title = _font(22, bold=True)
    font_panel = _font(19, bold=True)
    font_label = _font(15)
    font_tick = _font(13)

    if title:
        _text(draw, width / 2.0, 14, str(title), font_title, anchor='ma')

    n_panel = max(1, len(panels))
    gap = 70
    margin_left = 88
    margin_right = 28
    panel_width = (width - margin_left - margin_right - gap * (n_panel - 1)) / float(n_panel)
    top = 82
    bottom = height - 62

    for panel_index, panel in enumerate(panels):
        left = margin_left + panel_index * (panel_width + gap)
        right = left + panel_width

        series = [entry for entry in panel.get('series', []) if entry.get('values')]
        values = [float(value) for entry in series for value in entry['values']
                  if value == value]
        if not values:
            continue
        low, high = min(values), max(values)
        non_negative = low >= 0.0
        if high - low < 1e-9:
            low, high = low - 0.5, high + 0.5
        padding = (high - low) * 0.10
        low -= padding
        high += padding
        if non_negative:
            # a metric such as perplexity never becomes negative, so the padding must not
            # push the axis below zero
            low = max(low, 0.0)
        if panel.get('y_min') is not None:
            low = float(panel['y_min'])
        if panel.get('y_max') is not None:
            high = float(panel['y_max'])
        if high - low < 1e-9:
            high = low + 1.0

        step = _nice_step(high - low)
        low = math.floor(round(low / step, 9)) * step
        high = math.ceil(round(high / step, 9)) * step
        n_tick = max(1, int(round((high - low) / step)))
        fmt = _tick_format(step)

        # grid and y axis ticks
        for index in range(n_tick + 1):
            y = bottom - (bottom - top) * index / float(n_tick)
            draw.line([(left, y), (right, y)], fill=(226, 226, 226), width=1)
            value = low + step * index
            if abs(value) < step * 1e-6:
                value = 0.0
            _text(draw, left - 8, y, fmt % value, font_tick, fill=(70, 70, 70), anchor='ra')

        # x axis: one tick per epoch (or every few epochs when there are many)
        lengths = [len(entry['values']) for entry in series]
        n_point = max(lengths) if lengths else 1
        stride = max(1, int(math.ceil(n_point / 9.0)))
        for index in range(0, n_point, stride):
            x = left + (right - left) * index / float(max(n_point - 1, 1))
            draw.line([(x, top), (x, bottom)], fill=(242, 242, 242), width=1)
            _text(draw, x, bottom + 8, str(index + 1), font_tick, fill=(70, 70, 70), anchor='ma')
        if (n_point - 1) % stride:
            # the last epoch always gets a label, even when the stride skips it
            _text(draw, right, bottom + 8, str(n_point), font_tick, fill=(70, 70, 70), anchor='ma')

        draw.rectangle([left, top, right, bottom], outline=(90, 90, 90), width=2)

        for series_index, entry in enumerate(series):
            color = entry.get('color') or PALETTE[series_index % len(PALETTE)]
            points = []
            for index, value in enumerate(entry['values']):
                value = float(value)
                if value != value:
                    continue
                x = left + (right - left) * index / float(max(n_point - 1, 1))
                y = bottom - (bottom - top) * (value - low) / (high - low)
                points.append((x, y))
            if entry.get('dashed'):
                for start in range(0, len(points) - 1):
                    if start % 2 == 0:
                        draw.line([points[start], points[start + 1]], fill=color, width=2)
            elif len(points) > 1:
                draw.line(points, fill=color, width=3, joint='curve')
            for x, y in points:
                draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)

        # panel title and axis labels
        _text(draw, (left + right) / 2.0, top - 32, str(panel.get('title', '')), font_panel,
              anchor='ma')
        _text(draw, (left + right) / 2.0, bottom + 32, x_label, font_label, anchor='ma')
        _vertical_text(image, panel.get('y_label', ''), font_label,
                       (left - 42, (top + bottom) / 2.0))

        # legend, top right inside the axes, on a white background
        if len(series) > 1:
            entries = [(entry.get('label', 'series ' + str(index)),
                        entry.get('color') or PALETTE[index % len(PALETTE)])
                       for index, entry in enumerate(series)]
            legend_width = max(_text_width(draw, label, font_label) for label, _ in entries) + 54
            legend_height = 24 * len(entries) + 10
            legend_x = right - legend_width - 6
            legend_y = top + 8
            draw.rectangle([legend_x, legend_y, right - 6, legend_y + legend_height],
                           fill=(255, 255, 255), outline=(200, 200, 200))
            for index, (label, color) in enumerate(entries):
                y = legend_y + 7 + index * 24
                draw.line([(legend_x + 10, y + 8), (legend_x + 36, y + 8)], fill=color, width=3)
                _text(draw, legend_x + 42, y, label, font_label)

    folder = os.path.dirname(png_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    image.save(png_path)
    return png_path


def _text_width(draw, text, font):
    try:
        left, _, right, _ = draw.textbbox((0, 0), text, font=font)
        return right - left
    except Exception:
        return draw.textsize(text, font=font)[0]


# --------------------------------------------------------------------------------------
# bar chart
# --------------------------------------------------------------------------------------
def bar_chart(categories, groups, png_path, title=None, y_label='', figure_size=(1100, 500),
              value_format='%.3f'):
    '''Grouped bars; one group of bars per entry of `groups` ({label, values, color})'''
    width, height = figure_size
    image = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font_title = _font(22, bold=True)
    font_label = _font(15)
    font_tick = _font(13)

    if title:
        _text(draw, width / 2.0, 14, str(title), font_title, anchor='ma')

    left, right = 90, width - 30
    top, bottom = 90, height - 70
    values = [float(value) for group in groups for value in group['values'] if value == value]
    high = max(values) if values else 1.0
    high = high * 1.18 if high > 0 else 1.0
    step = _nice_step(high)
    high = math.ceil(round(high / step, 9)) * step
    n_tick = max(1, int(round(high / step)))

    for index in range(n_tick + 1):
        y = bottom - (bottom - top) * index / float(n_tick)
        draw.line([(left, y), (right, y)], fill=(226, 226, 226), width=1)
        _text(draw, left - 8, y, _tick_format(step) % (step * index), font_tick,
              fill=(70, 70, 70), anchor='ra')
    draw.rectangle([left, top, right, bottom], outline=(90, 90, 90), width=2)
    _vertical_text(image, y_label, font_label, (left - 44, (top + bottom) / 2.0))

    n_category = max(1, len(categories))
    n_group = max(1, len(groups))
    slot = (right - left) / float(n_category)
    bar_width = slot * 0.72 / n_group

    for category_index, category in enumerate(categories):
        base_x = left + slot * (category_index + 0.5)
        for group_index, group in enumerate(groups):
            value = float(group['values'][category_index])
            color = group.get('color') or PALETTE[group_index % len(PALETTE)]
            x0 = base_x - (bar_width * n_group) / 2.0 + group_index * bar_width
            y0 = bottom - (bottom - top) * value / high
            draw.rectangle([x0 + 2, y0, x0 + bar_width - 2, bottom], fill=color)
            _text(draw, x0 + bar_width / 2.0, y0 - 6, value_format % value, _font(12),
                  anchor='mb')
        _text(draw, base_x, bottom + 12, str(category), _font(13), anchor='ma')

    if len(groups) > 1:
        legend_x = left + 8
        for group_index, group in enumerate(groups):
            color = group.get('color') or PALETTE[group_index % len(PALETTE)]
            x = legend_x + group_index * 190
            draw.rectangle([x, top + 12, x + 16, top + 26], fill=color)
            _text(draw, x + 24, top + 12, str(group.get('label', '')), font_tick)

    folder = os.path.dirname(png_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    image.save(png_path)
    return png_path


# --------------------------------------------------------------------------------------
# architecture diagram
# --------------------------------------------------------------------------------------
def _block(draw, box, lines, fonts, fill=(232, 240, 250), outline=(70, 100, 140)):
    draw.rounded_rectangle(box, radius=10, fill=fill, outline=outline, width=2)
    x0, y0, x1, y1 = box
    line_height = 24
    total = line_height * len(lines)
    y = (y0 + y1) / 2.0 - total / 2.0
    for index, line in enumerate(lines):
        _text(draw, (x0 + x1) / 2.0, y + index * line_height, line,
              fonts[0] if index == 0 else fonts[1], anchor='ma')


def _arrow(draw, x, y0, y1):
    draw.line([(x, y0), (x, y1)], fill=(90, 90, 90), width=2)
    draw.polygon([(x - 6, y1 - 8), (x + 6, y1 - 8), (x, y1)], fill=(90, 90, 90))


def architecture_diagram(png_path, figure_size=(1320, 720)):
    '''
    Draw the two Stage 4 models: the sentiment classifier (left) and the joke generator
    (right).  Both share the embedding + recurrent encoder, which is the point of the
    ablation study.
    '''
    width, height = figure_size
    image = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font_title = _font(20, bold=True)
    font_block = _font(15, bold=True)
    font_small = _font(13)

    columns = [(40, 620, 'Text classification'), (690, 1270, 'Text generation')]
    blocks = [
        ['Input: token ids (T = 300)', 'vocabulary 30,000', 'Embedding (V x 128)'],
        ['Recurrent encoder', 'RNN / GRU / LSTM, H = 128'],
        ['Pooling', 'last state / mean / max', 'Dropout 0.3'],
        ['Linear (128 x 2)', 'softmax', 'negative / positive'],
    ]
    blocks_generation = [
        ['Input: token ids (T <= 64)', '<bos> w1 w2 ... <eos>', 'Embedding (V x 128)'],
        ['Recurrent encoder', 'RNN / GRU / LSTM, H = 256'],
        ['Linear (256 x V)', 'one score per word', 'softmax over the vocabulary'],
        ['Sampling', 'temperature + top-k', 'next word -> append'],
    ]

    for (x0, x1, title), panel_blocks in zip(columns, [blocks, blocks_generation]):
        _text(draw, (x0 + x1) / 2.0, 16, title, font_title, anchor='ma')
        box_height = 92
        gap = 46
        top = 70
        center = (x0 + x1) / 2.0
        for index, lines in enumerate(panel_blocks):
            y0 = top + index * (box_height + gap)
            box = (x0 + 60, y0, x1 - 60, y0 + box_height)
            fill = (232, 240, 250) if index < 3 else (250, 238, 228)
            _block(draw, box, lines, (font_block, font_small), fill=fill)
            if index < len(panel_blocks) - 1:
                _arrow(draw, center, y0 + box_height, y0 + box_height + gap)

    _text(draw, width / 2.0, height - 26,
          'Both models use the same encoder; only the head and the loss differ.',
          _font(14), fill=(80, 80, 80), anchor='ma')

    folder = os.path.dirname(png_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    image.save(png_path)
    return png_path
