'''
Run every Stage 4 experiment in order and then redraw the figures.

    python -m script.stage_4_script.run_all

The three steps are independent, so a single failing step does not stop the next one; the exit
code is zero only when all of them succeeded.
'''

import os
import subprocess
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STEPS = [
    ['-m', 'script.stage_4_script.script_classification',
     '--models', 'rnn,gru,lstm', '--epochs', '10', '--batch-size', '128'],
    ['-m', 'script.stage_4_script.script_classification',
     '--models', 'gru_bidirectional,lstm_2layer,gru_meanpool', '--epochs', '10',
     '--batch-size', '128', '--output-folder', 'result/stage_4_result/ablation'],
    ['-m', 'script.stage_4_script.script_generation'],
    ['-m', 'script.stage_4_script.make_figures'],
]


def main():
    environment = dict(os.environ)
    environment.setdefault('PYTHONPATH', _PROJECT_ROOT)
    # the MIOpen cache must live in the workspace, otherwise the first recurrent kernel fails
    cache = os.path.join(_PROJECT_ROOT, 'result', 'stage_4_result', 'miopen')
    environment.setdefault('MIOPEN_CUSTOM_CACHE_DIR', cache)
    environment.setdefault('MIOPEN_USER_DB_PATH', cache)

    failures = 0
    for step in STEPS:
        print('=' * 78)
        print('running: ' + ' '.join([sys.executable] + step))
        print('=' * 78)
        start = time.time()
        code = subprocess.call([sys.executable] + step, cwd=_PROJECT_ROOT, env=environment)
        print('finished in ' + str(round(time.time() - start, 1)) + ' s, exit code ' + str(code))
        if code != 0:
            failures += 1

    print('')
    if failures:
        print(str(failures) + ' step(s) failed')
    else:
        print('all Stage 4 experiments finished; results in result/stage_4_result/')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
