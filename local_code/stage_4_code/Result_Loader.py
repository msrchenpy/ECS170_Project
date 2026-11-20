'''
Concrete ResultModule class that loads a Stage 4 ResultModule output back into memory
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import os
import pickle

from local_code.base_class.result import result


class Result_Loader(result):
    data = None
    fold_count = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def _destination_path(self):
        folder = self.result_destination_folder_path or '.'
        file_name = str(self.result_destination_file_name)
        if self.fold_count is not None:
            file_name += '_' + str(self.fold_count)
        return os.path.join(folder, file_name)

    def load(self):
        print('loading results...')
        path = self._destination_path()
        if not os.path.exists(path):
            raise FileNotFoundError('result file not found: ' + path)
        with open(path, 'rb') as file:
            self.data = pickle.load(file)
        print('  result loaded from ' + path)
        return self.data
