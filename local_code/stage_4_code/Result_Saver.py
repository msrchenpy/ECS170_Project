'''
Concrete ResultModule class for a Stage 4 ResultModule output
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import os
import pickle

from local_code.base_class.result import result


class Result_Saver(result):
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

    def save(self):
        print('saving results...')
        path = self._destination_path()
        folder = os.path.dirname(path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        with open(path, 'wb') as file:
            pickle.dump(self.data, file)
        print('  result saved to ' + path)
        return path
