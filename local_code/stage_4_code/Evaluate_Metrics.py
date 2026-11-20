'''
Concrete Evaluate class for the binary sentiment classification of Stage 4-3.

The task is binary (negative / positive), so Precision, Recall and F1 are reported for the
positive class (the class the classifier is usually interested in) together with the macro
and weighted averages over the two classes.  Accuracy alone can be misleading on a data set
with a different number of positive and negative reviews, which is why the three other
metrics asked for by the report template are computed as well.
'''

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from local_code.base_class.evaluate import evaluate


class Evaluate_Metrics(evaluate):
    data = None
    label_names = ['negative', 'positive']

    def evaluate(self):
        print('evaluating performance...')
        true_y = np.asarray(self.data['true_y']).astype(np.int64).ravel()
        pred_y = np.asarray(self.data['pred_y']).astype(np.int64).ravel()
        labels = list(range(len(self.label_names)))

        metrics = {
            'accuracy': float(accuracy_score(true_y, pred_y)),
            'precision': float(precision_score(true_y, pred_y, labels=labels, pos_label=1,
                                               average='binary', zero_division=0)),
            'recall': float(recall_score(true_y, pred_y, labels=labels, pos_label=1,
                                         average='binary', zero_division=0)),
            'f1': float(f1_score(true_y, pred_y, labels=labels, pos_label=1,
                                 average='binary', zero_division=0)),
            'precision_macro': float(precision_score(true_y, pred_y, labels=labels,
                                                     average='macro', zero_division=0)),
            'recall_macro': float(recall_score(true_y, pred_y, labels=labels,
                                               average='macro', zero_division=0)),
            'f1_macro': float(f1_score(true_y, pred_y, labels=labels, average='macro',
                                       zero_division=0)),
            'precision_weighted': float(precision_score(true_y, pred_y, labels=labels,
                                                        average='weighted', zero_division=0)),
            'recall_weighted': float(recall_score(true_y, pred_y, labels=labels,
                                                  average='weighted', zero_division=0)),
            'f1_weighted': float(f1_score(true_y, pred_y, labels=labels, average='weighted',
                                          zero_division=0)),
            'confusion_matrix': confusion_matrix(true_y, pred_y, labels=labels),
        }
        return metrics

    def print_metrics(self, metrics, title='evaluation result'):
        print('************ ' + str(title) + ' ************')
        print('accuracy            : ' + str(round(metrics['accuracy'], 4)))
        print('precision (positive): ' + str(round(metrics['precision'], 4)))
        print('recall    (positive): ' + str(round(metrics['recall'], 4)))
        print('f1        (positive): ' + str(round(metrics['f1'], 4)))
        print('precision (macro)   : ' + str(round(metrics['precision_macro'], 4)))
        print('recall    (macro)   : ' + str(round(metrics['recall_macro'], 4)))
        print('f1        (macro)   : ' + str(round(metrics['f1_macro'], 4)))

    def print_report(self, title='per class report'):
        print('************ ' + str(title) + ' ************')
        true_y = np.asarray(self.data['true_y']).astype(np.int64).ravel()
        pred_y = np.asarray(self.data['pred_y']).astype(np.int64).ravel()
        print(classification_report(true_y, pred_y, labels=[0, 1],
                                    target_names=self.label_names, zero_division=0))
