import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def explain_binary_results(y_pred, y_test, positive):
    """
    y_pred, y_test are np.array and positive is the string representing the positive/1 case
    calculate the metrics, such as precision, accuracy with explanation
    if y_pred actually is y_pred_proba, we also plot ROC and PR curves
    """
    if len(y_pred.shape) > 1:  # y_pred is y_pred_proba
        y_pred_proba = y_pred[:, 1]
        y_pred = np.where(y_pred_proba > 0.5, 1, 0)

        # ROC
        FPR, TPR, _ = metrics.roc_curve(y_test, y_pred_proba)
        AUC = metrics.roc_auc_score(y_test, y_pred_proba)

        # PR
        precisions, recalls, _ = metrics.precision_recall_curve(y_test, y_pred_proba, pos_label=1)
        PR = metrics.auc(recalls, precisions)

        # plot ROC and PR
        fig, axes = plt.subplots(2, figsize=(7, 10))
        axes[0].plot(FPR, TPR, 'b.-', markersize=1, alpha=0.5)
        axes[0].set_xlabel("false positive rate")
        axes[0].set_ylabel("true positive rate")
        axes[0].set_title(f"$\\bf ROC$ curve, AUC ROC {AUC:.2f}")
        axes[1].plot(recalls, precisions, 'b.-', markersize=1, alpha=0.5)
        axes[1].set_xlabel("recall")
        axes[1].set_ylabel("precision")
        axes[1].set_title(f"$\\bf PR$ curve, AUC PR {PR:.2f}")
        plt.show()

    # calculate metrics
    confusion = metrics.confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = confusion.ravel()
    false_positive_rate = fp / (fp + tn)
    true_positive_rate = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    df_conf = pd.DataFrame(confusion, columns=[['Predicted', 'Predicted'], [f'not {positive}', f'{positive}']],
                           index=[['Actual', 'Actual'], [f'not {positive}', f'{positive}']])

    # print the calculated metrics with explanation
    print(df_conf, end='\n\n')
    print(f"* False positive rate is {false_positive_rate * 100:.2f}%: "
        f"among all {fp + tn} cases of actually not {positive}, "
        f'there are {fp} cases wrongly predicted as {positive}', end='\n\n')

    print(f'* True positive rate (also called recall) is {true_positive_rate * 100:.2f}%: '
          f'among all {tp + fn} cases of actually {positive}, '
          f'there are {tp} cases correctly predicted as {positive}', end='\n\n')

    print(f'* Precision is {precision * 100:.2f}%: among all {tp + fp} cases predicted as {positive}, '
          f'there are {tp} cases correctly predicted as {positive}', end='\n\n')

    print(f'* Accuracy is {accuracy * 100:.2f}%: among all {tn + fp + fn + tp} cases, '
          f'there are {tn + tp} cases have been correctly predicted', end='\n\n')
    print(f'* F1 score is {f1:.2f}')


# y_test = np.array([0, 0, 0, 1, 1, 1, 1, 1])
# y_pred = np.array([0, 1, 1, 1, 0, 1, 0, 1])
# explain_binary_results(y_pred, y_test, 'fraud')