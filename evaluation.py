
import itertools, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
def plot_confusion_and_curves(y_true, y_score, class_names, save_prefix: str = None):
    y_pred = (y_score >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    plt.figure(figsize=(4,4)); plt.imshow(cm, interpolation='nearest'); plt.title('Confusion Matrix'); plt.colorbar()
    t = np.arange(len(class_names)); plt.xticks(t, class_names, rotation=45); plt.yticks(t, class_names)
    thresh = cm.max()/2.0
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,cm[i,j], ha='center', color='white' if cm[i,j]>thresh else 'black')
    plt.ylabel('True'); plt.xlabel('Pred'); plt.tight_layout()
    if save_prefix: plt.savefig(save_prefix + "_cm.png", dpi=140, bbox_inches='tight')
    plt.show()
    fpr, tpr, _ = roc_curve(y_true, y_score); auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(5,4)); plt.plot(fpr, tpr, label=f'ROC AUC={auc:.3f}'); plt.plot([0,1],[0,1],'--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend(); plt.tight_layout()
    if save_prefix: plt.savefig(save_prefix + "_roc.png", dpi=140, bbox_inches='tight')
    plt.show()
    precision, recall, _ = precision_recall_curve(y_true, y_score); ap = average_precision_score(y_true, y_score)
    plt.figure(figsize=(5,4)); plt.plot(recall, precision, label=f'AP={ap:.3f}'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR Curve'); plt.legend(); plt.tight_layout()
    if save_prefix: plt.savefig(save_prefix + "_pr.png", dpi=140, bbox_inches='tight')
    plt.show()
    print(classification_report(y_true, y_pred, target_names=class_names))
