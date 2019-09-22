import matplotlib.pyplot as plt
import seaborn as sns
from common import load_file
from sklearn.metrics import roc_curve, roc_auc_score

sns.set_style("darkgrid")

#-------------------------------------------------------------------------------
X_test = load_file("./models/all_features/X_test_kaggle.pickle")
Ys = load_file("./models/all_features/Ys_kaggle.pickle")

fig, axes = plt.subplots(2,2, figsize=(8,8))

mbti_types = ["IE", "NS", "TF", "JP"]
line_styles = ["b-", 'k:', 'y-', 'r--']
model_names = ["LR", "RF", "NB", "SVM"]
for ax, mbti_type in zip(axes.flatten(), mbti_types):

    # loop through each model and plot on the same axis
    aucs = []
    for model, line_style in zip(model_names, line_styles):
        print(f"Plotting {model} on {mbti_type}")

        # clf = load_file(f"./models/{model}_clf_{mbti_type}_kaggle.pickle")
        clf = load_file(f"./models/all_features/{model}_clf_{mbti_type}_kaggle.pickle")

        y_true = Ys[mbti_type + "_test"]
        y_pred = clf.predict(X_test)
        y_probs = clf.predict_proba(X_test)[:, 1].flatten()

        # plot ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        auc = roc_auc_score(y_true, y_probs)
        aucs.append(auc)

        ax.plot(fpr, tpr, line_style)

    ax.plot([0,1], [0, 1], 'k--')  # diagonal line

    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])

    # legend with AUC scores
    ax.legend([f"{model} ({auc_val:.3f})" for model,auc_val in zip(model_names, aucs)], loc="lower right")

    ax.set_title(f"{mbti_type[0]}-{mbti_type[1]}")

plt.tight_layout()
plt.show(fig)
