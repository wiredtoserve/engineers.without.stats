import pandas as pd
from common import load_file, dummy_fn
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_color_codes("pastel")

fig, axes = plt.subplots(2,2, figsize=(12,12), sharex=True)

mbti_types = ["IE", "NS", "TF", "JP"]
colours = ["b", "r", "grey", "g"]

cv = load_file("./models/cv.pickle")

for mbti_type, colour, ax in zip(mbti_types, colours, axes.flatten()):
    rf_clf = load_file(f"./models/RF_clf_{mbti_type}_kaggle.pickle")

    feature_names = cv.get_feature_names()

    # get feature importances
    F = rf_clf.feature_importances_
    sorted_feats = sorted([(val, idx, feature_names[idx]) for idx,val in enumerate(F)], reverse=True)


    df = pd.DataFrame(sorted_feats, columns=["Importance", "Idx", "Token"])

    plot_df = df[:20]

    # b = sns.barplot(x="Importance", y="Token", data=plot_df, palette=sns.cubehelix_palette(25))
    b = sns.barplot(x="Importance", y="Token", data=plot_df, color=colour, ax=ax)

    b.axes.set_title(f"{mbti_type[0]}-{mbti_type[1]}", fontsize=20)
    b.axes.tick_params(labelsize=15)
    b.axes.set_xlabel("Feature importance", fontsize=15)
    b.axes.set_ylabel("", fontsize=15)

# manually tweak axes
axes[0, 0].set_xlabel("")
axes[0, 1].set_xlabel("")
fig.tight_layout()
plt.show(fig)
