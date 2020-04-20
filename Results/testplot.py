import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages("reportanalogy.pdf") as pp:
    df = pd.read_pickle("Results_analogy.pickle")
    df.to_csv("results_analogy.csv")

    for i, size in enumerate(df["Word count"]):
        df["Word count"][i] = size // 1000000

    df.set_index(["Dataset","Dimension","Sampling"], inplace=True)
    indexes = set(df.index)

    x = "Word count"
    y = "Score"
    for index in indexes:
        fig, axes = plt.subplots(ncols=2, figsize=(14,6))
        df_i = df.loc[index]
        ax_bar = sns.barplot(x='Word count', y='Score', hue='Window', data=df_i, ax=axes[0])
        ax_line = sns.lineplot(x='Word count', y='Score', hue='Window', data=df_i, ax=axes[1])
        # plot_bar = df_i.plot(
        #     x = x,
        #     y = y,
        #     kind = "bar",
        #     ax=axes[0])
        # plot_line = df_i.plot(
        #     x = x,
        #     y = y,
        #     kind = "line",
        #     ax=axes[1])

        for ax in axes.flat:
            ax.set(xlabel="Size, mln. words")
        axes.flat[0].set(ylabel="Evaluation score")

        title = "Dataset: {}\nDimension: {}\nSampling: {}".format(index[0], index[1], index[2])
        # plot_line.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # plot_line.set_title(title)
        # plot_bar.get_legend().remove()
        # plot_bar.set_title(title)
        ax_bar.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax_bar.set_title(title)
        ax_line.get_legend().remove()
        ax_line.set_title(title)
        plt.xticks(df_i["Word count"])
        plt.tight_layout()
        pp.savefig(bbox_inches="tight")

    # df_counts = df.groupby(["Dataset"]).mean()
    # plot_counts = df_counts.reset_index().plot(
    #     x = "Dataset",
    #     y = ["Count of Low"
    #        , "Count of Middle"
    #        , "Count of High"
    #        , "Count of Mixed"
    #     ],
    #     kind = "barh"
    # )
    # plt.ylabel("")
    # plt.title("Counts of words pairs for each dataset")
    # pp.savefig(bbox_inches="tight")