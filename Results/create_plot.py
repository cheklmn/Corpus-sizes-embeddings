import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

with PdfPages("report10.pdf") as pp:
    df = pd.read_pickle("Results.pickle")
    df.to_csv("results.csv")

    for i, size in enumerate(df["Word count"]):
        df["Word count"][i] = size // 1000000

    df.set_index(["Method","Dataset"], inplace=True)
    indexes = set(df.index)

    x = "Word count"
    y = [
        "Low bin score",
        "Middle bin score",
        "High bin score",
        "Mixed bin score",
        "General score"
        ]

    for index in indexes:
        fig, axes = plt.subplots(ncols=2, figsize=(14,6))
        df_i = df.loc[index]
        plot_bar = df_i.plot(
            x = x,
            y = y,
            kind = "bar",
            ax=axes[0])
        plot_line = df_i.plot(
            x = x,
            y = y,
            kind = "line",
            ax=axes[1])

        for ax in axes.flat:
            ax.set(xlabel="Word count, mln. words")
        axes.flat[0].set(ylabel="Spearman coefficient")

        title = "Method: {}\nDataset: {}".format(index[0], index[1])
        plot_line.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plot_line.set_title(title)
        plot_bar.get_legend().remove()
        plot_bar.set_title(title)
        plt.xticks(df_i["Word count"])
        plt.tight_layout()
        pp.savefig(bbox_inches="tight")

    print(df.head())
    df_counts = df.groupby("Dataset").mean()
    plot_counts = df_counts.reset_index().plot(
        x = "Dataset",
        y = [
            "Low bin pair count",
            "Middle bin pair count",
            "High bin pair count",
            "Mixed bin pair count"
            ],
        kind = "barh"
    )
    plt.ylabel("")
    plt.title("Counts of words pairs for each dataset")
    pp.savefig(bbox_inches="tight")
