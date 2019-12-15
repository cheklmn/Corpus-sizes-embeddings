import pandas as pd
import matplotlib.pyplot as plt

from plot_params import *
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('reporttest.pdf') as pp:
    df = pd.read_pickle('Results.pickle')
    df.to_csv('results.csv')

    for i, size in enumerate(df['Word count']):
        df['Word count'][i] = size // 1000000

    for dimensions in graph_dimensions:
        fixed_dims = stats.copy()
        try:
            fixed_dims.remove(dimensions[0])
            fixed_dims.remove(dimensions[1])
        except:
            print('value not in list')


        df.set_index(fixed_dims, inplace=True)
        indexes = set(df.index)
        x = dimensions[0]
        y = dimension_values_map[dimensions[1]]

        for index in indexes:
            fig, axes = plt.subplots(ncols=2, figsize=(14,6))
            # print(axes[0])
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
                ax.set(xlabel=dimensions[0])
            axes.flat[0].set(ylabel="Spearman coefficient")

            title = ''
            for i in range(len(fixed_dims)):
                # print(index[i])
                # print(fixed_dims[i])
                title += fixed_dims[i] + ": " + str(index[i]) + ';'
                if i % 2 != 0:
                    title += '\n'

            plot_line.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plot_line.set_title(title)
            plot_bar.get_legend().remove()
            plot_bar.set_title(title)
            plt.xticks(df_i[x])
            plt.tight_layout()
            pp.savefig(bbox_inches="tight")

        df.reset_index(inplace=True)
    # index_array = [
    #     (["Word count", "Dataset"], "Window", "Dimension"),
    #     (["Word count", "Dataset"], "Dimension", "Window")
    # ]
    #
    # for array in index_array:
    #     df_x = df[array[2]].unique()
    #     df.set_index(array[0], inplace=True)
    #     indexes = set(df.index)
    #     x = array[1]
    #     y = [
    #         "Low bin score",
    #         "Middle bin score",
    #         "High bin score",
    #         "Mixed bin score",
    #         "General score"
    #         ]
    #
    #     for index in indexes:
    #         fig, axes = plt.subplots(ncols=2, figsize=(14,6))
    #         df_i = df.loc[index]
    #         for tick in df_x:
    #             df_i_filt = df_i[df_i[array[2]] == tick]
    #             plot_bar = df_i_filt.plot(
    #                 x = x,
    #                 y = y,
    #                 kind = "bar",
    #                 ax=axes[0])
    #             plot_line = df_i_filt.plot(
    #                 x = x,
    #                 y = y,
    #                 kind = "line",
    #                 ax=axes[1])
    #
    #             for ax in axes.flat:
    #                  ax.set(xlabel=array[1])
    #             axes.flat[0].set(ylabel="Spearman coefficient")
    #
    #             title = "{}: {}; {}: {} \nDataset: {}".format(array[0][0], index[0], array[2], tick, index[1])
    #             plot_line.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #             plot_line.set_title(title)
    #             plot_bar.get_legend().remove()
    #             plot_bar.set_title(title)
    #             plt.xticks(df_i_filt[array[1]])
    #             plt.tight_layout()
    #             pp.savefig(bbox_inches="tight")
    #     df.reset_index(inplace=True)

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
