import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def display_heatmap(matrix, title, xlabel, ylabel, plot_corr, save_corr,
                    league, cmap='bwr'):
    plt.figure(figsize=(60, 48))
    sns.heatmap(matrix, annot=True, cmap=cmap, fmt=".2f")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_corr:
        plt.savefig(f"backend/models/ml_imgs/{league}_model/{title.lower().replace(' ', '_')}.png")
    if plot_corr:
        plt.show()
    plt.close()


def correlation_matrix(dataframe, threshold_mag=None, lower=False, k=0,
                       plot_corr=False, xlabel=None, ylabel=None,
                       title_append=None, save_corr=False, league=None):
    title = "Feature Correlation Matrix"
    corr_matrix = dataframe.copy().corr()

    if threshold_mag:
        corr_matrix *= (abs(corr_matrix) >= threshold_mag)
        corr_matrix = corr_matrix.map(lambda x: 0 if abs(x) < 1e-10 else x)
        title += f" (Threshold Magnitude of {threshold_mag})"

    if lower:
        corr_matrix = corr_matrix.where(
            np.tril(np.ones(corr_matrix.shape), k=k).astype(bool)
            )
        title = "Lower Triangular " + title

    if title_append:
        title += title_append
    if plot_corr or save_corr:
        display_heatmap(
            corr_matrix, title, xlabel, ylabel, plot_corr, save_corr, league
            )

    return corr_matrix


def get_dropped_features(dataframe, threshold_mag=0.7):
    corr_matrix = correlation_matrix(
        dataframe=dataframe, threshold_mag=threshold_mag, lower=False,
        plot_corr=False
        )

    edge_conns = {}
    col_names_df = list(corr_matrix.columns)
    corr_matrix = corr_matrix.values.tolist()

    for i in range(len(corr_matrix)):
        num_edge = 0
        nodes_connected_to = set()
        for j in range(len(corr_matrix[i])):
            if j == i:
                continue
            val = corr_matrix[i][j]
            if abs(float(val)) >= 0.7:
                num_edge += 1
                nodes_connected_to.add(col_names_df[j])

        if num_edge != 0:
            edge_conns[col_names_df[i]] = [num_edge, nodes_connected_to]

    dropped_features = set()
    while edge_conns:
        edge_conns = dict(
            sorted(
                edge_conns.items(), key=lambda item: item[1][0], reverse=True
                )
            )
        safe_node = list(edge_conns.keys())[0]
        neighbors = edge_conns[safe_node][1] or []
        newly_dropped_nodes = list(neighbors)
        dropped_features.update(newly_dropped_nodes)
        del edge_conns[safe_node]

        for dropped_node in newly_dropped_nodes:
            new_num_conn_nodes = edge_conns[dropped_node][0] - 1
            if new_num_conn_nodes == 0:
                del edge_conns[dropped_node]
            else:
                edge_conns[dropped_node][0] = new_num_conn_nodes
                edge_conns[dropped_node][1].discard(safe_node)

    return dropped_features

