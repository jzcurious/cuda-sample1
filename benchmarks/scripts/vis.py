import pandas as pd
import plotly.graph_objects as go
import argparse


def make_chart(
    *dfs,
    data_range=[0, -1],
    xlabel="",
    ylabel="",
    title="",
    legend={},
    colors={},
    xlog=False,
    ylog=False,
    use_markers=False,
):
    if not xlabel:
        xlabel = dfs[0].columns[0]

    if not ylabel:
        ylabel = dfs[0].columns[1]

    if isinstance(data_range, tuple):
        data_range = list(data_range)

    if data_range[0] < 0 or data_range[0] >= len(dfs[0]):
        data_range[0] = 0

    if data_range[1] <= 0 or data_range[1] >= len(dfs[0]):
        data_range[1] = None

    fig = go.Figure(
        layout=go.Layout(
            xaxis=dict(
                autorange=True,
                title_text=xlabel,
            ),
            yaxis=dict(
                autorange=True,
                title_text=ylabel,
            ),
            title_text=title,
            hovermode="closest",
            title_x=0.5,
            template="plotly_dark",
        ),
    )

    for i, df in enumerate(dfs):
        fig.add_trace(
            go.Scatter(
                x=df[xlabel][data_range[0] : data_range[1]],
                y=df[ylabel][data_range[0] : data_range[1]],
                mode="lines+markers" if use_markers else "lines",
                line=dict(dash="solid"),
                name=f"{i}",
            ),
        )

        if i in colors:
            fig.update_traces(
                line=dict(color=colors[i]),
                selector=({"name": f"{i}"}),
            )

        if i in legend:
            fig.update_traces(
                name=legend.get(i), showlegend=True, selector=({"name": f"{i}"})
            )
        else:
            fig.update_traces(showlegend=False)

    if xlog:
        fig.update_xaxes(type="log")

    if ylog:
        fig.update_yaxes(type="log")

    fig.show()


def read_data(fnames):
    return [pd.read_csv(fname, skipinitialspace=True) for fname in fnames]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        nargs="+",
        help="specify the input data",
    )

    parser.add_argument(
        "-x", "--xlabel", default="", type=str, help="specify the X-axis label"
    )
    parser.add_argument(
        "-y", "--ylabel", default="", type=str, help="specify the Y-axis label"
    )
    parser.add_argument(
        "-t", "--title", default="", type=str, help="specify the chart title"
    )
    parser.add_argument(
        "-l", "--legend", type=str, nargs="+", help="specify the legend"
    )
    parser.add_argument("-c", "--colors", type=str, nargs="+", help="specify colors")
    parser.add_argument(
        "-r",
        "--range",
        type=int,
        nargs=2,
        help="specify the data range",
        default=(0, 0),
    )
    parser.add_argument("--xlog", action="store_true")
    parser.add_argument("--ylog", action="store_true")
    parser.add_argument("-m", "--usemarkers", action="store_true")

    args = parser.parse_args()
    data = read_data(args.input)
    legend = {i: name for i, name in enumerate(args.legend)} if args.legend else {}
    colors = {i: color for i, color in enumerate(args.colors)} if args.colors else {}

    make_chart(
        *data,
        data_range=args.range,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        title=args.title,
        legend=legend,
        colors=colors,
        xlog=args.xlog,
        ylog=args.ylog,
        use_markers=args.usemarkers,
    )
