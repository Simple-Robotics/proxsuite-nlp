#! /usr/bin/env python3
#

import re
import json
import typing
import argparse
from pathlib import Path

import pandas as pd

from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import Colorblind8
from bokeh.models import TabPanel, Tabs

SOLVER_NAME_RE = re.compile(r"^[^<>]*\<([^<>]*)\<.*\>\>$")


class ResultEntry(typing.NamedTuple):
    source: str
    problem_type: str
    solver_name: str
    problem_size: int
    rank_reduction: int
    cpu_time: float


TOOLS = "pan,hover,wheel_zoom,box_zoom,reset,save"


def main():
    parser = argparse.ArgumentParser(
        prog="plot_bench", description="Plot ProxNLP benchmarks"
    )
    parser.add_argument(
        "bench_results", type=argparse.FileType("r"), nargs="+")
    args = parser.parse_args()

    values = []
    for bench_result_ft in args.bench_results:
        bench_result = json.load(bench_result_ft)
        source = Path(bench_result_ft.name).stem
        for e in bench_result["benchmarks"]:
            name_entries = e["name"].split("/")
            name = name_entries[0]
            solver_name = SOLVER_NAME_RE.findall(name)[0]
            size = int(name_entries[1].replace("dim:", ""))
            if "pos_def" in name:
                problem_type = "pos_def"
                rank_reduction = 0
            elif "pos_sem_def" in name:
                problem_type = "pos_sem_def"
                rank_reduction = int(name_entries[2].replace("non_def:", ""))
            elif "indefinite" in name:
                problem_type = "indefinite"
                rank_reduction = 0
            else:
                raise RuntimeError(f"{name} is a unknown benchmark type")

            cpu_time = int(e["cpu_time"])
            values.append(
                ResultEntry(source, problem_type, solver_name,
                            size, rank_reduction, cpu_time)
            )

    frame = pd.DataFrame(values)

    sources = frame["source"].sort_values().unique()
    problem_sizes = frame["problem_size"].sort_values().unique()
    problem_types = frame["problem_type"].sort_values().unique()

    # Plot result by problem type and matrix rank
    # Each sources are in a different tabs
    tabs = []
    for source in sources:
        plot_lines = []
        for pb_type in problem_types:
            ptf = frame[frame["problem_type"] == pb_type]
            solver_names = ptf["solver_name"].sort_values().unique()
            rank_reductions = ptf["rank_reduction"].sort_values().unique()
            for rank_red in rank_reductions:
                p1 = figure(
                    title=f"Compute on problem {
                        pb_type} rank reduction {rank_red}",
                    tools=TOOLS, y_axis_type="log"
                )
                for s, color in zip(solver_names, Colorblind8):
                    v = ptf[(ptf["solver_name"] == s)
                            & (ptf["source"] == source)
                            & (ptf["rank_reduction"] == rank_red)
                            ].sort_values("problem_size")
                    y = v["cpu_time"]
                    p1.circle(problem_sizes, y, legend_label=s, color=color)
                    p1.line(problem_sizes, y, legend_label=s,
                            color=color, width=2)
                p1.legend.location = "top_left"
                p1.legend.click_policy = "hide"
                p1.xaxis.axis_label = "problem size"
                p1.yaxis.axis_label = "cpu time (us)"
                plot_lines.append((p1, ))
        bench_plot = gridplot(
            plot_lines, sizing_mode="stretch_width", height=600)
        tabs.append(TabPanel(child=bench_plot, title=source))

    # Plot performance of the same solver on different sources
    # Each problem are on a different tabs
    for pb_type in problem_types:
        plot_lines = []
        ptf = frame[frame["problem_type"] == pb_type]
        solver_names = ptf["solver_name"].sort_values().unique()
        rank_reductions = ptf["rank_reduction"].sort_values().unique()
        for rank_red in rank_reductions:
            for s in solver_names:
                p1 = figure(title=f"Compute on solver {
                    s} rank reduction {rank_red}", tools=TOOLS, y_axis_type="log")
                for source, color in zip(sources, Colorblind8):
                    v = ptf[(ptf["solver_name"] == s)
                            & (ptf["source"] == source)
                            & (ptf["rank_reduction"] == rank_red)
                            ].sort_values("problem_size")
                    y = v["cpu_time"]
                    p1.circle(problem_sizes, y,
                              legend_label=source, color=color)
                    p1.line(problem_sizes, y, legend_label=source,
                            color=color, width=2)
                p1.legend.location = "top_left"
                p1.legend.click_policy = "hide"
                p1.xaxis.axis_label = "problem size"
                p1.yaxis.axis_label = "cpu time (us)"
                plot_lines.append((p1, ))

        bench_plot = gridplot(
            plot_lines, sizing_mode="stretch_width", height=600)
        tabs.append(TabPanel(child=bench_plot,
                    title=f"Solver on problem {pb_type}"))

    output_name = "_".join(sources)
    output_file(f"dense_{output_name}.html", title=output_name)

    show(Tabs(tabs=tabs, sizing_mode="stretch_width"))


if __name__ == "__main__":
    main()
