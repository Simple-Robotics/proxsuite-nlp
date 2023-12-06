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
    function: str
    solver_name: str
    problem_type: int
    problem_size: int
    cpu_time: float


TOOLS = "pan,hover,wheel_zoom,box_zoom,reset,save"


def main():
    parser = argparse.ArgumentParser(
        prog="plot_bench", description="Plot ProxNLP benchmarks"
    )
    parser.add_argument("bench_results", type=argparse.FileType("r"), nargs="+")
    args = parser.parse_args()

    values = []
    for bench_result_ft in args.bench_results:
        bench_result = json.load(bench_result_ft)
        source = Path(bench_result_ft.name).stem
        for e in bench_result["benchmarks"]:
            name_entries = e["name"].split("/")
            name = name_entries[0]
            size = int(name_entries[1])
            function = "compute" if "compute" in name else "solve"
            problem = int(name_entries[2])
            solver_name = SOLVER_NAME_RE.findall(name)[0]
            cpu_time = int(e["cpu_time"])
            values.append(
                ResultEntry(source, function, solver_name, problem, size, cpu_time)
            )

    frame = pd.DataFrame(values)
    compute_frame = frame[frame["function"] == "compute"]
    solve_frame = frame[frame["function"] == "solve"]

    sources = frame["source"].sort_values().unique()
    solver_names = frame["solver_name"].sort_values().unique()
    problem_sizes = frame["problem_size"].sort_values().unique()
    problem_types = frame["problem_type"].sort_values().unique()

    # Plot result by problem type
    # Each sources are in a different tabs
    tabs = []
    for source in sources:
        plot_lines = []
        for pb_type in problem_types:
            p1 = figure(
                title=f"Compute on problem {pb_type}", tools=TOOLS, y_axis_type="log"
            )
            for s, color in zip(solver_names, Colorblind8):
                v = compute_frame[
                    (compute_frame["solver_name"] == s)
                    & (compute_frame["problem_type"] == pb_type)
                    & (compute_frame["source"] == source)
                ].sort_values("problem_size")
                y = v["cpu_time"]
                p1.circle(problem_sizes, y, legend_label=s, color=color)
                p1.line(problem_sizes, y, legend_label=s, color=color, width=2)
            p1.legend.location = "top_left"
            p1.legend.click_policy = "hide"
            p1.xaxis.axis_label = "problem size"
            p1.yaxis.axis_label = "cpu time (us)"

            p2 = figure(
                title=f"Solve on problem {pb_type}", tools=TOOLS, y_axis_type="log"
            )
            for s, color in zip(solver_names, Colorblind8):
                v = solve_frame[
                    (solve_frame["solver_name"] == s)
                    & (solve_frame["problem_type"] == pb_type)
                    & (solve_frame["source"] == source)
                ].sort_values("problem_size")
                y = v["cpu_time"]
                p2.circle(problem_sizes, y, legend_label=s, color=color)
                p2.line(problem_sizes, y, legend_label=s, color=color, width=2)
            p2.legend.location = "top_left"
            p2.legend.click_policy = "hide"
            p2.xaxis.axis_label = "problem size"
            p2.yaxis.axis_label = "cpu time (us)"
            plot_lines.append((p1, p2))
        bench_plot = gridplot(plot_lines, sizing_mode="stretch_width", height=600)
        tabs.append(TabPanel(child=bench_plot, title=source))

    # Plot performance of the same solver on different sources
    # Each problem are on a different tabs
    for pb_type in problem_types:
        plot_lines = []
        for s in solver_names:
            p1 = figure(title=f"Compute on solver {s}", tools=TOOLS, y_axis_type="log")
            for source, color in zip(sources, Colorblind8):
                v = compute_frame[
                    (compute_frame["solver_name"] == s)
                    & (compute_frame["problem_type"] == pb_type)
                    & (compute_frame["source"] == source)
                ].sort_values("problem_size")
                y = v["cpu_time"]
                p1.circle(problem_sizes, y, legend_label=source, color=color)
                p1.line(problem_sizes, y, legend_label=source, color=color, width=2)
            p1.legend.location = "top_left"
            p1.legend.click_policy = "hide"
            p1.xaxis.axis_label = "problem size"
            p1.yaxis.axis_label = "cpu time (us)"

            p2 = figure(title=f"Solve on solver {s}", tools=TOOLS, y_axis_type="log")
            for source, color in zip(sources, Colorblind8):
                v = solve_frame[
                    (solve_frame["solver_name"] == s)
                    & (solve_frame["problem_type"] == pb_type)
                    & (solve_frame["source"] == source)
                ].sort_values("problem_size")
                y = v["cpu_time"]
                p2.circle(problem_sizes, y, legend_label=source, color=color)
                p2.line(problem_sizes, y, legend_label=source, color=color, width=2)
            p2.legend.location = "top_left"
            p2.legend.click_policy = "hide"
            p2.xaxis.axis_label = "problem size"
            p2.yaxis.axis_label = "cpu time (us)"
            plot_lines.append((p1, p2))
        bench_plot = gridplot(plot_lines, sizing_mode="stretch_width", height=600)
        tabs.append(TabPanel(child=bench_plot, title=f"Solver on problem {pb_type}"))

    output_name = "_".join(sources)
    output_file(f"block_sparse_{output_name}.html", title=output_name)

    show(Tabs(tabs=tabs, sizing_mode="stretch_width"))


if __name__ == "__main__":
    main()
