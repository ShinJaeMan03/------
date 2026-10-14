import tkinter as tk
import math
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

REQUIRED_COLUMNS = ("time_s", "force_N")
SHAPE_DIMENSIONS = {
    "사각형": (("폭 b (mm)", "b"), ("높이 h (mm)", "h")),
    "원형봉": (("직경 d (mm)", "d"),),
    "원형관": (("외경 D (mm)", "D"), ("두께 t (mm)", "t")),
    "사각형관": (
        ("외측 폭 B (mm)", "B"),
        ("외측 높이 H (mm)", "H"),
        ("두께 t (mm)", "t"),
    ),
    "H형강": (
        ("전체 높이 H (mm)", "H"),
        ("플랜지 폭 B (mm)", "B"),
        ("웨브 두께 tw (mm)", "tw"),
        ("플랜지 두께 tf (mm)", "tf"),
    ),
}


def calculate_area(shape, dimensions):
    if shape == "사각형":
        return dimensions["b"] * dimensions["h"]
    if shape == "원형봉":
        return math.pi * dimensions["d"] ** 2 / 4
    if shape == "원형관":
        outer_diameter = dimensions["D"]
        thickness = dimensions["t"]
        if thickness * 2 >= outer_diameter:
            raise ValueError("원형관은 두께 t가 외경 D의 절반보다 작아야 합니다.")
        inner_diameter = outer_diameter - 2 * thickness
        return math.pi * (outer_diameter**2 - inner_diameter**2) / 4
    if shape == "사각형관":
        outer_width = dimensions["B"]
        outer_height = dimensions["H"]
        thickness = dimensions["t"]
        if thickness * 2 >= min(outer_width, outer_height):
            raise ValueError("사각형관은 두께 t가 폭과 높이의 절반보다 작아야 합니다.")
        return outer_width * outer_height - (outer_width - 2 * thickness) * (
            outer_height - 2 * thickness
        )
    if shape == "H형강":
        height = dimensions["H"]
        flange_width = dimensions["B"]
        web_thickness = dimensions["tw"]
        flange_thickness = dimensions["tf"]
        if height <= 2 * flange_thickness:
            raise ValueError("H형강은 전체 높이 h가 플랜지 두께 2tf보다 커야 합니다.")
        return 2 * flange_width * flange_thickness + (
            height - 2 * flange_thickness
        ) * web_thickness
    raise ValueError(f"지원하지 않는 단면 종류: {shape}")


def update_dimension_fields(shape_var, dimensions_frame, dimension_entries):
    for child in dimensions_frame.winfo_children():
        child.destroy()
    dimension_entries.clear()

    for row, (label_text, key) in enumerate(SHAPE_DIMENSIONS[shape_var.get()]):
        ttk.Label(dimensions_frame, text=label_text).grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=3
        )
        entry = ttk.Entry(dimensions_frame, width=14)
        entry.grid(row=row, column=1, sticky="w", pady=3)
        dimension_entries[key] = entry


def clear_graph(axis, y_label="Force (N)"):
    axis.clear()
    axis.set_xlabel("Time (s)")
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.3)


def analyze_file(
    file_path,
    path_var,
    count_var,
    status_var,
    graph_title_var,
    axis,
    canvas,
    data_holder,
):
    try:
        data = pd.read_csv(file_path)
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
        if missing_columns:
            missing = ", ".join(missing_columns)
            required = ", ".join(REQUIRED_COLUMNS)
            raise ValueError(f"필수 열이 없습니다: {missing}. 필요한 열: {required}")

        numeric_time = pd.to_numeric(data["time_s"], errors="coerce")
        numeric_force = pd.to_numeric(data["force_N"], errors="coerce")
        valid_mask = (
            numeric_time.notna()
            & numeric_force.notna()
            & np.isfinite(numeric_time)
            & np.isfinite(numeric_force)
        )
        valid_data = data.loc[valid_mask].copy()
        valid_data["time_s"] = numeric_time.loc[valid_data.index]
        valid_data["force_N"] = numeric_force.loc[valid_data.index]

        path_var.set(str(file_path))
        excluded_count = len(data) - len(valid_data)
        count_var.set(
            f"유효 데이터 수: {len(valid_data)}개, 제외 데이터 수: {excluded_count}개"
        )
        status_var.set("")
        graph_title_var.set("시간-하중 그래프")
        data_holder["data"] = valid_data
        clear_graph(axis)

        if valid_data.empty:
            status_var.set("유효한 데이터가 없어 그래프를 표시할 수 없습니다.")
        else:
            maximum_index = valid_data["force_N"].idxmax()
            maximum_time = valid_data.loc[maximum_index, "time_s"]
            maximum_force = valid_data.loc[maximum_index, "force_N"]
            axis.plot(
                valid_data["time_s"],
                valid_data["force_N"],
                marker="o",
                color="#1565c0",
            )
            axis.scatter([maximum_time], [maximum_force], color="red", zorder=3)
            axis.annotate(
                f"{maximum_time:g} s, {maximum_force:g} N",
                (maximum_time, maximum_force),
                textcoords="offset points",
                xytext=(8, 8),
            )
            status_var.set(f"최대 하중: {maximum_time:g} s, {maximum_force:g} N")
            axis.margins(x=0.05, y=0.1)

        canvas.draw()
    except (OSError, ValueError, pd.errors.ParserError) as error:
        count_var.set("유효 데이터 수: -, 제외 데이터 수: -")
        status_var.set(f"파일을 읽을 수 없습니다: {error}")
        graph_title_var.set("시간-하중 그래프")
        data_holder["data"] = None
        clear_graph(axis)
        canvas.draw()


def choose_file(
    path_var,
    count_var,
    status_var,
    graph_title_var,
    axis,
    canvas,
    data_holder,
):
    selected_path = filedialog.askopenfilename(
        title="CSV 파일 선택",
        filetypes=(("CSV 파일", "*.csv"), ("모든 파일", "*.*")),
    )
    if selected_path:
        analyze_file(
            Path(selected_path),
            path_var,
            count_var,
            status_var,
            graph_title_var,
            axis,
            canvas,
            data_holder,
        )


def analyze_stress(
    data_holder,
    shape_var,
    dimension_entries,
    area_var,
    status_var,
    graph_title_var,
    axis,
    canvas,
):
    data = data_holder["data"]
    if data is None or data.empty:
        status_var.set("먼저 유효한 CSV 파일을 선택하세요.")
        return

    try:
        dimensions = {
            key: float(entry.get()) for key, entry in dimension_entries.items()
        }
    except (TypeError, ValueError):
        area_var.set("단면적 A: -")
        status_var.set("모든 치수에 숫자를 입력하세요. 빈칸은 사용할 수 없습니다.")
        return

    if any(not math.isfinite(value) or value <= 0 for value in dimensions.values()):
        area_var.set("단면적 A: -")
        status_var.set("모든 치수는 0보다 큰 유한한 숫자여야 합니다.")
        return

    try:
        area = calculate_area(shape_var.get(), dimensions)
    except ValueError as error:
        area_var.set("단면적 A: -")
        status_var.set(str(error))
        return

    stress_data = data.copy()
    stress_data["stress_MPa"] = stress_data["force_N"] / area
    maximum_index = stress_data["stress_MPa"].idxmax()
    maximum_time = stress_data.loc[maximum_index, "time_s"]
    maximum_stress = stress_data.loc[maximum_index, "stress_MPa"]

    area_var.set(f"단면적 A: {area:.2f} mm²")
    status_var.set(f"최대 응력: {maximum_time:g} s, {maximum_stress:g} MPa")
    graph_title_var.set("시간-응력 그래프")
    clear_graph(axis, "Stress (MPa)")
    axis.plot(
        stress_data["time_s"],
        stress_data["stress_MPa"],
        marker="o",
        color="#1565c0",
    )
    axis.scatter([maximum_time], [maximum_stress], color="red", zorder=3)
    axis.annotate(
        f"{maximum_time:g} s, {maximum_stress:g} MPa",
        (maximum_time, maximum_stress),
        textcoords="offset points",
        xytext=(8, 8),
    )
    axis.margins(x=0.05, y=0.1)
    canvas.draw()


def main():
    window = tk.Tk()
    window.title("하중 데이터 분석기")
    window.geometry("900x780")
    window.minsize(700, 620)

    frame = ttk.Frame(window, padding=16)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)
    frame.rowconfigure(9, weight=1)

    ttk.Label(frame, text="하중 데이터 분석기", font=("맑은 고딕", 16, "bold")).grid(
        row=0, column=0, columnspan=3, sticky="w", pady=(0, 14)
    )

    path_var = tk.StringVar()
    count_var = tk.StringVar(value="유효 데이터 수: 0개, 제외 데이터 수: 0개")
    status_var = tk.StringVar(value="CSV 파일을 선택하세요.")
    graph_title_var = tk.StringVar(value="시간-하중 그래프")
    shape_var = tk.StringVar(value="사각형")
    area_var = tk.StringVar(value="단면적 A: -")
    data_holder = {"data": None}
    dimension_entries = {}

    ttk.Button(frame, text="CSV 찾아보기", command=lambda: choose_file(
        path_var,
        count_var,
        status_var,
        graph_title_var,
        axis,
        canvas,
        data_holder,
    )).grid(row=1, column=0, padx=(0, 8), sticky="w")
    ttk.Entry(frame, textvariable=path_var, state="readonly").grid(
        row=1, column=1, columnspan=2, sticky="ew"
    )

    ttk.Label(frame, textvariable=count_var).grid(
        row=2, column=0, columnspan=3, sticky="nw", pady=(12, 8)
    )
    ttk.Label(
        frame,
        text="필수 열: time_s, force_N · 빈칸, 문자, 무한대가 있는 행은 제외합니다.",
    ).grid(row=3, column=0, columnspan=3, sticky="w")
    ttk.Label(frame, textvariable=status_var).grid(
        row=4, column=0, columnspan=3, sticky="w", pady=(6, 8)
    )

    ttk.Label(frame, text="단면 종류:").grid(row=5, column=0, sticky="w", pady=4)
    shape_combo = ttk.Combobox(
        frame,
        textvariable=shape_var,
        values=tuple(SHAPE_DIMENSIONS),
        state="readonly",
        width=12,
    )
    shape_combo.grid(row=5, column=1, sticky="w", pady=4)

    dimensions_frame = ttk.Frame(frame)
    dimensions_frame.grid(row=6, column=0, columnspan=4, sticky="w", pady=(2, 4))
    shape_combo.bind(
        "<<ComboboxSelected>>",
        lambda event: update_dimension_fields(
            shape_var, dimensions_frame, dimension_entries
        ),
    )
    update_dimension_fields(shape_var, dimensions_frame, dimension_entries)

    ttk.Label(frame, textvariable=area_var).grid(row=7, column=0, columnspan=2, sticky="w", pady=4)
    ttk.Button(
        frame,
        text="분석",
        command=lambda: analyze_stress(
            data_holder,
            shape_var,
            dimension_entries,
            area_var,
            status_var,
            graph_title_var,
            axis,
            canvas,
        ),
    ).grid(row=7, column=2, sticky="w", padx=(18, 0), pady=4)

    ttk.Label(frame, textvariable=graph_title_var).grid(
        row=8, column=0, columnspan=4, sticky="w", pady=(10, 6)
    )

    figure = Figure(figsize=(8.6, 4.7), dpi=100)
    axis = figure.add_subplot(111)
    clear_graph(axis)
    canvas = FigureCanvasTkAgg(figure, master=frame)
    canvas.get_tk_widget().grid(row=9, column=0, columnspan=4, sticky="nsew")

    window.mainloop()


if __name__ == "__main__":
    main()
