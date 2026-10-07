import tkinter as tk
from tkinter import ttk

NEWTONS_PER_KILONEWTON = 1000
NEWTONS_PER_KGF = 9.81


def convert_force(force_kn):
    force_n = force_kn * NEWTONS_PER_KILONEWTON
    force_kgf = force_n / NEWTONS_PER_KGF
    return force_n, force_kgf


def convert_from_unit(value, unit):
    if unit == "kN":
        return value
    if unit == "N":
        return value / NEWTONS_PER_KILONEWTON
    if unit == "kgf":
        return value * NEWTONS_PER_KGF / NEWTONS_PER_KILONEWTON
    raise ValueError(f"지원하지 않는 단위: {unit}")


def calculate(value_entry, unit_combo, result_label):
    try:
        value = float(value_entry.get())
        if value < 0:
            raise ValueError
        force_kn = convert_from_unit(value, unit_combo.get())
        force_n, force_kgf = convert_force(force_kn)
        result_label.config(
            text=(
                f"변환 결과\n"
                f"{force_kn:.2f} kN = {force_n:.2f} N\n"
                f"{force_kgf:.2f} kgf"
            ),
            foreground="#1b5e20",
        )
    except ValueError:
        result_label.config(
            text="잘못된 입력입니다. 0 이상의 숫자를 입력하세요.",
            foreground="#b71c1c",
        )


def reset(value_entry, unit_combo, result_label):
    value_entry.delete(0, tk.END)
    unit_combo.set("kN")
    result_label.config(text="결과가 여기에 표시됩니다.", foreground="#333333")
    value_entry.focus()


def main():
    window = tk.Tk()
    window.title("힘 단위 변환 계산기")
    window.resizable(False, False)

    frame = ttk.Frame(window, padding=20)
    frame.grid()

    ttk.Label(frame, text="힘 단위 변환 계산기", font=("맑은 고딕", 16, "bold")).grid(
        row=0, column=0, columnspan=2, pady=(0, 15)
    )
    ttk.Label(frame, text="숫자 입력:").grid(row=1, column=0, sticky="w", pady=5)
    value_entry = ttk.Entry(frame, width=20)
    value_entry.grid(row=1, column=1, sticky="ew", pady=5)

    ttk.Label(frame, text="단위 선택:").grid(row=2, column=0, sticky="w", pady=5)
    unit_combo = ttk.Combobox(frame, values=("kN", "N", "kgf"), state="readonly", width=17)
    unit_combo.set("kN")
    unit_combo.grid(row=2, column=1, sticky="ew", pady=5)

    button_frame = ttk.Frame(frame)
    button_frame.grid(row=3, column=0, columnspan=2, pady=15)
    result_label = tk.Label(
        frame,
        text="결과가 여기에 표시됩니다.",
        justify="left",
        anchor="w",
        width=36,
        height=4,
    )
    result_label.grid(row=4, column=0, columnspan=2, sticky="ew")

    ttk.Button(
        button_frame,
        text="변환",
        command=lambda: calculate(value_entry, unit_combo, result_label),
    ).grid(row=0, column=0, padx=5)
    ttk.Button(
        button_frame,
        text="초기화",
        command=lambda: reset(value_entry, unit_combo, result_label),
    ).grid(row=0, column=1, padx=5)

    value_entry.focus()
    window.mainloop()


if __name__ == "__main__":
    main()