from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = ("time_s", "force_N")
area__mm2 = 100
REFERENCE_STRESS_MPA = 6


def main():
    data_path = Path(__file__).with_name("load_data.csv")
    data = pd.read_csv(data_path)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"필수 열이 없습니다: {missing}")

    if data.empty:
        raise ValueError("CSV 파일에 데이터가 없습니다.")

    numeric_time = pd.to_numeric(data["time_s"], errors="coerce")
    numeric_force = pd.to_numeric(data["force_N"], errors="coerce")
    valid_mask = numeric_time.notna() & numeric_force.notna()
    invalid_rows = data[~valid_mask]

    for row_index, row in invalid_rows.iterrows():
        csv_row_number = row_index + 2
        for column in REQUIRED_COLUMNS:
            value = row[column]
            if pd.isna(value) or pd.isna(pd.to_numeric(value, errors="coerce")):
                display_value = "빈칸" if pd.isna(value) else value
                print(f"제외한 행: CSV {csv_row_number}행, {column} 문제값={display_value}")

    data = data.loc[valid_mask].copy()
    data["time_s"] = numeric_time.loc[data.index]
    data["force_N"] = numeric_force.loc[data.index]
    print(f"제외한 행 수: {len(invalid_rows)}개")
    print(f"유효한 데이터 수: {len(data)}개")
    if data.empty:
        print("유효한 데이터가 없어 계산과 그래프 생성을 중지합니다.")
        return

    data["stress_Mpa"] = data["force_N"] / area__mm2
    result_path = Path(__file__).with_name("load_result.csv")
    data.to_csv(result_path, index=False)

    maximum_force_index = data["force_N"].idxmax()
    maximum_force = data.loc[maximum_force_index, "force_N"]
    maximum_force_time = data.loc[maximum_force_index, "time_s"]
    maximum_stress = data.loc[maximum_force_index, "stress_Mpa"]
    maximum_stress_time = data.loc[maximum_force_index, "time_s"]
    exceeded_reference_count = (data["stress_Mpa"] > REFERENCE_STRESS_MPA).sum()

    plt.plot(data["time_s"], data["stress_Mpa"], marker="o")
    plt.scatter([maximum_stress_time], [maximum_stress], color="red", zorder=3)
    plt.annotate(
        f"{maximum_stress_time:g} s, {maximum_stress:g} MPa",
        (maximum_stress_time, maximum_stress),
        textcoords="offset points",
        xytext=(8, 8),
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Stress (MPa)")
    plt.savefig(Path(__file__).with_name("stress_plot.png"))
    plt.close()

    print(f"데이터 개수: {len(data)}개")
    print(f"최대 하중: {maximum_force} N")
    print(f"최대 하중 발생 시간: {maximum_force_time} s")
    print(f"최대 응력: {maximum_stress} MPa")
    print(f"최대 응력 발생 시간: {maximum_stress_time} s")
    print(f"기준 응력 {REFERENCE_STRESS_MPA} MPa 초과 데이터 개수: {exceeded_reference_count}개")


if __name__ == "__main__":
    main()
