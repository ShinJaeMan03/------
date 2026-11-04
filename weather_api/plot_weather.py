from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path(__file__).resolve().parent / "wind_weather.csv"
IMAGE_PATH = Path(__file__).resolve().parent / "wind_speed.png"


def load_wind_data() -> pd.DataFrame:
    """풍속 CSV를 읽고 datetime을 KST 시각으로 변환해 시간순으로 정렬한다."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"wind_weather.csv를 찾을 수 없습니다: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    required_columns = {"datetime", "wind_speed_ms"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"CSV에 필요한 열이 없습니다: {sorted(missing_columns)}")
    if data.empty:
        raise ValueError("wind_weather.csv에 자료가 없습니다.")

    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    data["wind_speed_ms"] = pd.to_numeric(data["wind_speed_ms"], errors="coerce")
    data = data.dropna(subset=["datetime"]).sort_values("datetime")
    if data.empty:
        raise ValueError("변환할 수 있는 datetime 자료가 없습니다.")

    return data.set_index("datetime")[["wind_speed_ms"]]


def make_hourly_data(data: pd.DataFrame) -> pd.DataFrame:
    """관측 범위의 누락 시각을 1시간 간격으로 추가해 결측 기온을 NaN으로 둔다."""
    hourly_index = pd.date_range(
        start=data.index.min().floor("h"),
        end=data.index.max().ceil("h"),
        freq="h",
    )
    return data.reindex(hourly_index)


def plot_wind_speed(data: pd.DataFrame) -> None:
    """시간-풍속 선 그래프를 저장하고 화면에 표시한다."""
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(data.index, data["wind_speed_ms"], color="tab:blue", linewidth=1.5)
    axis.set_title("Wind speed over time")
    axis.set_xlabel("Time (KST)")
    axis.set_ylabel("Wind speed (m/s)")
    axis.grid(True, alpha=0.3)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(IMAGE_PATH, dpi=150)
    print(f"그래프 저장 위치: {IMAGE_PATH}")
    plt.show()


def main() -> None:
    """CSV를 읽고 시간축을 보완한 뒤 기온 그래프를 만든다."""
    wind_data = make_hourly_data(load_wind_data())
    missing_times = int(wind_data["wind_speed_ms"].isna().sum())
    print(f"전체 시간 수: {len(wind_data)}")
    print(f"풍속 결측 시각 수: {missing_times}")
    plot_wind_speed(wind_data)


if __name__ == "__main__":
    main()
