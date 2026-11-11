import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt


CSV_PATH = Path(__file__).with_name("asos_108_2023_2024.csv")
DATASET_PATH = Path(__file__).with_name("wind_dataset.csv")
PREDICTIONS_PATH = Path(__file__).with_name("predictions.csv")
GRAPH_PATH = Path(__file__).with_name("prediction_week.png")
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

FEATURE_FIELDS = [
    "wind_speed_current",
    "wind_speed_1h_before",
    "temperature_current",
    "humidity_current",
    "wind_speed_2h_before",
]
OUTPUT_FIELDS = ["prediction_time", "target_time", "wind_speed_target"]


def is_blank(value: str | None) -> bool:
    return value is None or value.strip() == ""


def create_dataset() -> tuple[int, int]:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        source_rows = list(csv.DictReader(file))

    observations = {}
    for row in source_rows:
        if not is_blank(row.get("datetime")):
            observation_time = datetime.strptime(row["datetime"], TIME_FORMAT)
            observations[observation_time] = row

    first_time = min(observations)
    last_time = max(observations)
    hourly_rows = []
    current_time = first_time
    while current_time <= last_time:
        hourly_rows.append(observations.get(current_time, {}))
        current_time += timedelta(hours=1)

    fieldnames = [
        "prediction_time",
        "target_time",
        "wind_speed_current",
        "temperature_current",
        "humidity_current",
        "wind_speed_1h_before",
        "wind_speed_2h_before",
        "wind_speed_target",
    ]
    dataset_rows = []
    for index in range(2, len(hourly_rows) - 1):
        prediction_time = first_time + timedelta(hours=index)
        values = {
            "prediction_time": prediction_time.strftime(TIME_FORMAT),
            "target_time": (prediction_time + timedelta(hours=1)).strftime(TIME_FORMAT),
            "wind_speed_current": hourly_rows[index].get("wind_speed_ms", ""),
            "temperature_current": hourly_rows[index].get("temperature_C", ""),
            "humidity_current": hourly_rows[index].get("humidity_pct", ""),
            "wind_speed_1h_before": hourly_rows[index - 1].get("wind_speed_ms", ""),
            "wind_speed_2h_before": hourly_rows[index - 2].get("wind_speed_ms", ""),
            "wind_speed_target": hourly_rows[index + 1].get("wind_speed_ms", ""),
        }
        if all(not is_blank(values[field]) for field in fieldnames[2:]):
            dataset_rows.append(values)

    with DATASET_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset_rows)
    return len(hourly_rows), len(dataset_rows)


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [matrix[row][:] + [vector[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("선형회귀 계산에 필요한 독립적인 입력이 부족합니다.")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                left - factor * right
                for left, right in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def fit_linear_regression(rows: list[dict[str, str]]) -> list[float]:
    feature_count = len(FEATURE_FIELDS) + 1
    matrix = [[0.0] * feature_count for _ in range(feature_count)]
    vector = [0.0] * feature_count
    for row in rows:
        features = [1.0] + [float(row[field]) for field in FEATURE_FIELDS]
        target = float(row["wind_speed_target"])
        for left in range(feature_count):
            vector[left] += features[left] * target
            for right in range(feature_count):
                matrix[left][right] += features[left] * features[right]
    return solve_linear_system(matrix, vector)


def predict(coefficients: list[float], row: dict[str, str]) -> float:
    values = [1.0] + [float(row[field]) for field in FEATURE_FIELDS]
    return max(0.0, sum(coefficient * value for coefficient, value in zip(coefficients, values)))


def mean_absolute_error(actual: list[float], predicted: list[float]) -> float:
    return sum(abs(left - right) for left, right in zip(actual, predicted)) / len(actual)


def build_tree(
    samples: list[tuple[list[float], float]],
    depth: int,
    max_depth: int,
    min_samples_leaf: int,
    rng: random.Random,
) -> tuple:
    target_mean = sum(target for _, target in samples) / len(samples)
    if depth >= max_depth or len(samples) < min_samples_leaf * 2:
        return ("leaf", target_mean)

    feature_count = len(samples[0][0])
    feature_indices = rng.sample(range(feature_count), max(1, int(feature_count**0.5)))
    best_split = None
    best_error = float("inf")
    for feature_index in feature_indices:
        values = sorted({features[feature_index] for features, _ in samples})
        if len(values) < 2:
            continue
        step = max(1, len(values) // 8)
        thresholds = [
            (values[index - 1] + values[index]) / 2
            for index in range(step, len(values), step)
        ]
        for threshold in thresholds:
            left = [sample for sample in samples if sample[0][feature_index] <= threshold]
            right = [sample for sample in samples if sample[0][feature_index] > threshold]
            if len(left) < min_samples_leaf or len(right) < min_samples_leaf:
                continue
            left_mean = sum(target for _, target in left) / len(left)
            right_mean = sum(target for _, target in right) / len(right)
            error = sum((target - left_mean) ** 2 for _, target in left)
            error += sum((target - right_mean) ** 2 for _, target in right)
            if error < best_error:
                best_error = error
                best_split = (feature_index, threshold, left, right)

    if best_split is None:
        return ("leaf", target_mean)
    feature_index, threshold, left, right = best_split
    return (
        "node",
        feature_index,
        threshold,
        build_tree(left, depth + 1, max_depth, min_samples_leaf, rng),
        build_tree(right, depth + 1, max_depth, min_samples_leaf, rng),
    )


def tree_predict(tree: tuple, features: list[float]) -> float:
    if tree[0] == "leaf":
        return tree[1]
    if features[tree[1]] <= tree[2]:
        return tree_predict(tree[3], features)
    return tree_predict(tree[4], features)


def fit_random_forest(
    training_rows: list[dict[str, str]], evaluation_rows: list[dict[str, str]]
) -> list[float]:
    training_features = [
        [float(row[field]) for field in FEATURE_FIELDS]
        for row in training_rows
    ]
    training_targets = [float(row["wind_speed_target"]) for row in training_rows]
    evaluation_features = [
        [float(row[field]) for field in FEATURE_FIELDS]
        for row in evaluation_rows
    ]
    rng = random.Random(42)
    samples = list(zip(training_features, training_targets))
    trees = []
    for _ in range(100):
        bootstrap = [
            samples[rng.randrange(len(samples))]
            for _ in range(min(len(samples), 2000))
        ]
        trees.append(build_tree(bootstrap, 0, 6, 5, rng))
    return [
        max(0.0, sum(tree_predict(tree, features) for tree in trees) / len(trees))
        for features in evaluation_features
    ]


def save_week_graph(prediction_rows: list[dict[str, str]]) -> int:
    week_start = datetime(2024, 12, 1)
    week_end = datetime(2024, 12, 8)
    week_rows = [
        row for row in prediction_rows
        if week_start <= datetime.strptime(row["target_time"], TIME_FORMAT) < week_end
    ]
    times = [
        datetime.strptime(row["target_time"], TIME_FORMAT)
        for row in week_rows
    ]
    actual = [float(row["wind_speed_target"]) for row in week_rows]
    baseline = [float(row["baseline_prediction"]) for row in week_rows]
    regression = [float(row["linear_regression_prediction"]) for row in week_rows]

    plt.figure(figsize=(14, 6))
    plt.plot(times, actual, color="black", label="Observed")
    plt.plot(times, baseline, color="gray", linestyle="--", label="Current wind baseline")
    plt.plot(times, regression, color="blue", label="Linear regression")
    plt.xlabel("Target time (KST)")
    plt.ylabel("Wind speed (m/s)")
    plt.title("Wind speed predictions: 2024-12-01 to 2024-12-07")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(GRAPH_PATH, dpi=150)
    plt.close()
    return len(week_rows)


def main() -> None:
    hourly_count, dataset_count = create_dataset()
    with DATASET_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    evaluation_start = datetime(2024, 12, 1)
    evaluation_end = datetime(2025, 1, 1)
    training_rows = [
        row for row in rows
        if datetime.strptime(row["target_time"], TIME_FORMAT) < evaluation_start
    ]
    evaluation_rows = [
        row for row in rows
        if evaluation_start <= datetime.strptime(row["target_time"], TIME_FORMAT) < evaluation_end
    ]

    coefficients = fit_linear_regression(training_rows)
    actual = [float(row["wind_speed_target"]) for row in evaluation_rows]
    baseline_predictions = [float(row["wind_speed_current"]) for row in evaluation_rows]
    regression_predictions = [predict(coefficients, row) for row in evaluation_rows]
    random_forest_predictions = fit_random_forest(training_rows, evaluation_rows)

    prediction_rows = []
    for row, baseline, regression, random_forest in zip(
        evaluation_rows,
        baseline_predictions,
        regression_predictions,
        random_forest_predictions,
    ):
        prediction_rows.append({
            "prediction_time": row["prediction_time"],
            "target_time": row["target_time"],
            "wind_speed_target": row["wind_speed_target"],
            "baseline_prediction": f"{baseline:.6f}",
            "linear_regression_prediction": f"{regression:.6f}",
            "random_forest_prediction": f"{random_forest:.6f}",
        })

    with PREDICTIONS_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = [
            "prediction_time",
            "target_time",
            "wind_speed_target",
            "baseline_prediction",
            "linear_regression_prediction",
            "random_forest_prediction",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)

    week_count = save_week_graph(prediction_rows)
    print("=== 실습 3 결과 ===")
    print(f"시간 간격을 맞춘 뒤의 행 수: {hourly_count:,}")
    print(f"저장된 데이터셋 행 수: {dataset_count:,}")
    print(f"학습 행 수: {len(training_rows):,}")
    print(f"12월 평가 행 수: {len(evaluation_rows):,}")
    print(f"기준 예측 MAE: {mean_absolute_error(actual, baseline_predictions):.6f}")
    print(f"선형회귀 예측 MAE: {mean_absolute_error(actual, regression_predictions):.6f}")
    print(
        "랜덤포레스트 예측 MAE: "
        f"{mean_absolute_error(actual, random_forest_predictions):.6f}"
    )
    print("랜덤포레스트 설정: n_estimators=100, max_depth=6, min_samples_leaf=5, random_state=42")
    print(f"저장 파일: {PREDICTIONS_PATH.name}")
    print(f"그래프 대상 행 수: {week_count:,}")
    print(f"그래프 파일: {GRAPH_PATH.name}")
    print("\n=== 첫 평가 결과 ===")
    for row in prediction_rows[:1]:
        print(row)


if __name__ == "__main__":
    main()
