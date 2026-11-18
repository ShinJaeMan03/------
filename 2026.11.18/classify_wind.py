import csv
import math
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_NAME = "wind_dataset.csv"
OUTPUT_NAME = "classification_predictions.csv"
SCORES_OUTPUT_NAME = "classification_scores.csv"
GRAPH_OUTPUT_NAME = "strong_wind_december.png"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
TRAINING_START = datetime(2023, 1, 1)
EVALUATION_START = datetime(2024, 12, 1)
EVALUATION_END = datetime(2025, 1, 1)
FEATURE_COLUMNS = (
    "wind_speed_current",
    "temperature_current",
    "humidity_current",
    "wind_speed_1h_before",
    "wind_speed_2h_before",
)


def find_dataset() -> Path:
    """Find the dataset in this folder, with a fallback to the adjacent source folder."""
    candidates = [
        SCRIPT_DIR / DATASET_NAME,
        SCRIPT_DIR.parent / "2026.11.11" / DATASET_NAME,
    ]
    for path in candidates:
        if path.is_file():
            return path
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"{DATASET_NAME}를 찾을 수 없습니다. 검색 위치: {searched}")


def parse_time(value: str, column_name: str, row_number: int) -> datetime:
    try:
        return datetime.strptime(value, TIME_FORMAT)
    except ValueError as error:
        raise ValueError(
            f"{row_number}행의 {column_name} 값이 올바른 KST 형식이 아닙니다: {value!r}"
        ) from error


def read_and_classify(
    dataset_path: Path,
) -> list[dict[str, str | int | float | datetime]]:
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"prediction_time", "target_time", "wind_speed_target"}
        required_columns.update(FEATURE_COLUMNS)
        if not required_columns.issubset(reader.fieldnames or set()):
            missing = required_columns.difference(reader.fieldnames or set())
            raise ValueError(f"필수 열이 없습니다: {', '.join(sorted(missing))}")

        classified_rows: list[dict[str, str | int | float | datetime]] = []
        previous_prediction_time: datetime | None = None
        for row_number, row in enumerate(reader, start=2):
            prediction_time = parse_time(
                row["prediction_time"], "prediction_time", row_number
            )
            target_time = parse_time(row["target_time"], "target_time", row_number)

            if target_time - prediction_time != timedelta(hours=1):
                raise ValueError(
                    f"{row_number}행의 두 시각 차이가 정확히 1시간이 아닙니다: "
                    f"{row['prediction_time']} -> {row['target_time']}"
                )
            if (
                previous_prediction_time is not None
                and prediction_time < previous_prediction_time
            ):
                raise ValueError(f"{row_number}행에서 시간이 이전 행보다 빠릅니다.")
            previous_prediction_time = prediction_time

            try:
                target_wind = float(row["wind_speed_target"])
            except ValueError as error:
                raise ValueError(
                    f"{row_number}행의 wind_speed_target이 숫자가 아닙니다: "
                    f"{row['wind_speed_target']!r}"
                ) from error

            row_with_answer = dict(row)
            row_with_answer["answer"] = int(target_wind >= 4.0)
            row_with_answer["_target_time"] = target_time
            for column in FEATURE_COLUMNS:
                try:
                    row_with_answer[column] = float(row[column])
                except ValueError as error:
                    raise ValueError(
                        f"{row_number}행의 {column}이 숫자가 아닙니다: "
                        f"{row[column]!r}"
                    ) from error
            classified_rows.append(row_with_answer)

    return classified_rows


def count_answers(
    rows: list[dict[str, str | int | float | datetime]],
) -> tuple[int, int, int]:
    zero_count = sum(row["answer"] == 0 for row in rows)
    one_count = sum(row["answer"] == 1 for row in rows)
    return len(rows), zero_count, one_count


def make_features(
    rows: list[dict[str, str | int | float | datetime]],
) -> list[list[float]]:
    return [[float(row[column]) for column in FEATURE_COLUMNS] for row in rows]


def standardize(
    training_features: list[list[float]],
    evaluation_features: list[list[float]],
) -> tuple[list[list[float]], list[list[float]]]:
    feature_count = len(FEATURE_COLUMNS)
    means = [
        sum(row[index] for row in training_features) / len(training_features)
        for index in range(feature_count)
    ]
    standard_deviations = []
    for index, mean in enumerate(means):
        variance = sum(
            (row[index] - mean) ** 2 for row in training_features
        ) / len(training_features)
        standard_deviations.append(math.sqrt(variance) or 1.0)

    def transform(features: list[list[float]]) -> list[list[float]]:
        return [
            [
                (value - means[index]) / standard_deviations[index]
                for index, value in enumerate(row)
            ]
            for row in features
        ]

    return transform(training_features), transform(evaluation_features)


def fit_logistic_regression(
    features: list[list[float]], answers: list[int]
) -> tuple[float, list[float]]:
    """Fit binary logistic regression with L2 regularization and no class weights."""
    feature_count = len(FEATURE_COLUMNS)
    intercept = 0.0
    coefficients = [0.0] * feature_count
    learning_rate = 0.1
    regularization = 1.0

    for _ in range(2000):
        intercept_gradient = 0.0
        coefficient_gradients = [0.0] * feature_count
        for row, answer in zip(features, answers):
            linear_value = intercept + sum(
                coefficient * value
                for coefficient, value in zip(coefficients, row)
            )
            probability = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, linear_value))))
            error = probability - answer
            intercept_gradient += error
            for index, value in enumerate(row):
                coefficient_gradients[index] += error * value

        sample_count = len(features)
        intercept_gradient /= sample_count
        intercept -= learning_rate * intercept_gradient
        maximum_change = abs(learning_rate * intercept_gradient)
        for index in range(feature_count):
            gradient = (
                coefficient_gradients[index] / sample_count
                + regularization * coefficients[index] / sample_count
            )
            change = learning_rate * gradient
            coefficients[index] -= change
            maximum_change = max(maximum_change, abs(change))
        if maximum_change < 1e-7:
            break
    return intercept, coefficients


def logistic_probability(
    features: list[float], intercept: float, coefficients: list[float]
) -> float:
    linear_value = intercept + sum(
        coefficient * value for coefficient, value in zip(coefficients, features)
    )
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, linear_value))))


def create_predictions(
    training_rows: list[dict[str, str | int | float | datetime]],
    evaluation_rows: list[dict[str, str | int | float | datetime]],
) -> list[dict[str, str]]:
    training_features, evaluation_features = standardize(
        make_features(training_rows), make_features(evaluation_rows)
    )
    training_answers = [int(row["answer"]) for row in training_rows]

    intercept, coefficients = fit_logistic_regression(
        training_features, training_answers
    )
    logistic_probabilities = [
        logistic_probability(features, intercept, coefficients)
        for features in evaluation_features
    ]

    predictions: list[dict[str, str]] = []
    for row, probability in zip(evaluation_rows, logistic_probabilities):
        current_wind = float(row["wind_speed_current"])
        current_wind_prediction = int(current_wind >= 4.0)
        predictions.append(
            {
                "prediction_time": str(row["prediction_time"]),
                "target_time": str(row["target_time"]),
                "actual_wind_speed": str(row["wind_speed_target"]),
                "answer": str(row["answer"]),
                "current_wind_speed": f"{current_wind:.1f}",
                "current_wind_prediction": str(current_wind_prediction),
                "logistic_probability": f"{probability:.6f}",
                "logistic_prediction": str(int(probability >= 0.50)),
                "logistic_prediction_020": str(int(probability >= 0.20)),
            }
        )
    return predictions


def save_predictions(output_path: Path, predictions: list[dict[str, str]]) -> None:
    fieldnames = [
        "prediction_time",
        "target_time",
        "actual_wind_speed",
        "answer",
        "current_wind_speed",
        "current_wind_prediction",
        "logistic_probability",
        "logistic_prediction",
        "logistic_prediction_020",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)


def calculate_score(
    method: str, actuals: list[int], predicted: list[int]
) -> dict[str, str]:
    true_positive = sum(actual == 1 and prediction == 1 for actual, prediction in zip(actuals, predicted))
    false_positive = sum(actual == 0 and prediction == 1 for actual, prediction in zip(actuals, predicted))
    false_negative = sum(actual == 1 and prediction == 0 for actual, prediction in zip(actuals, predicted))
    true_negative = sum(actual == 0 and prediction == 0 for actual, prediction in zip(actuals, predicted))
    total = len(actuals)
    predicted_positive = true_positive + false_positive
    actual_positive = true_positive + false_negative
    precision = (
        f"{true_positive / predicted_positive:.6f}"
        if predicted_positive
        else "해당 없음"
    )
    recall = (
        f"{true_positive / actual_positive:.6f}"
        if actual_positive
        else "해당 없음"
    )
    return {
        "method": method,
        "accuracy": f"{(true_positive + true_negative) / total:.6f}",
        "precision": precision,
        "recall": recall,
        "true_positive": str(true_positive),
        "false_positive": str(false_positive),
        "false_negative": str(false_negative),
        "true_negative": str(true_negative),
        "predicted_positive": str(predicted_positive),
        "actual_positive": str(actual_positive),
        "evaluation_rows": str(total),
    }


def save_scores(output_path: Path, scores: list[dict[str, str]]) -> None:
    fieldnames = [
        "method",
        "accuracy",
        "precision",
        "recall",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
        "predicted_positive",
        "actual_positive",
        "evaluation_rows",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scores)


def save_week_graph(
    predictions: list[dict[str, str]], output_path: Path
) -> list[dict[str, str]]:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    graph_start = datetime(2024, 12, 1)
    graph_end = datetime(2024, 12, 8)
    week_predictions = [
        prediction
        for prediction in predictions
        if graph_start
        <= datetime.strptime(prediction["target_time"], TIME_FORMAT)
        < graph_end
    ]
    if not week_predictions:
        raise ValueError("12월 1~7일 평가 행이 없어 그래프를 만들 수 없습니다.")

    times = [
        datetime.strptime(prediction["target_time"], TIME_FORMAT)
        for prediction in week_predictions
    ]
    actual_wind = [
        float(prediction["actual_wind_speed"]) for prediction in week_predictions
    ]
    actual_answers = [int(prediction["answer"]) for prediction in week_predictions]
    prediction_050 = [
        int(prediction["logistic_prediction"]) for prediction in week_predictions
    ]
    prediction_020 = [
        int(prediction["logistic_prediction_020"])
        for prediction in week_predictions
    ]

    figure, (wind_axis, classification_axis) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    wind_axis.plot(times, actual_wind, color="tab:blue", label="실제 풍속")
    wind_axis.axhline(
        4.0, color="tab:red", linestyle="--", label="강풍 기준 4.0 m/s"
    )
    wind_axis.set_ylabel("풍속 (m/s)")
    wind_axis.set_title("2024년 12월 1~7일 풍속과 강풍 분류")
    wind_axis.grid(alpha=0.25)
    wind_axis.legend()

    classification_axis.plot(
        times,
        actual_answers,
        color="black",
        marker="o",
        linestyle="None",
        label="실제 강풍",
    )
    classification_axis.plot(
        times,
        prediction_050,
        color="tab:orange",
        marker="x",
        linestyle="None",
        label="로지스틱 0.50",
    )
    classification_axis.plot(
        times,
        prediction_020,
        color="tab:green",
        marker="+",
        linestyle="None",
        label="로지스틱 0.20",
    )
    classification_axis.set_yticks([0, 1])
    classification_axis.set_yticklabels(["강풍 아님 (0)", "강풍 (1)"])
    classification_axis.set_ylabel("분류")
    classification_axis.set_xlabel("정답 시각 (KST)")
    classification_axis.grid(alpha=0.25)
    classification_axis.legend(ncol=3)

    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return week_predictions


def find_december_second_case(
    predictions: list[dict[str, str]],
) -> dict[str, str] | None:
    for prediction in predictions:
        target_time = datetime.strptime(prediction["target_time"], TIME_FORMAT)
        if target_time.date() != datetime(2024, 12, 2).date():
            continue
        actual = int(prediction["answer"])
        prediction_050 = int(prediction["logistic_prediction"])
        prediction_020 = int(prediction["logistic_prediction_020"])
        if prediction_050 != actual or prediction_020 != actual:
            return prediction
    return None


def main() -> None:
    dataset_path = find_dataset()
    rows = read_and_classify(dataset_path)

    training_rows = [
        row
        for row in rows
        if TRAINING_START <= row["_target_time"] < EVALUATION_START
    ]
    evaluation_rows = [
        row
        for row in rows
        if EVALUATION_START <= row["_target_time"] < EVALUATION_END
    ]

    training_total, training_zero, training_one = count_answers(training_rows)
    evaluation_total, evaluation_zero, evaluation_one = count_answers(evaluation_rows)

    print(f"입력 파일: {dataset_path}")
    print("시각 기준: KST")
    print("행 순서: 시간순 확인 완료")
    print("각 행의 prediction_time과 target_time 차이: 모두 정확히 1시간")
    print(
        f"학습 기간 (target_time: 2023-01-01 ~ 2024-11-30): "
        f"전체 {training_total:,}행, 0={training_zero:,}, 1={training_one:,}"
    )
    print(
        f"평가 기간 (target_time: 2024-12-01 ~ 2024-12-31): "
        f"전체 {evaluation_total:,}행, 0={evaluation_zero:,}, 1={evaluation_one:,}"
    )

    if evaluation_rows:
        first_evaluation_row = evaluation_rows[0]
        print("첫 평가 행:")
        print(
            f"  prediction_time={first_evaluation_row['prediction_time']}, "
            f"target_time={first_evaluation_row['target_time']}, "
            f"wind_speed_target={first_evaluation_row['wind_speed_target']}, "
            f"answer={first_evaluation_row['answer']}"
        )
    else:
        print("첫 평가 행: 없음")

    if not training_rows or not evaluation_rows:
        raise ValueError("학습 또는 평가 데이터가 없어 예측을 만들 수 없습니다.")

    predictions = create_predictions(training_rows, evaluation_rows)
    output_path = SCRIPT_DIR / OUTPUT_NAME
    save_predictions(output_path, predictions)
    actuals = [int(prediction["answer"]) for prediction in predictions]
    baseline_predictions = [
        int(prediction["current_wind_prediction"]) for prediction in predictions
    ]
    logistic_predictions = [
        int(prediction["logistic_prediction"]) for prediction in predictions
    ]
    logistic_predictions_020 = [
        int(prediction["logistic_prediction_020"]) for prediction in predictions
    ]
    always_zero_predictions = [0] * len(predictions)
    scores = [
        calculate_score("항상 0", actuals, always_zero_predictions),
        calculate_score("현재 상태 유지", actuals, baseline_predictions),
        calculate_score("로지스틱 0.50", actuals, logistic_predictions),
        calculate_score("로지스틱 0.20", actuals, logistic_predictions_020),
    ]
    scores_output_path = SCRIPT_DIR / SCORES_OUTPUT_NAME
    save_scores(scores_output_path, scores)
    graph_output_path = SCRIPT_DIR / GRAPH_OUTPUT_NAME
    week_predictions = save_week_graph(predictions, graph_output_path)
    december_second_case = find_december_second_case(predictions)
    print("\n=== 12월 평가 예측 ===")
    print("표준화: 학습 행으로만 fit")
    print("로지스틱 회귀: 학습 행으로만 fit, L2 정규화, class_weight 미사용")
    print(f"저장 파일: {output_path}")
    print("CSV 첫 5행:")
    for prediction in predictions[:5]:
        print(prediction)
    print("\n=== 방법별 평가 ===")
    print("네 방법 모두 동일한 12월 평가 행 744개를 사용했습니다.")
    print("실제 강풍 수:", sum(actuals))
    for score in scores:
        print(score)
    print(f"점수 저장 파일: {scores_output_path}")
    print(f"그래프 대상 행 수: {len(week_predictions)}")
    print(f"그래프 저장 파일: {graph_output_path}")
    print("\n=== 12월 2일 틀린 사례 ===")
    if december_second_case is None:
        print("없음")
    else:
        print(december_second_case)


if __name__ == "__main__":
    main()
