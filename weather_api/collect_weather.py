from __future__ import annotations

import json
import re
import sys
from csv import DictWriter
from datetime import datetime
from getpass import getpass
from pathlib import Path

import requests


API_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php"
START_TIME = "202510080000"
END_TIME = "202510142300"
STATION = "108"
TIMEOUT_SECONDS = 10
OUTPUT_DIR = Path(__file__).resolve().parent
RAW_RESPONSE_PATH = OUTPUT_DIR / "wind_weather_raw.txt"
CSV_PATH = OUTPUT_DIR / "wind_weather.csv"
MISSING_VALUES = {"-9", "-9.0", "-99", "-99.0", "-999", "-999.0"}


def extract_message(body: str) -> str:
    """응답 본문에서 API 오류 메시지를 추출한다."""
    try:
        data = json.loads(body)
        if isinstance(data, dict):
            message = data.get("message")
            if message:
                return str(message)
            result = data.get("result")
            if isinstance(result, dict) and result.get("message"):
                return str(result["message"])
    except json.JSONDecodeError:
        pass

    match = re.search(
        r"(?:message|메시지)\s*[:=]\s*[\"']?(.+?)[\"']?(?:\r?\n|$)",
        body,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    for line in body.splitlines():
        stripped = line.strip()
        if stripped and ("error" in stripped.lower() or "오류" in stripped):
            return stripped

    return "API 요청에 실패했습니다."


def is_data_line(line: str) -> bool:
    """주석·제어 행을 제외하고 관측 시각으로 시작하는 자료 행인지 확인한다."""
    fields = line.split()
    return len(fields) > 13 and bool(re.fullmatch(r"\d{12}", fields[0]))


def parse_value(value: str) -> str:
    """API가 사용하는 결측 표식만 빈 문자열로 바꾸고 실제 음수는 보존한다."""
    return "" if value in MISSING_VALUES else value


def parse_weather_data(body: str) -> list[dict[str, str]]:
    """공백으로 구분된 관측 행에서 필요한 네 열을 추출하고 시간을 변환한다."""
    records: list[dict[str, str]] = []
    for line in body.splitlines():
        if line.lstrip().startswith("#") or not is_data_line(line):
            continue

        fields = line.split()
        try:
            observation_time = datetime.strptime(fields[0], "%Y%m%d%H%M")
        except ValueError:
            continue

        records.append(
            {
                "datetime": observation_time.isoformat(sep=" "),
                "temperature_C": parse_value(fields[11]),
                "humidity_pct": parse_value(fields[13]),
                "wind_speed_ms": parse_value(fields[3]),
            }
        )
    return records


def save_weather_csv(
    records: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    """시간순 정렬과 중복 제거 후 UTF-8 BOM CSV로 저장한다."""
    records.sort(key=lambda record: record["datetime"])
    unique_records: list[dict[str, str]] = []
    seen_times: set[str] = set()
    duplicate_count = 0

    for record in records:
        if record["datetime"] in seen_times:
            duplicate_count += 1
            continue
        seen_times.add(record["datetime"])
        unique_records.append(record)

    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = DictWriter(
            csv_file,
            fieldnames=[
                "datetime",
                "temperature_C",
                "humidity_pct",
                "wind_speed_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(unique_records)

    return unique_records, duplicate_count


def print_summary(records: list[dict[str, str]], row_count: int, duplicate_count: int) -> None:
    """저장된 표의 행 수, 결측 수, 시간 범위와 저장 위치를 출력한다."""
    missing_counts = {
        column: sum(record[column] == "" for record in records)
        for column in records[0]
    }
    print(f"\n행 수: {row_count}")
    print(f"중복 행 수: {duplicate_count}")
    print(f"열별 결측 수: {missing_counts}")
    print(f"첫 시각: {records[0]['datetime']}")
    print(f"마지막 시각: {records[-1]['datetime']}")
    print(f"저장 위치: {CSV_PATH}")


def main() -> None:
    auth_key = getpass("KMA API 인증키를 입력하세요: ").strip()
    if not auth_key:
        print("인증키가 입력되지 않았습니다.")
        sys.exit(1)

    params = {
        "tm1": START_TIME,
        "tm2": END_TIME,
        "stn": STATION,
        "help": "1",
        "authKey": auth_key,
    }

    try:
        response = requests.get(
            API_URL,
            params=params,
            timeout=TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        print(f"요청 중 네트워크 오류가 발생했습니다: {exc}")
        sys.exit(1)

    print(f"HTTP 상태 코드: {response.status_code}")
    RAW_RESPONSE_PATH.write_bytes(response.content)
    response.encoding = response.apparent_encoding or response.encoding
    body = response.text
    print(f"원본 응답 저장 위치: {RAW_RESPONSE_PATH}")

    if response.status_code != 200:
        print(f"오류 메시지: {extract_message(body)}")
        sys.exit(1)

    records = parse_weather_data(body)
    if not records:
        print("관측 자료 행이 없어 wind_weather.csv로 저장하지 않았습니다.")
        print("원인: 응답에 공백으로 구분된 유효한 관측 자료가 없습니다.")
        return

    unique_records, duplicate_count = save_weather_csv(records)
    print_summary(unique_records, len(unique_records), duplicate_count)


if __name__ == "__main__":
    main()
