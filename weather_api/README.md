# 기상청 시간자료 풍속 분석

## 제출 파일

- `collect_weather.py`: 기상청 API에서 시간자료를 받아 CSV로 정리
- `plot_weather.py`: CSV를 읽어 풍속 그래프 생성
- `wind_weather.csv`: 분석에 사용한 관측자료
- `wind_speed.png`: 시간별 풍속 그래프

## 관측 정보

- 관측소: 서울(지점번호 `108`)
- 기간: 2025-10-08 00:00 ~ 2025-10-14 23:00 (KST)
- 관측 항목: 풍속 (`wind_speed_ms`, m/s)
- 자료 행 수: 168행
- 중복 행 수: 0행
- 풍속 결측 수: 0개
- 첫 시각: 2025-10-08 00:00:00
- 마지막 시각: 2025-10-14 23:00:00

## 출처

기상청 API 허브 시간자료(기간 조회):

<https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php>

요청에는 지점번호 108, 시작시각 202510080000, 종료시각 202510142300을 사용했습니다. 인증키는 코드, 이미지, README에 저장하지 않으며 실행할 때 터미널에서 입력합니다.

## 실행 방법

VS Code 터미널에서 제출 폴더로 이동한 뒤 실행합니다.

```powershell
cd "C:\Users\심재만\Desktop\코딩과 응용\weather_api"
& "C:\Users\심재만\Desktop\코딩과 응용\.venv\Scripts\python.exe" .\collect_weather.py
& "C:\Users\심재만\Desktop\코딩과 응용\.venv\Scripts\python.exe" .\plot_weather.py
```

`collect_weather.py` 실행 중 `KMA API 인증키를 입력하세요:`가 표시되면 본인의 인증키를 입력하고 Enter를 누릅니다. `getpass()`를 사용하므로 입력한 문자는 화면에 표시되지 않습니다.

## 확인 결과

수집 실행 결과 HTTP 상태 코드는 200이었고, 168개의 자료 행이 저장되었습니다. 시간순 정렬 후 중복은 없었으며 1시간 간격 자료에서 풍속 결측도 없었습니다.

## 그래프 해석

2025년 10월 8일부터 14일까지 풍속은 시간대에 따라 변동했지만 전체 관측 기간에 자료가 끊긴 구간은 없었습니다. 이 기간의 평균 풍속은 약 2.71 m/s로, 앞선 10월 1일부터 7일까지의 약 2.24 m/s보다 약 0.46 m/s 높았습니다.
