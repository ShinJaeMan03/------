import math

def main():
    # P 입력: 숫자만 허용
    while True:
        try:
            P = float(input("축하중 P [kN] (인장 +, 압축 - ) = "))
            break
        except ValueError:
            print("숫자를 입력하세요.")

    # d 입력: 숫자만 허용, 0 이하 금지
    while True:
        try:
            d = float(input("직경 d [mm] = "))
            if d <= 0:
                print("직경 d는 0보다 커야 합니다.")
                continue
            break
        except ValueError:
            print("숫자를 입력하세요.")

    # 원형 단면적 [mm^2]
    A = math.pi * (d / 1000.0) ** 2 / 4.0

    # 축응력 [MPa]
    sigma = (P * 1000.0) / A  # P: kN -> N, A: mm^2 => N/mm^2 = MPa
    sigma_abs = abs(sigma) / 1_000_000.0

    if P > 0:
        stress_type = "인장"
    elif P < 0:
        stress_type = "압축"
    else:
        stress_type = "무응력"

    print("[1] 원형 봉의 축응력 (인장/압축)")
    print(f"축하중 P [kN] (인장 +, 압축 - ) = {P}")
    print(f"직경 d [mm] = {d}")
    print(f"응력 sigma = {sigma_abs:.2f} MPa ({stress_type})")

if __name__ == "__main__":
    main()