import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import math
from sympy import sympify, symbols, lambdify


def make_function_from_string(expr: str):
    x = symbols("x")

    try:
        sym_expr = sympify(expr, evaluate=True)
    except Exception as e:
        raise ValueError(f"Ошибка при разборе выражения SymPy: {e}")

    try:
        f_lambdified = lambdify(x, sym_expr, modules=["math"])
    except Exception as e:
        raise ValueError(f"Ошибка при создании функции: {e}")

    def f(x_val: float) -> float:
        try:
            return float(f_lambdified(x_val))
        except Exception as e:
            raise RuntimeError(f"Ошибка при вычислении функции в x={x_val}: {e}")

    return f


def estimate_L(f, a, b, n_samples = 20, safety = 1.1):
    xs = list(np.linspace(a, b, n_samples))
    fs = []
    for x in xs:
        fs.append(f(x))

    max_slope = 0.0
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i-1]
        if dx == 0:
            continue
        slope = abs((fs[i] - fs[i-1]) / dx)
        if slope > max_slope:
            max_slope = slope
    L = max_slope * safety
    if L <= 0:
        L = 1.0
    return L, xs, fs


def piyavskii_shubert(f, a, b, eps = 1e-3, max_iters = 10000, initial_samples_for_L = 20, safety_factor = 1.1):
    start_time = time.time()

    L, xs_init, fs_init = estimate_L(f, a, b, n_samples=initial_samples_for_L, safety=safety_factor)

    xs = list(xs_init)
    fs = list(fs_init)

    pairs = sorted(zip(xs, fs), key=lambda p: p[0])
    xs, fs = map(list, zip(*pairs))

    idx_min = int(np.argmin(fs))
    x_best = xs[idx_min]
    f_best = fs[idx_min]


    iterations = 1
    while iterations < max_iters:
        iterations += 1

        candidates = []
        for i in range(len(xs)-1):
            xi, xj = xs[i], xs[i+1]
            fi, fj = fs[i], fs[i+1]
            v = 0.5*(xi + xj) + (fi - fj) / (2.0 * L)
            if v <= xi or v >= xj:
                continue
            u = fi - L * abs(v - xi)
            candidates.append((u, v, i, i+1))
        if not candidates:
            intervals = [(xs[i+1]-xs[i], i) for i in range(len(xs)-1)]
            length, idx = max(intervals, key=lambda p: p[0])
            v = 0.5*(xs[idx] + xs[idx+1])
            u = max([fi - L * abs(v - xi) for xi, fi in zip(xs, fs)])
            candidates.append((u, v, idx, idx+1))

        psi_min, x_next, left_i, right_i = min(candidates, key=lambda t: t[0])

        if (f_best - psi_min) < eps:
            total_time = time.time() - start_time
            return x_best, f_best, iterations, total_time, xs, fs, L

        fx_next = f(x_next)

        lo, hi = 0, len(xs)
        while lo < hi:
            mid = (lo + hi) // 2
            if xs[mid] < x_next:
                lo = mid + 1
            else:
                hi = mid
        insert_pos = lo

        if insert_pos < len(xs) and abs(xs[insert_pos] - x_next) < 1e-12:
            fs[insert_pos] = fx_next
        else:
            xs.insert(insert_pos, x_next)
            fs.insert(insert_pos, fx_next)

        if fx_next < f_best:
            f_best = fx_next
            x_best = x_next

    total_time = time.time() - start_time
    return x_best, f_best, iterations, total_time, xs, fs, L


def plot_result(f, a, b, xs, fs, L, x_min, f_min):
    X = np.linspace(a, b, 1000)
    Y = np.array([f(x) for x in X])

    # миноранта
    psi_vals = np.full_like(X, -1e300, dtype=float)
    for xi, fi in zip(xs, fs):
        psi_vals = np.maximum(psi_vals, fi - L * np.abs(X - xi))

    plt.figure(figsize=(10, 6))
    plt.plot(X, Y, label="f(x) исходная функция", color='black', linewidth=1.5)
    plt.plot(X, psi_vals, linestyle='--', color='green', linewidth=1.5, label="миноранта (ломаная)")
    plt.scatter(xs, fs, color='blue', s=30, label="Точки вычисления f(x)")
    plt.scatter([x_min], [f_min], color='red', s=60, label="Найденный минимум", zorder=5)
    plt.title("Метод ломаных\nL ≈ {:.6g}".format(L))
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Использование: python piyavskii.py <входной_файл>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Файл '{input_file}' не найден.")
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f_in:
        lines = [line.strip() for line in f_in if line.strip()]

    if len(lines) < 4:
        print("Ошибка: в файле должно быть минимум 4 строки (функция, a, b, eps).")
        sys.exit(1)

    expr = lines[0]
    try:
        a = float(lines[1])
        b = float(lines[2])
        eps = float(lines[3])
    except ValueError:
        print("Ошибка: a, b и eps должны быть вещественными числами.")
        sys.exit(1)

    if b <= a:
        raise ValueError("Требуется a < b")
    
    if eps < 0:
        raise ValueError("Точность вычисления eps не может быть отрицательной")

    try:
        f = make_function_from_string(expr)
    except Exception as e:
        print("Ошибка при разборе функции:", e)
        sys.exit(1)

    x_min, f_min, func_calls, elapsed, xs_evals, fs_evals, L = piyavskii_shubert(f, a, b, eps=eps)

    def decimals_from_eps(eps):
        decimals = max(0, math.ceil(-math.log10(eps)))
        return decimals

    dec = decimals_from_eps(eps)

    if abs(x_min) < 1e-12:
        x_display = 0.0
    else:
        x_display = round(x_min, dec)

    f_display = round(f_min, dec+2)

    print("\nРезультаты:")
    print(f"Приближённое значение аргумента = {x_display}")
    print(f"Минимальное значение функции = {f_display}")
    print(f"Число итераций = {func_calls}")
    print(f"Затраченное время = {elapsed:.6f} с")
    print(f"Оценка константы Липшица L ≈ {L:.6g}")

    plot_result(f, a, b, xs_evals, fs_evals, L, x_min, f_min)


if __name__ == '__main__':
    main()
