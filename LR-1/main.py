import sys

def parse_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines()]
    lines = [ln for ln in lines if ln and not ln.lstrip().startswith('#')]

    if not lines:
        raise ValueError("Файл пустой")

    var_specs = None
    constraints = []

    i = 0
    if lines[0].lower().startswith('vars:'):
        var_specs = lines[0][5:].strip()
        i += 1

    if i >= len(lines):
        raise ValueError("Нет строки с целевой функцией.")
    first = lines[i]
    i += 1
    if first.lower().startswith('max:'):
        sense = 'max'
        coeffs = [float(s) for s in first[4:].strip().split()]
    elif first.lower().startswith('min:'):
        sense = 'min'
        coeffs = [float(s) for s in first[4:].strip().split()]
    else:
        raise ValueError("Первая непустая строка после vars должна быть max: или min:")
    while i < len(lines):
        ln = lines[i]; i += 1
        parts = ln.split()
        rel = None
        rel_idx = None
        for j, tok in enumerate(parts):
            if tok in ('<=', '>=', '='):
                rel = tok
                rel_idx = j
                break
        if rel is None:
            raise ValueError(f"Не найден знак отношения в ограничении: {ln}")
        a = [float(x) for x in parts[:rel_idx]]
        b = float(parts[rel_idx+1])
        constraints.append((a, rel, b))

    n = len(coeffs)
    for a, _, _ in constraints:
        if len(a) != n:
            raise ValueError("Число коэффициентов в ограничениях должно совпадать с числом коэффициентов целевой функции.")
    return sense, coeffs, constraints, var_specs


def construct_dual(sense, coeffs, constraints, var_specs=None):
    A = [list(a) for a, rel, b in constraints]
    rels = [rel for a, rel, b in constraints]
    b = [b for a, rel, b in constraints]

    if not A:
        raise ValueError("Нет ограничений в примальной задаче.")

    m = len(A)
    n = len(A[0])

    A_T = [[A[i][j] for i in range(m)] for j in range(n)]

    coeffs_dual = [float(x) for x in b]

    sense_dual = 'min' if sense == 'max' else 'max'

    y_specs = []
    for rel in rels:
        if rel == '<=':
            y_specs.append(">=0")
        elif rel == '=':
            y_specs.append("free")
        elif rel == '>=':
            y_specs.append("<=0")

    x_signs = ['>=0'] * n
    if var_specs:
        toks = var_specs.split()
        for j in range(n):
            if j < len(toks):
                t = toks[j]
                if 'free' in t:
                    x_signs[j] = 'free'
                elif '<=' in t:
                    x_signs[j] = '<=0'
                elif '>=' in t:
                    x_signs[j] = '>=0'
                else:
                    x_signs[j] = '>=0'

    constraints_dual = []
    for j in range(n):
        a_row = [float(v) for v in A_T[j]]
        x_sign = x_signs[j]
        if x_sign == '>=0':
            rel_dual = '>=' if sense_dual == 'min' else '<='
        elif x_sign == '<=0':
            rel_dual = '<=' if sense_dual == 'min' else '>='
        else:  # free
            rel_dual = '='
        rhs = float(coeffs[j])
        constraints_dual.append((a_row, rel_dual, rhs))

    var_specs_dual = " ".join(f"y{i+1}{y_specs[i]}" for i in range(m))

    if len(coeffs_dual) != m:
        raise RuntimeError(f"Внутренняя проверка: coeffs_dual.length ({len(coeffs_dual)}) != m ({m}). "
                           "Проверьте порядок ограничений в parse_file.")

    for a_row, rel, rhs in constraints_dual:
        if len(a_row) != m:
            raise RuntimeError("Внутреняя проверка: длина строки ограничений двойственной != m. "
                               f"len(a_row)={len(a_row)}, m={m}")
        
    return sense_dual, coeffs_dual, constraints_dual, var_specs_dual


def canonicalize(sense, coeffs, constraints, var_specs):
    n = len(coeffs)
    var_types = ['>=0'] * n
    if var_specs:
        tokens = var_specs.split()
        for j, tok in enumerate(tokens):
            if j >= n:
                break
            if 'free' in tok:
                var_types[j] = 'free'
            elif '<=' in tok:
                var_types[j] = '<=0'
            elif '>=' in tok:
                var_types[j] = '>=0'

    new_coeffs = []
    canon_to_orig = []

    for j, var_type in enumerate(var_types):
        c_j = coeffs[j]
        if var_type == 'free':
            new_coeffs.append(c_j)
            new_coeffs.append(-c_j)
            canon_to_orig.append((j, 1))
            canon_to_orig.append((j, -1))
        elif var_type == '<=0':
            new_coeffs.append(-c_j)
            canon_to_orig.append((j, -1))
        else:
            new_coeffs.append(c_j)
            canon_to_orig.append((j, 1))

    new_constraints = []
    for a, rel, b in constraints:
        if b < 0:
            a = [-ai for ai in a]
            b = -b
            if rel == '<=':
                rel = '>='
            elif rel == '>=':
                rel = '<='
        new_a = [0 for _ in range(len(new_coeffs))]
        

        new_a = [0 for _ in range(len(new_coeffs))]
        current_col = 0
        for j, var_type in enumerate(var_types):
            a_j = a[j]
            if var_type == 'free':
                new_a[current_col] += a_j
                new_a[current_col + 1] += -a_j
                current_col += 2
            elif var_type == '<=0':
                new_a[current_col] += -a_j
                current_col += 1
            else:
                new_a[current_col] += a_j
                current_col += 1

        new_constraints.append((new_a, rel, b))

    reconstruction_info = {
        'var_types': var_types,
        'canon_to_orig': canon_to_orig,
        'num_original': n
    }

    return new_coeffs, new_constraints, reconstruction_info


def reconstruct_original_solution(canon_x, reconstruction_info):
    num_original = reconstruction_info['num_original']
    canon_to_orig = reconstruction_info['canon_to_orig']
    
    orig_x = [0 for _ in range(num_original)]
    
    for canon_idx, (orig_idx, multiplier) in enumerate(canon_to_orig):
        if canon_idx < len(canon_x):
            orig_x[orig_idx] += multiplier * canon_x[canon_idx]
    
    return orig_x


def build_tableau(sense, coeffs, constraints):
    c = [ci for ci in coeffs]
    constr = [([x for x in a], rel, b) for (a, rel, b) in constraints]

    len_constrains = len(constr)
    len_coeffs = len(c)
    col_types = []
    A_rows = []
    b = []
    for (a, rel, bi) in constr:
        A_rows.append(a)
        b.append(bi)
    for i, (a, rel, bi) in enumerate(constr):
        if rel == '<=':
            for r in range(len_constrains):
                A_rows[r].append(1 if r == i else 0)
            col_types.append('slack')
        elif rel == '>=':
            for r in range(len_constrains):
                A_rows[r].append(-1 if r == i else 0)
            col_types.append('surplus')
            for r in range(len_constrains):
                A_rows[r].append(1 if r == i else 0)
            col_types.append('art')
        elif rel == '=':
            for r in range(len_constrains):
                A_rows[r].append(1 if r == i else 0)
            col_types.append('art')

    col_types = ['orig'] * len_coeffs + col_types
    c_ext = c + [0 for _ in range(len(col_types)-len_coeffs)]
    return A_rows, b, c_ext, col_types


def find_basic_columns(A):
    m = len(A)
    n = len(A[0])
    basic_cols = [-1]*m
    for j in range(n):
        col = [A[i][j] for i in range(m)]
        ones = [i for i,v in enumerate(col) if v == 1]
        if len(ones)==1 and all((v==0 or v==1) for v in col):
            row_idx = ones[0]
            if basic_cols[row_idx] == -1:
                basic_cols[row_idx] = j
    return basic_cols


def pivot(A, b, c, basis_rows, pivot_row, pivot_col):
    m = len(A)
    n = len(A[0])
    pv = A[pivot_row][pivot_col]
    A[pivot_row] = [val / pv for val in A[pivot_row]]
    b[pivot_row] = b[pivot_row] / pv
    for i in range(m):
        if i == pivot_row:
            continue
        factor = A[i][pivot_col]
        if factor != 0:
            A[i] = [A[i][j] - factor * A[pivot_row][j] for j in range(n)]
            b[i] = b[i] - factor * b[pivot_row]
    basis_rows[pivot_row] = pivot_col


def compute_reduced_costs(A, b, c, basis):
    m = len(A)
    n = len(A[0])
    cb = [c[basis[i]] for i in range(m)]
    reduced = [None]*n
    for j in range(n):
        s = c[j]
        for i in range(m):
            s -= cb[i] * A[i][j]
        reduced[j] = s
    z = sum(cb[i] * b[i] for i in range(m))
    return reduced, z


def simplex(A_init, b_init, c_init, art_cols_idx=None, maximize=True):
    A = [row[:] for row in A_init]
    b = [bi for bi in b_init]
    c = [ci for ci in c_init]
    m = len(A)
    n = len(A[0])

    basis = find_basic_columns(A)
    for i in range(m):
        if basis[i] == -1:
            found = False
            if art_cols_idx:
                for j in art_cols_idx:
                    if A[i][j] == 1:
                        col = [A[r][j] for r in range(m)]
                        if sum(1 for v in col if v != 0) == 1:
                            basis[i] = j
                            found = True
                            break
            if not found:
                for j in range(n):
                    col = [A[r][j] for r in range(m)]
                    if col[i] == 1 and all(col[r]==0 for r in range(m) if r!=i):
                        basis[i] = j
                        found = True
                        break
            if not found:
                pass

    iters = 0
    MAX_ITERS = 1000
    while True:
        iters += 1
        if iters > MAX_ITERS:
            raise RuntimeError("Слишком много итераций симплекс-метода")
        reduced, z = compute_reduced_costs(A, b, c, basis)
        entering = -1
        best = 0
        if maximize:
            for j in range(len(reduced)):
                if reduced[j] > best and (j not in basis):
                    best = reduced[j]
                    entering = j
        else:
            for j in range(len(reduced)):
                if reduced[j] < best and (j not in basis):
                    best = reduced[j]
                    entering = j

        if entering == -1:
            x = [0 for _ in range(n)]
            for i in range(m):
                if 0 <= basis[i] < n:
                    x[basis[i]] = b[i]
            return {
                'status': 'optimal',
                'x': x,
                'value': z,
                'basis': basis,
                'A': A,
                'b': b,
                'c': c,
                'reduced': reduced
            }
        col = [A[i][entering] for i in range(m)]
        ratios = []
        for i in range(m):
            aij = col[i]
            if aij > 0:
                ratios.append((b[i] / aij, i))
        if not ratios:
            return {'status':'unbounded', 'entering': entering}
        ratios.sort(key=lambda t: (t[0], basis[t[1]] if basis[t[1]]!=-1 else 10**9))
        _, pivot_row = ratios[0]
        pivot(A, b, c, basis, pivot_row, entering)

def two_phase_solve(sense, coeffs, constraints):
    A_rows, b, c_ext, col_types = build_tableau(sense, coeffs, constraints)
    m = len(A_rows)
    n = len(A_rows[0])
    art_cols_idx = [j for j,t in enumerate(col_types) if t == 'art']
    c_phase1 = [0] * n
    for j in art_cols_idx:
        c_phase1[j] = 1
    res1 = simplex(A_rows, b, c_phase1, art_cols_idx=art_cols_idx, maximize=False)

    if res1['status'] == 'unbounded':
        return {'status': 'unbounded_phase1'}
    min_val = res1['value']
    if min_val > 0:
        return {'status': 'infeasible', 'min_phase1': min_val}
    basis = res1['basis']
    A = res1['A']
    bsol = res1['b']

    current_A = [row[:] for row in A]
    current_b = bsol[:]
    current_basis = basis[:]
    current_c = c_ext[:]
    keep_cols = [j for j in range(n) if j not in art_cols_idx]
    col_index_map = {old: new for new, old in enumerate(keep_cols)}
    new_n = len(keep_cols)
    A2 = []
    for i in range(m):
        A2.append([current_A[i][j] for j in keep_cols])
    b2 = current_b[:]
    c2 = [current_c[j] for j in keep_cols]
    new_basis = []
    for i in range(m):
        old = current_basis[i]
        if old in col_index_map:
            new_basis.append(col_index_map[old])
        else:
            new_basis.append(-1)
    for i in range(m):
        if new_basis[i] == -1:
            for j in range(new_n):
                col = [A2[r][j] for r in range(m)]
                if col[i]==1 and all(col[r]==0 for r in range(m) if r!=i):
                    new_basis[i] = j
                    break

    maximize = (sense == 'max')
    res2 = simplex(A2, b2, c2, maximize=maximize)
    if res2['status']=='unbounded':
        return {'status':'unbounded'}
    if res2['status']!='optimal':
        return {'status':'failed', 'detail':res2}
    n_orig = sum(1 for t in col_types if t=='orig')
    result_x = [0] * n_orig
    for old_idx in range(n):
        if old_idx < n_orig and old_idx in keep_cols:
            new_j = col_index_map[old_idx]
            result_x[old_idx] = res2['x'][new_j]
    obj_value = sum(coeffs[i] * result_x[i] for i in range(n_orig))
    return {
        'status': 'optimal',
        'x': result_x,
        'value': obj_value,
        'phase1_min': min_val
    }


def main():
    if len(sys.argv) < 2:
            print("Использование: python script.py <файл_задачи> [--dual | -d]", file=sys.stderr)
            return

    path = sys.argv[1]
    dual_mode = False

    for arg in sys.argv[2:]:
        if arg in ('--dual', '-d'):
            dual_mode = True
            break
        else:
            print(f"Неизвестный аргумент: {arg}", file=sys.stderr)
            print("Использование: python script.py <файл_задачи> [--dual | -d]", file=sys.stderr)
            return

    try:
        sense, coeffs, constraints, var_specs = parse_file(path)
    except Exception as e:
        print("Ошибка при разборе файла:", e)
        return

    if dual_mode:
        sense, coeffs, constraints, var_specs = construct_dual(sense, coeffs, constraints, var_specs)

    c2, constraints2, reconstruction_info = canonicalize(sense, coeffs, constraints, var_specs)

    res = two_phase_solve(sense, c2, constraints2)

    if res['status'] == 'infeasible':
        print("Задача не имеет допустимых решений: несовместна (вспомогательная задача имеет минимальное значение > 0).")
        print("min(функции искусственных переменных) =", res.get('min_phase1', 'unknown'))
    elif res['status'] == 'unbounded':
        print("Задача не имеет оптимального решения: целевая функция неограничена на допустимом множестве.")
    elif res['status'] == 'optimal':
        print("Оптимальное решение найдено.")
        canon_solution = res['x']
        original_solution = reconstruct_original_solution(canon_solution, reconstruction_info)
        
        if dual_mode:
            var_names = []
            if var_specs:
                tokens = var_specs.split()
                for i in range(len(original_solution)):
                    if i < len(tokens):
                        name = tokens[i].split('>')[0].split('<')[0].split('=')[0].split('free')[0].rstrip()
                        if name and name[0] in 'xy':
                            var_names.append(name)
                        else:
                            var_names.append(f"y{i+1}")
                    else:
                        var_names.append(f"y{i+1}")
            else:
                var_names = [f"y{i+1}" for i in range(len(original_solution))]
        else:
            var_names = [f"x{i+1}" for i in range(len(original_solution))]

        print("Вектор переменных (исходные переменные):")
        for i, val in enumerate(original_solution):
            print(f" {var_names[i]} = {round(val, 5)}")
        print("Значение целевой функции:", res['value'])
    else:
        print("Не удалось решить задачу. Статус:", res['status'])
        if 'detail' in res:
            print(res['detail'])

if __name__ == '__main__':
    main()
