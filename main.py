import numpy as np
import sympy as sp
from scipy.integrate import quad, dblquad

def main():
    
    # Задание 1 (Создать матрицу 5х5 случ. вещ. чисел (0, 2), Транспонировать, вычислить определитель)
    matrix = np.random.uniform(0, 2, (5, 5))
    
    print('Задание 1')
    print('=' * 50)
    print(f'Матрица:\n{matrix}')
    print(f'Транспонированная матрица:\n{matrix.T}')
    print(f'Определитель матрицы:\n{np.linalg.det(matrix)}')
    print('=' * 50)
    print('\n')
    
    # Задание 2 (Создать вектор-столбец и матрицу подходящего размера. Умножить)
    vec = np.random.uniform(0, 3, (1, 5)).T
    
    print('Задание 2')
    print('=' * 50)
    print(f'Вектор-столбец:\n{vec}')
    print(f'Умножение вектора на матрицу из задание 1:\n{np.dot(matrix, vec)}')
    print('=' * 50)
    print('\n')
    
    # Задание 3 (Упростить выражение 28/b * (a + b) + (2 * a - 7 * b) ^ 2. И найти значения при a = sqrt(3), b = -3.42)
    a, b = sp.symbols('a, b')
    expr = 28 / b * (a + b) + (2 * a - 7 * b) ** 2
    expr_f = sp.sympify(expr)
    
    print('Задание 3')
    print('=' * 50)
    print(f'Выражение:\n{expr}')
    print(f'Упрощённое выражение:\n{sp.simplify(expr)}')
    print(f'Значение выражения при a = sqrt(3), b = -3.42\n{expr_f.subs({a: sp.sqrt(3), b: -3.42}).evalf()}')
    print('=' * 50)
    print('\n')
    
    # Задание 4 (Найти частные производные от выражения выше)
    expr_da = sp.diff(expr, a)
    expr_db = sp.diff(expr, b)
    
    print('Задание 4')
    print('=' * 50)
    print(f'Производная {expr} по a:\n{expr_da}')
    print(f'Производная {expr} по b:\n{expr_db}')
    print('=' * 50)
    print('\n')
    
    # Задание 5 (Найти соб. векотр и значение матрицы [ [0, -3, -1], [3, 8, 2], [-7, -15, -3] ])
    A = np.array([ [0, -3, -1], [3, 8, 2], [-7, -15, -3] ], dtype=np.int32)
    vals, vecs = np.linalg.eig(A)
    
    print('Задание 5')
    print('=' * 50)
    print(f'Матрица А:\n{A}')
    print(f'Собственные числа:\n{vals}')
    print(f'Собственные вектора:\n{vecs}')
    print('=' * 50)
    print('\n')
    
    # Задание 6 (Вычислить интеграл 4_0 (dx/(1 + sqrt(2 * x + 1))))
    f = lambda x: 1 / (1 + np.sqrt(2 * x + 1))
    x = sp.symbols('x')
    g = 1 / (1 + sp.sqrt(2 * x + 1))
    res_f, error = quad(f, 0, 4)
    res_g = sp.integrate(g, (x, 0, 4)).evalf()
    
    print('Задание 6')
    print('=' * 50)
    print(f'Опр. интеграл 4_0 (dx/(1 + sqrt(2 * x + 1))) через SciPy:\n{res_f}')
    print(f'Опр. интеграл 4_0 (dx/(1 + sqrt(2 * x + 1))) через SymPy:\n{res_g}')
    print('=' * 50)
    print('\n')
    
    # Задание 7 (Вычислить двойной интеграл pi_0 dx x_0 cos (x + y) dy)
    f = lambda y, x: np.cos(x + y)
    res_f, error = dblquad(f, 0, np.pi, lambda x: 0, lambda x: x)
    
    x, y = sp.symbols('x y')
    g = sp.cos(x + y)
    res_g = sp.integrate(g, (y, 0, x), (x, 0, sp.pi))
       
    print('Задание 7')
    print('=' * 50)
    print(f'Опр. интеграл pi/2_0 dx x_0 cos (x + y) dy через SciPy:\n{res_f}')
    print(f'Опр. интеграл pi_0 dx x_0 cos (x + y) dy через SymPy:\n{res_g}')
    print('=' * 50)
    print('\n')
    
if __name__ == '__main__':
    main()