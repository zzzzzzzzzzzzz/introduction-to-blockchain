# coding=utf-8

from random import randrange, getrandbits
import numpy as np
from numpy.random import randint


# These functions will be used later, don't look at them
# Вспомогательные функции, которые вызываются при разделении и сборе секрета. Забей, тебе нужны только схемы ниже.

def is_prime(num, test_count=1000):
    if num == 1:
        return False
    if test_count >= num:
        test_count = num - 1
    for x in range(test_count):
        val = randint(1, num - 1)
        if pow(val, num - 1, num) != 1:
            return False
    return True


def generate_prime(M):
    found_prime = False
    while not found_prime:
        p = randint(M, 2 ** M)
        if is_prime(p):
            found_prime = True
            return p


def generate_big_prime(m, n, test_count=1000):
    found_prime = False
    while not found_prime:
        p = randint(2 ** (n - 1), 2 ** n)
        if is_prime(p, test_count):
            return p


def prime_test(n, k=128):
    if n == 2 or n == 3:
        return True
    if n <= 1 or n % 2 == 0:
        return False
    # find r and s
    s = 0
    r = n - 1
    while r & 1 == 0:
        s += 1
        r //= 2
    # do k tests
    for _ in range(k):
        a = randrange(2, n - 1)
        x = pow(a, r, n)
        if x != 1 and x != n - 1:
            j = 1
            while j < s and x != n - 1:
                x = pow(x, 2, n)
                if x == 1:
                    return False
                j += 1
            if x != n - 1:
                return False
    return True

def generate_prime_candidate(length):
    p = getrandbits(length)
    p |= (1 << length - 1) | 1 # big and not even
    return p


def generate_prime_number(length=1024):
    p = 4
    while not prime_test(p, 128):
        p = generate_prime_candidate(length)
    return p


def eea(a, b):
    if b == 0: return (1, 0)
    (q, r) = (a // b, a % b)
    (s, t) = eea(b, r)
    return (t, s - (q * t))


def find_inverse(a, f):
    b = eea(a, f)[0]
    if b < 1: b += f  # we only want positive values
    return b


def gcd(a, b):
    if (a == 0):
        return b
    return gcd(b % a, a)


def gen_mut_prime_to(d, n, p):
    found_mut_prime = False
    while not found_mut_prime:
        k = randint(p, 2 ** (n))
        g = []
        for i in range(len(d)):
            g.append(gcd(k, d[i]))
        if (all(g[i] == 1 for i in range(len(g)))):
            return k


def gen_mut_prime_to_2(n):
    k = generate_prime_number(n)
    return k


def gen_mut_prime_sequence(p, n):
    d = []
    while len(d) != n:
        di = gen_mut_prime_to_2(n)
        if di not in d:
            d.append(di)
    return sorted(d)


# Schemes

# All schemes need to get a secret, a threshold and number of users.

# ОГРАНИЧЕНИЯ

# В каждой схеме значение порога не должно превышать число участников
# В первой схеме (Асмут-Блум) ограничений на секрет нет, а во второй (Миньотт) сначала запускается функция,
# генерирующая открытый ключ-последовательность, которая накладывает ограничения на секрет.
# Подробнее опишу в комментах к схеме Миньотта.

# Asmuth-Bloom sharing sheme

# Эта функция тебе тоже не нужна, она вызывается в следующей.

def gen_sequence_for_AB(p, m, n):  # generating special sequence for algorithm, you don't need it
    found = False
    while not found:
        d = gen_mut_prime_sequence(p, n)
        D = np.array(d)
        if (np.prod(d[:m]) > p * np.prod(d[n - m + 1:])):
            found = True
            return d


# Asmuth-Bloom scheme

# Input: secret M, threshold m and number of users n. Output: numpy array(n,3) where rows are shares.

# Функция, разделяющая секрет по схеме Асмута-Блума.
# Собственно, на входе нужны три параметра: секрет, порог и число участников. Их задает пользователь приложения.
# На выходе получаешь массив, каждая строка которого - часть секрета для i-го участника.

def Asmuth_Bloom_sharing(M, m, n):
    Shares = np.zeros((n,3))
    p = randint(M, 4*M)
    d = gen_sequence_for_AB(p,m,n)
    r = randint(1,2**n)
    Mr = M+r*p
    for i in range(n):
        Shares[i,0] = p
        Shares[i,1] = d[i]
        Shares[i,2] = Mr%d[i]
    return Shares.astype(int)


# Secret recovering. Input: array of several shares (each share is a row) and number of shares. Output: secret

# Сбор секрета для схемы Асмута-Блума.
# На вход подается (в виде массива с 3мя столбцами) несколько строк из той матрицы, которую выдала предыдущая функкция (число строк должно быть больше порога)
# и значение порога (число людей, которого достаточно, чтобы восстановить секрет).
# Было бы здорово, если бы пользователь мог сам выбрать строки, которые пойдут на вход для восстановления секрета.

# В качестве результата выдает секрет.

def rec_secret_AB(Shares_k, k):
    p = Shares_k[0, 0]
    S = np.prod(Shares_k[:k, 1], axis=0)
    Sd = np.zeros(k)
    Sd_1 = np.zeros(k)
    for i in range(k):
        Sd[i] = int(S / Shares_k[i, 1])
        Sd_1[i] = int(find_inverse(Sd[i], Shares_k[i, 1]))
    Mr = sum(np.multiply(np.multiply(Shares_k[:k, 2], Sd), Sd_1)) % S
    M = Mr % p
    return int(M)


# Mignotte scheme.
# Вторая схема в этом файлике.
# Здесь нужно сначала сгенерировать последовательность Миньотта. На вход принимается порог m и общее число пользователей n.
# Эта функция выдает последовательность Миньотта и два числа: нижнее и верхнее ограничение на значение секрета.
# Эти числа нужно будет вывести на экран, чтобы пользователь выбрал в качестве секрета число, между этими двумя.
# Последовательность, впрочем, тоже можно вывести.

def gen_Mignotte_sequence(m, n):
    found = False
    while not found:
        d = gen_mut_prime_sequence(1, n)
        D = np.array(d)
        if (np.prod(d[:m]) > np.prod(d[n - m + 1:])):
            found = True
            return d, np.prod(d[n - m + 1:]), np.prod(d[:m])


# Input: secret M, threshhold m and number of users n. Output: numpy array(n,2) where rows are shares.
# Функция разделения секрета по схеме Миньотта. Принимает на вход секрет, порог, общее число участников
# И ПОСЛЕДОВАТЕЛЬНОСТЬ, СГЕНЕРИРОВАННУЮ ПРЕДЫДУЩЕЙ ФУНКЦИЕЙ

# Выдает массив, каждая строка которого - часть секрета.

def Mignotte_sharing(M, m, n, d):
    Shares = np.zeros((n, 2))
    for i in range(n):
        Shares[i, 1] = d[i]
        Shares[i, 0] = M % d[i]
    return Shares.astype(int)


# Secret recovering. Input: array of several shares (each share is a row) and number of shares. Output: secret

# Функция восстановления секрета по схеме Миньотта. На входе принимает несколько строк матрицы, полученной с помощью предыдущей функции
# и значение порога
# Выдает значение секрета.

def rec_secret_Mignotte(Shares_k, k):
    S = np.prod(Shares_k[:k, 1], axis=0)
    Sd = np.zeros(k)
    Sd_1 = np.zeros(k)
    for i in range(k):
        Sd[i] = int(S / Shares_k[i, 1])
        Sd_1[i] = int(find_inverse(Sd[i], Shares_k[i, 1]))
    M = sum(np.multiply(np.multiply(Shares_k[:k, 0], Sd), Sd_1)) % S
    return int(M)


# Naive (but not totally) secret sharing
# Наивная схема. Принимает на вход битовую строку и число людей (все, это наивная схема)

def naive_sharing(b, n):
    a = str(b)
    Shares = np.zeros((n, len(a)))
    for i in range(len(a)):
        for j in range(n - 1):
            Shares[j, i] = randint(0, 10)
        Shares[n - 1, i] = (int(a[i]) - sum(Shares[:, i])) % 10
    return Shares


# Восстановление секрета.
# Принимает на вход ту матрицу, которая генерилась предыдущей функцией и возвращает секрет. Можно даже не задействовать в приложении, пусть просто будет.

def rec_naive(Shares):
    return (sum(Shares) % 10).astype(int)


def make_Shamirs(secret, t, n, p):
    # secret
    coef = [secret]
    # generate our polynom (a_0, a_1, ...)
    coef = np.hstack((coef, randint(p, size=t - 1)))
    # points to share
    shares = []
    # computing the values, use x = 1, 2, 3, 4,...
    for i in range(1, n + 1):
        value = 0
        for j in range(t - 1, -1, -1):
            value = value * i + coef[j]
        shares.append((i, value))
    return np.array(shares)


def Shamirs_rec_secret(shares, t, p):
    secret = 0
    for i in range(t):
        olala = 1
        for j in range(t):
            if j != i:
                olala = (olala * shares[j][0] * pow(int(shares[j][0] - shares[i][0]), p - 2, p)) % p
        secret = (secret + shares[i][1] * olala) % p
    return secret


def make_Blakley(secret, t, n, p):
    # Generate x vector, x_0 equals to secret
    x = [secret]
    x = np.hstack((x, randint(p, size=t - 1)))
    # Generate Pascal Matrix (to avoid collinearity)
    A = np.ones((n, t)).astype(int)
    for r in range(1, n):
        for c in range(1, t):
            A[r, c] = (A[r, c - 1] + A[r - 1, c]) % p
    # Generate y vector, where Ax = y
    y = np.dot(A, x) % p
    # Split keys
    keys = [A[i].tolist() + [y[i]] for i in range(n)]
    # Return keys
    return np.array(keys)


def Blakley_rec_secret(keys, p):
    k = len(keys[0]) - 1
    # make full-rank matrix and corresponding y
    B = np.matrix([keys[i][:-1] for i in range(k)])[:k]
    y = np.array([keys[i][-1] for i in range(k)])[:k]
    for i in range(B.shape[0]):
        y[i] = (y[i] * pow(int(B[i, i]), p - 2, p)) % p
        B[i] = (B[i] * pow(int(B[i, i]), p - 2, p)) % p
        for j in range(i + 1, B.shape[1]):
            y[j] = (y[j] - B[j, i] * y[i]) % p
            B[j] = (B[j] - B[j, i] * B[i]) % p
    x = np.array(list([0] * k))
    for i in range(k - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, k):
            x[i] = (x[i] - x[j] * B[i, j]) % p
    return x[0]
