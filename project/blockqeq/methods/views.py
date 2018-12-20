import numpy as np
from django.shortcuts import render
import random

# Create your views here.
from methods.source import Asmuth_Bloom_sharing, rec_secret_AB, Mignotte_sharing, gen_Mignotte_sequence, \
    rec_secret_Mignotte, naive_sharing, rec_naive, make_Shamirs, Shamirs_rec_secret, generate_prime_number, \
    make_Blakley, Blakley_rec_secret


def hello(request):
    return render(request, 'hello.html')


def abscheme(request):
    if request.method == 'GET':
        return render(request, 'ab.html')
    if request.method == 'POST':
        M = int(request.POST['secret'])
        m = int(request.POST['threshold'])
        n = int(request.POST['participants'])
        shares = Asmuth_Bloom_sharing(M, m, n)
        sample = shares[random.sample(range(0,len(shares)), m)]
        return render(request, 'ab_result.html', context={'shares': shares, 'm': m, 'sample': sample})


def parse_shares_str(shares_str):
    res_tmp = []
    shares_str = shares_str.replace('[', '')
    #shares_str = shares_str.replace(']', '')
    shares_str = shares_str.replace('.', '')
    for row in shares_str.split(']\r\n'):
        res_tmp.append(row.replace(']','').split(' '))

    print(shares_str)
    res = []
    for elem in res_tmp:
        r = []
        for x in elem:
            if x != '':
                r.append(int(x))
        res.append(r)

    return np.array(res)


def abscheme_recover(request):
    if request.method == 'POST':
        shares = parse_shares_str(request.POST['shares'])
        print(shares)
        print(shares.shape)
        restored = rec_secret_AB(shares, shares.shape[0])
        return render(request, 'ab_recover.html', context={'M':restored})


def mignotte_sequence(request):
    if request.method == 'GET':
        return render(request, 'mignotte_sequence.html')
    if request.method == 'POST':
        m = int(request.POST['threshold'])
        n = int(request.POST['participants'])
        res = gen_Mignotte_sequence(m,n)
        return render(request, 'mignotte.html', context={'sequence':res[0], 'alpha': res[1], 'beta': res[2], 'm':m, 'n': n})


def mignotte_second_step(request):
    if request.method == 'POST':
        M = int(request.POST['secret'])
        m = int(request.POST['threshold'])
        n = int(request.POST['participants'])
        sequence = request.POST['sequence']
        print(sequence)
        shares = Mignotte_sharing(M, m, n, gen_Mignotte_sequence(m,n)[0])
        sample = shares[random.sample(range(0, len(shares)), m)]
        return render(request, 'mignotte_result.html', context={'shares': shares, 'm': m, 'sample': sample})


def mignotte_recover(request):
    if request.method == 'POST':
        shares = parse_shares_str(request.POST['shares'])
        print(shares)
        print(shares.shape)
        restored = rec_secret_Mignotte(shares, shares.shape[0])
        return render(request, 'mignotte_recover.html', context={'M':restored})


def naive(request):
    if request.method == 'GET':
        return render(request, 'naive.html')
    if request.method == 'POST':
        M = int(request.POST['secret'])
        n = int(request.POST['participants'])
        shares = naive_sharing(M, n)
        sample = shares
        return render(request, 'naive_result.html', context={'shares': shares, 'm': M, 'sample': sample})


def naive_recover(request):
    if request.method == 'POST':
        shares = parse_shares_str(request.POST['shares'])
        print(shares)
        print(shares.shape)
        restored = rec_naive(shares)
        return render(request, 'naive_recover.html', context={'M':''.join([str(i) for i in restored])})


def shamir(request):
    if request.method == 'GET':
        return render(request, 'shamir.html')
    if request.method == 'POST':
        M = int(request.POST['secret'])
        m = int(request.POST['threshold'])
        n = int(request.POST['participants'])
        p = generate_prime_number(16) # big length
        shares = make_Shamirs(M, m, n, p)
        sample = shares[random.sample(range(0, len(shares)), m)]
        return render(request, 'shamir_result.html', context={'shares': shares, 'm': m, 'sample': sample, 'p': p})


def shamir_recover(request):
    if request.method == 'POST':
        shares = parse_shares_str(request.POST['shares'])
        p = int(request.POST['prime'])
        print(shares)
        print(shares.shape)
        restored = Shamirs_rec_secret(shares, shares.shape[0], p)
        return render(request, 'shamir_recover.html', context={'M':restored})


def blakley(request):
    if request.method == 'GET':
        return render(request, 'blakley.html')
    if request.method == 'POST':
        M = int(request.POST['secret'])
        m = int(request.POST['threshold'])
        n = int(request.POST['participants'])
        p = generate_prime_number(32)  # big length
        shares = make_Blakley(M, m, n, p)
        sample = shares[random.sample(range(0, len(shares)), m)]
        return render(request, 'blakley_result.html', context={'shares': shares, 'm': m, 'sample': sample, 'p': p})


def blakley_recover(request):
    if request.method == 'POST':
        shares = parse_shares_str(request.POST['shares'])
        p = int(request.POST['prime'])
        print(shares)
        print(shares.shape)
        restored = Blakley_rec_secret(shares, p)
        return render(request, 'blakley_recover.html', context={'M': restored})
