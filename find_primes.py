import math

def is_prime(n):
    if n < 2: return False
    if n == 2 or n == 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    for i in range(5, int(math.sqrt(n)) + 1, 6):
        if n % i == 0 or n % (i + 2) == 0:
            return False
    return True

def get_primitive_root(p):
    if p == 2: return 1
    factors = []
    phi = p - 1
    n = phi
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            factors.append(i)
            while n % i == 0:
                n //= i
    if n > 1:
        factors.append(n)
    
    for res in range(2, p):
        ok = True
        for factor in factors:
            if pow(res, phi // factor, p) == 1:
                ok = False
                break
        if ok:
            return res
    return None

def verify_ntt(n, q, root):
    if (q - 1) % (2 * n) != 0:
        return False
    
    # root^((q-1)/n) mod q should be a primitive n-th root of unity
    exponent = (q - 1) // n
    n_root = pow(root, exponent, q)
    if pow(n_root, n, q) != 1:
        return False
    if pow(n_root, n // 2, q) == 1:
        return False
        
    # negacyclic twiddle: root^((q-1)/(2*n)) mod q
    negacyclic_exponent = (q - 1) // (2 * n)
    twiddle = pow(root, negacyclic_exponent, q)
    if pow(twiddle, 2 * n, q) != 1:
        return False
    if pow(twiddle, n, q) != (q - 1):
        return False
        
    return True

n = 32768
target_bits = 50
start_val = (1 << (target_bits - 1)) + 1
count = 0
found = []

# Existing primes to avoid
excluded = [
    1125899908022273, 1125899908612097, 1125899909398529, 1125899910316033,
    1125899911168001, 1125899911364609, 1125899913527297, 1125899914510337,
    562949953556481, 562949953687553
]

# We want 50 bits. 1 << 50 is approx 1.125e15. 
# 1 << 49 is 562949953421312.
# Let's start from a bit above 1 << 49.
k = (1 << 49) // (2 * n)
while count < 2:
    q = k * (2 * n) + 1
    if q > (1 << 50):
        break
    if q not in excluded and is_prime(q):
        root = get_primitive_root(q)
        if root and verify_ntt(n, q, root):
            found.append((q, root))
            count += 1
    k += 1

for q, root in found:
    print(f"Prime: {q}, Root: {root}, Bits: {q.bit_length()}")
