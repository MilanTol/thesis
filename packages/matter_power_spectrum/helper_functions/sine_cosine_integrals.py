import math

GAMMA = 0.5772156649015328606
tolerance = 1e-18

def _si_series_rec(x, tol=tolerance, max_terms=2000):
    """Stable Maclaurin series for Si(x) using recurrence for terms."""
    term = x  # k=0 term
    s = term
    k = 0
    while k < max_terms:
        # recurrence: a_{k+1}/a_k = - x^2 * (2k+1) / ((2k+3)^2 * (2k+2))
        r = - (x*x) * (2*k + 1) / ((2*k + 3)**2 * (2*k + 2))
        term *= r
        k += 1
        s += term
        if abs(term) < tol * max(1.0, abs(s)):
            break
    return s

def _ci_series_rec(x, tol=tolerance, max_terms=2000):
    """Series for Ci(x) = gamma + ln x + sum_{k>=1} (-1)^k x^{2k}/(2k (2k)!)
    Using a factorial-safe per-iteration update (we compute ratio via factorial
    expression, but k stays modest so this is stable)."""
    # start at k=1
    k = 1
    term = - (x*x) / (2 * math.factorial(2))  # k=1 term
    s = term
    while k < max_terms:
        # compute next term's ratio using factorial relation safely
        # term_{k+1} = - term_k * x^2 * (2k) / ((2k+2)*(2k+1)*(2k+2)!) ... simplified via factorials
        # use direct factorial ratio to avoid algebra mistakes:
        ratio = - (x*x) * (2*k) * math.factorial(2*k) / ((2*k+2) * math.factorial(2*k+2))
        term *= ratio
        k += 1
        s += term
        if abs(term) < tol * max(1.0, abs(s)):
            break
    return GAMMA + math.log(x) + s

def _si_asymp_adaptive(x, tol=tolerance, max_terms=1000):
    """Adaptive asymptotic expansion for Si(x) = pi/2 - [ cos(x)*C(x) + sin(x)*S(x) ].
    C(x) = sum_m (-1)^m (2m)! / x^{2m+1}
    S(x) = sum_m (-1)^m (2m+1)! / x^{2m+2}
    We compute terms by recurrence to avoid huge factorials."""
    # initial terms (m=0)
    term_c = 1.0 / x            # (-1)^0 (0)! / x^{1}
    term_s = 1.0 / (x*x)        # (-1)^0 (1)! / x^{2}
    cos_sum = term_c
    sin_sum = term_s
    last_tc = abs(term_c)
    last_ts = abs(term_s)
    for m in range(1, max_terms):
        # recurrence:
        # term_c_{m} = - term_c_{m-1} * (2m-1)*(2m) / x^2
        # term_s_{m} = - term_s_{m-1} * (2m)*(2m+1) / x^2
        term_c = -term_c * (2*(m-1) + 1) * (2*(m-1) + 2) / (x*x)
        term_s = -term_s * (2*(m-1) + 2) * (2*(m-1) + 3) / (x*x)
        cos_sum += term_c
        sin_sum += term_s
        tc = abs(term_c)
        ts = abs(term_s)
        # stop when both contributions are small relative to their sums
        if tc < tol * max(1.0, abs(cos_sum)) and ts < tol * max(1.0, abs(sin_sum)):
            break
        # asymptotic series eventually diverge; stop if terms grow
        if tc > last_tc and ts > last_ts:
            break
        last_tc = tc
        last_ts = ts
    return math.pi/2 - (math.cos(x) * cos_sum + math.sin(x) * sin_sum)

def _ci_asymp_adaptive(x, tol=tolerance, max_terms=1000):
    """Adaptive asymptotic expansion for Ci(x) = sin(x)*C(x) - cos(x)*D(x),
    where C(x) = sum (-1)^m (2m)!/x^{2m+1}, D(x) = sum (-1)^m (2m+1)!/x^{2m+2}.
    Recurrences are the same as in Si asymptotic sums."""
    term_c = 1.0 / x        # for C(x)
    term_d = 1.0 / (x*x)    # for D(x)
    Csum = term_c
    Dsum = term_d
    last_tc = abs(term_c)
    last_td = abs(term_d)
    for m in range(1, max_terms):
        term_c = -term_c * (2*(m-1) + 1) * (2*(m-1) + 2) / (x*x)
        term_d = -term_d * (2*(m-1) + 2) * (2*(m-1) + 3) / (x*x)
        Csum += term_c
        Dsum += term_d
        tc = abs(term_c)
        td = abs(term_d)
        if tc < tol * max(1.0, abs(Csum)) and td < tol * max(1.0, abs(Dsum)):
            break
        if tc > last_tc and td > last_td:
            break
        last_tc = tc
        last_td = td
    return math.sin(x) * Csum - math.cos(x) * Dsum

def Si(x, tol=tolerance):
    """Improved Si(x). Uses scipy.special.sici if available, else stable series/asymptotic.
    tol controls internal stopping (roughly relative error)."""
    if x == 0.0:
        return 0.0
    sign = 1.0 if x >= 0.0 else -1.0
    xa = abs(x)
    # try SciPy if installed (most accurate & fast)
    try:
        import scipy.special as sp
        s, c = sp.sici(xa)
        return sign * float(s)
    except Exception:
        pass
    # select method: series for small x, asymptotic for large x
    if xa < 6.0:
        return sign * _si_series_rec(xa, tol=tol)
    else:
        return sign * _si_asymp_adaptive(xa, tol=tol)

def Ci(x, tol=tolerance):
    """Improved Ci(x) for real x (uses |x| to keep real branch as in your original).
    For x==0 returns -inf (log divergence)."""
    if x == 0.0:
        return -float('inf')
    xa = abs(x)
    try:
        import scipy.special as sp
        s, c = sp.sici(xa)
        return float(c)
    except Exception:
        pass
    if xa < 6.0:
        return _ci_series_rec(xa, tol=tol)
    else:
        return _ci_asymp_adaptive(xa, tol=tol)