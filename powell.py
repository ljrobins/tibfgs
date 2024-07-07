import taichi as ti
ti.init(arch=ti.cpu)
import numpy as np

FLOAT32_SQRT_EPS = 0.0003452669770922512
vtype = ti.types.vector(3, dtype=ti.f32)

@ti.dataclass
class MinimizeScalarResult:
     x: ti.f32
     fval: ti.f32
     flag: ti.uint8
     num: ti.uint16
    

@ti.func
def is_finite_scalar(x: float) -> bool:
    if ti.math.isinf(x) or ti.math.isnan(x):
        return False
    return True


@ti.func
def minimize_scalar_bounded(bounds: ti.math.vec2, xatol=1e-5, maxiter=500, disp=0) -> MinimizeScalarResult:

    # Test bounds are of correct form
    x1, x2 = bounds

    assert x1 < x2, "The lower bound exceeds the upper bound."

    flag = 0

    golden_mean = 0.5 * (3.0 - ti.sqrt(5.0))
    a, b = x1, x2
    fulc = a + golden_mean * (b - a)
    nfc, xf = fulc, fulc
    rat = 0.0
    e = 0.0
    x = xf
    fx = func(x)
    num = 1
    fu = ti.math.inf

    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = FLOAT32_SQRT_EPS * ti.abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1

    while (ti.abs(xf - xm) > (tol2 - 0.5 * (b - a))):
        golden = 1
        # Check for parabolic fit
        if ti.abs(e) > tol1:
            golden = 0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = ti.abs(q)
            r = e
            e = rat

            # Check for acceptability of parabola
            if ((ti.abs(p) < ti.abs(0.5*q*r)) and (p > q*(a - xf)) and
                    (p < q * (b - xf))):
                rat = (p + 0.0) / q
                x = xf + rat

                if ((x - a) < tol2) or ((b - x) < tol2):
                    si = ti.math.sign(xm - xf) + ((xm - xf) == 0)
                    rat = tol1 * si
            else:      # do a golden-section step
                golden = 1

        if golden:  # do a golden-section step
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean*e

        si = ti.math.sign(rat) + (rat == 0)
        x = xf + si * ti.max(ti.abs(rat), tol1)
        fu = func(x)
        num += 1

        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = x, fu

        xm = 0.5 * (a + b)
        tol1 = FLOAT32_SQRT_EPS * ti.abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1

        if num >= maxiter:
            flag = 1
            break

    if ti.math.isnan(xf) or ti.math.isnan(fx) or ti.math.isnan(fu):
        flag = 2

    fval = fx

    return MinimizeScalarResult(x=x, fval=fval, flag=flag, num=num)

@ti.func
def line_for_search(x0: vtype, alpha: vtype, lower_bound: vtype, upper_bound: vtype) -> ti.math.vec2:
    """
    Given a parameter vector ``x0`` with length ``n`` and a direction
    vector ``alpha`` with length ``n``, and lower and upper bounds on
    each of the ``n`` parameters, what are the bounds on a scalar
    ``l`` such that ``lower_bound <= x0 + alpha * l <= upper_bound``.


    Parameters
    ----------
    x0 : np.array.
        The vector representing the current location.
        Note ``np.shape(x0) == (n,)``.
    alpha : np.array.
        The vector representing the direction.
        Note ``np.shape(alpha) == (n,)``.
    lower_bound : np.array.
        The lower bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded below, then ``lower_bound[i]``
        should be ``-np.inf``.
        Note ``np.shape(lower_bound) == (n,)``.
    upper_bound : np.array.
        The upper bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded above, then ``upper_bound[i]``
        should be ``np.inf``.
        Note ``np.shape(upper_bound) == (n,)``.

    Returns
    -------
    res : tuple ``(lmin, lmax)``
        The bounds for ``l`` such that
            ``lower_bound[i] <= x0[i] + alpha[i] * l <= upper_bound[i]``
        for all ``i``.

    """
    # get nonzero indices of alpha so we don't get any zero division errors.
    # alpha will not be all zero, since it is called from _linesearch_powell
    # where we have a check for this.

    for i in range(lower_bound.n):
        if alpha[i] == 0.0:
            lower_bound[i] = -ti.math.inf
            upper_bound[i] = ti.math.inf
            alpha[i] = 1.0

    low = (lower_bound - x0) / alpha
    high = (upper_bound - x0) / alpha

    # positive and negative indices
    pos = alpha > 0

    lmin_pos = pos * low
    lmin_neg = (~pos) * high

    lmax_pos = pos * high
    lmax_neg = (~pos) * low
    
    lmin = (lmin_pos + lmin_neg).max()
    lmax = (lmax_pos + lmax_neg).min()

    # if x0 is outside the bounds, then it is possible that there is
    # no way to get back in the bounds for the parameters being updated
    # with the current direction alpha.
    # when this happens, lmax < lmin.
    # If this is the case, then we can just return (0, 0)
    return ti.math.vec2([lmin, lmax]) if lmax >= lmin else ti.math.vec2([0.0, 0.0])

def linesearch_powell(p, xi, lower_bound, upper_bound, tol=1e-3):
    """Line-search algorithm using fminbound.

    Find the minimum of the function ``func(x0 + alpha*direc)``.

    lower_bound : np.array.
        The lower bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded below, then ``lower_bound[i]``
        should be ``-np.inf``.
        Note ``np.shape(lower_bound) == (n,)``.
    upper_bound : np.array.
        The upper bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded above, then ``upper_bound[i]``
        should be ``np.inf``.
        Note ``np.shape(upper_bound) == (n,)``.
    fval : number.
        ``fval`` is equal to ``func(p)``, the idea is just to avoid
        recomputing it so we can limit the ``fevals``.

    """
    @ti.func
    def myfunc(alpha):
        return func(p + alpha*xi)

    bound = line_for_search(p, xi, lower_bound, upper_bound)
    # we can use a bounded scalar minimization
    res = minimize_scalar_bounded(myfunc, bound, xatol=tol / 100)
    xi = res.x * xi
    return ti.struct(fval=res.fval, xp=p + xi, xi=xi)


@ti.func
def func(x):
    return x**3 - x**2

@ti.kernel
def run() -> MinimizeScalarResult:
    x = minimize_scalar_bounded(ti.math.vec2(0.0, 1.0))
    return x

@ti.kernel
def test_line_for_search() -> ti.math.vec2:
    x0 = vtype([1.0, 2.0, 3.0])
    alpha = vtype([1.0, 1.0, -1.0])
    lower_bound = vtype([-10.0, -10.0, -.1])
    upper_bound = vtype([10.0, 4.2, 1.0])
    x = line_for_search(x0, alpha, lower_bound, upper_bound)
    print(x)
    print(x0 + x[0] * alpha)
    print(x0 + x[1] * alpha)
    return x

@ti.kernel
def test_line_search_powell() -> ti.math.vec2:
    x0 = vtype([1.0, 2.0, 3.0])
    p = vtype([1.0, 1.0, -1.0])
    lower_bound = vtype([-10.0, -10.0, -.1])
    upper_bound = vtype([10.0, 4.2, 1.0])

    x = linesearch_powell(x0, p, lower_bound, upper_bound)
    print(x)
    print(x0 + x[0] * alpha)
    print(x0 + x[1] * alpha)
    return x

print(run())

x = test_line_search_powell()
print(x)