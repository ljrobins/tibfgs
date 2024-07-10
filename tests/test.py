from spbfgs import vecnorm, rosen as rosen_np, finite_difference_gradient
import numpy as np
from tibfgs import matnorm, VTYPE, MTYPE, rosen as rosen_ti, set_f
import taichi as ti
import time


def test_matnorm():
    m = np.array([[-1.0, 1.0], [2.0, 3.0]])

    @ti.kernel
    def call_matnorm_ti(ord: ti.f32) -> ti.f32:
        m = MTYPE([[-1.0, 1.0], [2.0, 3.0]])
        return matnorm(m, ord=ord)

    assert np.allclose(call_matnorm_ti(ti.math.inf), vecnorm(m, ord=np.inf))
    assert np.allclose(call_matnorm_ti(-ti.math.inf), vecnorm(m, ord=-np.inf))


def test_vecnorm():
    from tibfgs import vecnorm

    @ti.kernel
    def call_vecnorm_ti(ord: ti.f32) -> ti.f32:
        v = ti.Vector([1.0, 2.0, 3.0])
        return vecnorm(v, ord=ord)

    assert np.allclose(call_vecnorm_ti(ti.math.inf), 3)
    assert np.allclose(call_vecnorm_ti(-ti.math.inf), 1)
    assert np.allclose(call_vecnorm_ti(2), np.sqrt(14))
    assert np.allclose(call_vecnorm_ti(3), np.cbrt(36))


def test_fdiff():
    from tibfgs import fprime

    set_f(rosen_ti)

    @ti.kernel
    def call_fdiff() -> VTYPE:
        return fprime(ti.math.vec2([-1.0, 1.0]))

    g_ti = call_fdiff()
    g_np = finite_difference_gradient(
        rosen_np,
        np.array([-1.0, 1.0], dtype=np.float32),
        finite_difference_stepsize=1e-5,
    )
    assert np.allclose(g_ti - g_np, 0 * g_np, atol=1e-2)


def test_dcstep():
    set_f(rosen_ti)
    from spbfgs import dcstep as dcstep_sp
    from tibfgs import dcstep as dcstep_ti

    stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax = (
        0.0,
        4.0,
        -15.99522009389491,
        0.0,
        4.0,
        -15.99522009389491,
        0.2525377248993902,
        100.95257568773,
        -23.945207713640414,
        False,
        0,
        1.2626886244969509,
    )

    tup_sp = dcstep_sp(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax)

    @ti.kernel
    def run_ti_dcstep() -> ti.types.vector(n=8, dtype=ti.f32):
        x = ti.Vector(
            dcstep_ti(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax)
        )
        return x

    tup_ti = run_ti_dcstep()

    for x, y in zip(tup_sp, tup_ti):
        assert np.allclose(x, y)


def test_dcsearch():
    set_f(rosen_ti)
    from spbfgs import DCSRCH as DCSRCH_np
    from tibfgs import DCSRCH as DCSRCH_ti

    x0 = np.array([-1.0, 1.0])
    (ftol, gtol, xtol, stpmin, stpmax) = (0.0001, 0.9, 1e-7, 1e-10, 1e10)

    f = rosen_np
    fprime = lambda x: finite_difference_gradient(f, x, finite_difference_stepsize=1e-5)

    gfk = fprime(x0)

    gval = [gfk]
    gc = [0]
    fc = [0]

    xk = x0
    pk = np.array([3.99940246e00, -1.49011612e-04])

    def phi(s):
        fc[0] += 1
        return f(xk + s * pk)

    def derphi(s):
        gval[0] = fprime(xk + s * pk)
        gc[0] += 1
        return np.dot(gval[0], pk)

    alpha1, phi0, derphi0, maxiter = (
        0.0011208198582888297,
        3.991124871148457,
        -15.99522009389491,
        100,
    )
    tup_sp = DCSRCH_np(phi, derphi, ftol, gtol, xtol, stpmin, stpmax)(
        alpha1, phi0=phi0, derphi0=derphi0, maxiter=maxiter
    )

    assert tup_sp[-1] == b'CONVERGENCE'

    tup_sp = np.array([*tup_sp[:3], 4.0])

    set_f(rosen_ti)

    @ti.kernel
    def run_ti_dcsrch() -> ti.types.vector(4, ti.f32):
        i: ti.i32 = 1
        x = ti.Vector(
            DCSRCH_ti(VTYPE(xk), VTYPE(pk), ftol, gtol, xtol, stpmin, stpmax, i=i).call(
                alpha1, phi0=phi0, derphi0=derphi0, maxiter=maxiter
            )
        )
        return x
        # return ti.Vector([0.0,0.0,0.0,0.0])

    tup_ti = run_ti_dcsrch().to_numpy()
    print(tup_ti)
    print(tup_sp)

    for x, y in zip(tup_sp[:3], tup_ti[:3]):
        assert np.allclose(x, y, atol=1e-4)


def test_scalar_search_wolfe1():
    set_f(rosen_ti)
    xkl = [-1.0, 1.0]
    pkl = [3.99940246e00, -1.49011612e-04]
    gfkl = [-3.99940246e00, 1.49011612e-04]
    old_fval = 3.991124871148457
    old_old_fval = 4.0
    f = rosen_np

    xk = np.array(xkl)
    pk = np.array(pkl)
    gfk = np.array(gfkl)
    gval = [np.array(gfkl)]
    gc = [0]
    fc = [0]

    def phi(s):
        fc[0] += 1
        return f(xk + s * pk)

    fprime = lambda x: finite_difference_gradient(f, x, finite_difference_stepsize=1e-4)

    def derphi(s):
        gval[0] = fprime(xk + s * pk)
        gc[0] += 1
        return np.dot(gval[0], pk)

    from spbfgs import scalar_search_wolfe1

    derphi0 = np.dot(gfk, pk)

    tup_np = scalar_search_wolfe1(
        phi,
        derphi,
        phi0=old_fval,
        old_phi0=old_old_fval,
        derphi0=derphi0,
        amin=1e-10,
        amax=1e10,
        c1=1e-4,
        c2=0.9,
    )

    from tibfgs import (
        scalar_search_wolfe1 as scalar_search_wolfe1_ti,
    )

    @ti.kernel
    def run_ti() -> ti.types.vector(4, ti.f32):
        x = ti.Vector(
            scalar_search_wolfe1_ti(
                i=10,
                xk=ti.math.vec2(xkl),
                pk=ti.math.vec2(pkl),
                phi0=old_fval,
                old_phi0=old_old_fval,
                derphi0=derphi0,
                c1=1e-4,
                c2=0.9,
                amax=1e10,
                amin=1e-10,
                xtol=1e-7,
            )
        )

        return x

    tup_ti = run_ti().to_numpy()[:3]
    assert np.allclose(tup_np, tup_ti)


def test_wolfe1():
    set_f(rosen_ti)
    xkl = [-1.0, 1.0]
    pkl = [3.99940246e00, -1.49011612e-04]
    gfkl = [-3.99940246e00, 1.49011612e-04]
    old_fval = 3.991124871148457
    old_old_fval = 4.0
    f = rosen_np

    from spbfgs import line_search_wolfe1

    myfprime = lambda x: finite_difference_gradient(
        f, x, finite_difference_stepsize=1e-4
    )

    tup_np = line_search_wolfe1(
        f=f,
        fprime=myfprime,
        xk=np.array(xkl),
        pk=np.array(pkl),
        gfk=np.array(gfkl),
        old_fval=old_fval,
        old_old_fval=old_old_fval,
        amin=1e-100,
        amax=1e100,
        c1=1e-4,
        c2=0.9,
    )

    print(tup_np)

    from tibfgs import line_search_wolfe1 as ti_line_search_wolfe1

    @ti.kernel
    def call_ti_line_search() -> ti.types.vector(n=4, dtype=ti.f32):
        xk = VTYPE(xkl)
        pk = VTYPE(pkl)
        gfk = VTYPE(gfkl)
        x = ti_line_search_wolfe1(
            i=0,
            xk=xk,
            pk=pk,
            gfk=gfk,
            old_fval=old_fval,
            old_old_fval=old_old_fval,
            amin=1e-10,
            amax=1e10,
            c1=1e-4,
            c2=0.9,
            xtol=1e-7,
        )

        return ti.Vector(x[:4])

    x = call_ti_line_search()
    print(x)


def test_bfgs():
    from spbfgs import minimize_bfgs as minimize_bfgs_np
    from tibfgs import minimize_bfgs as minimize_bfgs_ti, NPART

    x0 = np.array([-1.0, 1.0])
    res = minimize_bfgs_np(rosen_np, x0, eps=1e-6)
    t1 = time.time()
    for _i in range(100):
        res = minimize_bfgs_np(rosen_np, x0, eps=1e-6)
    print(1 / ((time.time() - t1) / 100))

    print(res)

    set_f(rosen_ti)

    res = ti.types.struct(
        fun=ti.f32, jac=VTYPE, hess_inv=MTYPE, status=ti.u8, xk=VTYPE, k=ti.u32
    )

    res_field = ti.Struct.field(
        dict(fun=ti.f32, jac=VTYPE, hess_inv=MTYPE, status=ti.u8, xk=VTYPE, k=ti.u32),
        shape=(NPART,),
    )

    @ti.kernel
    def run() -> int:
        x0 = ti.math.vec2([-1.0, 1.0])
        fval = 0.0
        gfk = ti.math.vec2([0.0, 0.0])
        warnflag = 0
        xk = x0
        k = 0
        Hk = MTYPE([[0.0, 0.0], [0.0, 0.0]])
        for i in range(NPART):
            x0 = ti.math.vec2([ti.random(), ti.random()])
            fval, gfk, Hk, warnflag, xk, k = minimize_bfgs_ti(
                i=i, x0=x0, gtol=1e-3, eps=1e-6
            )
            res_field[i] = res(
                fun=fval, jac=gfk, hess_inv=Hk, status=warnflag, xk=xk, k=k
            )
        return 0

    t1 = time.time()
    run()
    print(1 / ((time.time() - t1) / 1e6))
    print(res_field.xk)
