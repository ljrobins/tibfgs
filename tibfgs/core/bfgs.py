import taichi as ti
import os
from typing import Callable

N: ti.u8 = int(os.environ['TI_DIM_X'])
NPART: ti.i32 = int(os.environ['TI_NUM_PARTICLES'])

MTYPE = ti.types.matrix(n=N, m=N, dtype=ti.f32)
VTYPE = ti.types.vector(n=N, dtype=ti.f32)

EVAL_COUNTS = ti.field(
    dtype=ti.types.vector(n=2, dtype=ti.i32), shape=NPART
)  # one for each particle, function eval count
GVALS = ti.field(dtype=VTYPE, shape=NPART)  # gradient values

res_dict = dict(
    fun=ti.f32,
    xk=VTYPE,
    x0=VTYPE,
    status=ti.u8,
    task=ti.u8,
    k=ti.u16,
    grad=VTYPE,
    hess_inv=MTYPE,
    feval=ti.i32,
    geval=ti.i32,
)

res = ti.types.struct(**res_dict)

res_field = ti.Struct.field(
    res_dict,
    shape=NPART,
)


FLAG_SUCCESS = ti.cast(0, ti.u8)
FLAG_MAX_ITER = ti.cast(1, ti.u8)
FLAG_PRECISION_LOSS = ti.cast(2, ti.u8)
FLAG_NAN = ti.cast(3, ti.u8)
FLAG_ERROR = ti.cast(4, ti.u8)
FLAG_MAX_FEVAL = ti.cast(5, ti.u8)


@ti.kernel
def minimize_kernel(
    x0s: ti.template(), gtol: ti.f32, maxiter: ti.u16, maxfeval: ti.u16
) -> int:
    for i in x0s:
        res_field[i] = minimize_bfgs(
            i=i, x0=x0s[i], gtol=gtol, maxiter=maxiter, maxfeval=maxfeval
        )
    return 0


f = None


def set_f(func: Callable, eps: float = 1e-5) -> None:
    global f
    f = func

    os.environ['TI_BFGS_EPS'] = str(eps)


@ti.func
def fprime(x: VTYPE) -> VTYPE:
    return two_point_gradient(x, eps=ti.static(float(os.environ['TI_BFGS_EPS'])))


@ti.func
def two_point_gradient(x0: VTYPE, eps: ti.f32) -> VTYPE:
    g = VTYPE(0.0)
    fx0 = f(x0)

    for pind in range(N):
        p = VTYPE(0.0)
        p[pind] = eps
        g[pind] = (f(x0 + p) - fx0) / eps
        p[pind] = 0.0
    return g


@ti.func
def matnorm(m: MTYPE, ord=ti.math.inf) -> ti.f32:
    v = ti.math.nan

    if ord == ti.math.inf:
        v = ti.abs(m).max()
    elif ord == -ti.math.inf:
        v = ti.abs(m).min()
    return v


@ti.func
def vecnorm(v, ord=2.0) -> ti.f32:
    n: ti.f32 = 0.0

    if ti.math.isinf(ord):
        if ord == ti.math.inf:
            n = ti.abs(v).max()
        elif ord == -ti.math.inf:
            n = ti.abs(v).min()
    elif ord == 2.0:
        n = v.norm()
    else:
        n = (ti.abs(v) ** ord).sum() ** (1.0 / ord)
    return n


@ti.func
def phi(i: ti.i32, xk: VTYPE, pk: VTYPE, s: ti.f32) -> ti.f32:
    ti.atomic_add(EVAL_COUNTS[i][0], 1)
    return f(xk + s * pk)


@ti.func
def derphi(i: ti.i32, xk: VTYPE, pk: VTYPE, s: ti.f32) -> ti.f32:
    GVALS[i] = fprime(xk + s * pk)
    ti.atomic_add(EVAL_COUNTS[i][1], 1)
    ti.atomic_add(EVAL_COUNTS[i][0], N)
    return ti.math.dot(GVALS[i], pk)


@ti.func
def line_search_wolfe1(
    i: int,
    xk: VTYPE,
    pk: VTYPE,
    gfk: VTYPE,
    old_fval: ti.f32,
    old_old_fval: ti.f32,
    c1: ti.f32,
    c2: ti.f32,
    amin: ti.f32,
    amax: ti.f32,
    xtol: ti.f32,
):
    """
    As `scalar_search_wolfe1` but do a line search to direction `pk`

    Parameters
    ----------
    xk : array_like
        Current point
    pk : array_like
        Search direction
    gfk : array_like, optional
        Gradient of `f` at point `xk`
    old_fval : float, optional
        Value of `f` at point `xk`
    old_old_fval : float, optional
        Value of `f` at point preceding `xk`

    The rest of the parameters are the same as for `scalar_search_wolfe1`.

    Returns
    -------
    stp, f_count, g_count, fval, old_fval
        As in `line_search_wolfe1`
    gval : array
        Gradient of `f` at the final point

    """
    # FCOUNT[i] = 0
    # GCOUNT[i] = 0
    EVAL_COUNTS[i] = [0, 0]

    derphi0 = ti.math.dot(gfk, pk)

    # print(i, xk, pk, old_fval, old_old_fval, derphi0,
    #         c1, c2, amax, amin, xtol)

    stp, fval, old_fval, task = scalar_search_wolfe1(
        i=i,
        xk=xk,
        pk=pk,
        phi0=old_fval,
        old_phi0=old_old_fval,
        derphi0=derphi0,
        c1=c1,
        c2=c2,
        amax=amax,
        amin=amin,
        xtol=xtol,
    )

    return stp, EVAL_COUNTS[i].x, EVAL_COUNTS[i].y, fval, old_fval, GVALS[i], task


@ti.func
def scalar_search_wolfe1(
    i: int,
    xk: VTYPE,
    pk: VTYPE,
    phi0: ti.f32,
    old_phi0: ti.f32,
    derphi0: ti.f32,
    c1: ti.f32,
    c2: ti.f32,
    amax: ti.f32,
    amin: ti.f32,
    xtol: ti.f32,
):
    """
    Scalar function search for alpha that satisfies strong Wolfe conditions

    alpha > 0 is assumed to be a descent direction.

    Parameters
    ----------
    phi0 : float, optional
        Value of phi at 0
    old_phi0 : float, optional
        Value of phi at previous point
    derphi0 : float, optional
        Value derphi at 0
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax, amin : float, optional
        Maximum and minimum step size
    xtol : float, optional
        Relative tolerance for an acceptable step.

    Returns
    -------
    alpha : float
        Step size, or None if no suitable step was found
    phi : float
        Value of `phi` at the new point `alpha`
    phi0 : float
        Value of `phi` at `alpha=0`

    """

    alpha1: ti.f32 = 0.0
    if derphi0 != 0:
        alpha1 = ti.min(1.0, 2.02 * (phi0 - old_phi0) / derphi0)
        if alpha1 < 0:
            alpha1 = 1.0
    else:
        alpha1 = 1.0

    # print(i, xk, pk, phi0, old_phi0, derphi0,
    #         c1, c2, amax, amin, xtol)

    dcsrch = DCSRCH(
        xk=xk, pk=pk, ftol=c1, gtol=c2, xtol=xtol, stpmin=amin, stpmax=amax, i=i
    )
    stp, phi1, phi0, task = dcsrch.call(alpha1, phi0=phi0, derphi0=derphi0, maxiter=10)

    return stp, phi1, phi0, task


@ti.func
def clip(x: ti.f32, min_v: ti.f32, max_v: ti.f32) -> ti.f32:
    v = x
    if x < min_v:
        v = min_v
    if x > max_v:
        v = max_v
    return v


TASK_START = ti.cast(0, ti.u8)
TASK_FG = ti.cast(2, ti.u8)
TASK_CONVERGENCE = ti.cast(1, ti.u8)

TASK_ERROR = ti.cast(200, ti.u8)
TASK_ERROR_STP_LT_STPMIN = ti.cast(201, ti.u8)
TASK_ERROR_STP_GT_STPMAX = ti.cast(202, ti.u8)
TASK_ERROR_IG_GE_ZERO = ti.cast(203, ti.u8)
TASK_ERROR_FTOL_LT_ZERO = ti.cast(204, ti.u8)
TASK_ERROR_GTOL_LT_ZERO = ti.cast(205, ti.u8)
TASK_ERROR_XTOL_LT_ZERO = ti.cast(206, ti.u8)
TASK_ERROR_STPMIN_LT_ZERO = ti.cast(207, ti.u8)
TASK_ERROR_STPMAX_LT_STPMIN = ti.cast(208, ti.u8)


TASK_WARNING = ti.cast(100, ti.u8)
TASK_MAX_ITER_WARNING = ti.cast(101, ti.u8)
TASK_MAX_INF_STP = ti.cast(102, ti.u8)
TASK_WARNING_ROUNDING = ti.cast(103, ti.u8)
TASK_WARNING_XTOL = ti.cast(104, ti.u8)
TASK_WARNING_STP_EQ_STPMAX = ti.cast(105, ti.u8)
TASK_WARNING_STP_EQ_STPMIN = ti.cast(106, ti.u8)


@ti.dataclass
class DCSRCH:
    xk: VTYPE
    pk: VTYPE
    # leave all assessment of tolerances/limits to the first call of
    # this object
    ftol: ti.f32
    gtol: ti.f32
    xtol: ti.f32
    stpmin: ti.f32
    stpmax: ti.f32
    i: ti.u32

    # these are initialized to zero
    brackt: ti.u1
    stage: ti.f32
    ginit: ti.f32
    gtest: ti.f32
    gx: ti.f32
    gy: ti.f32
    finit: ti.f32
    fx: ti.f32
    fy: ti.f32
    stx: ti.f32
    sty: ti.f32
    stmin: ti.f32
    stmax: ti.f32
    width: ti.f32
    width1: ti.f32

    @ti.func
    def call(self, alpha1, phi0, derphi0, maxiter):
        """
        Parameters
        ----------
        alpha1 : float
            alpha1 is the current estimate of a satisfactory
            step. A positive initial estimate must be provided.
        phi0 : float
            the value of `phi` at 0 (if known).
        derphi0 : float
            the derivative of `derphi` at 0 (if known).
        maxiter : int

        Returns
        -------
        alpha : float
            Step size, or None if no suitable step was found.
        phi : float
            Value of `phi` at the new point `alpha`.
        phi0 : float
            Value of `phi` at `alpha=0`.
        task : bytes
            On exit task indicates status information.

        If task[:4] == b'CONV' then the search is successful.

        If task[:4] == b'WARN' then the subroutine is not able
        to satisfy the convergence conditions. The exit value of
        stp contains the best point found during the search.

        If task[:5] == b'ERROR' then there is an error in the
        input arguments.
        """

        phi1: ti.f32 = phi0
        derphi1: ti.f32 = derphi0

        task: ti.u8 = TASK_START
        inf_stp = False
        max_iter_hit = False
        something_else = False
        stp: ti.f32 = 0.0

        # print(self.xk, self.pk, self.ftol, self.gtol, self.xtol, self.stpmin, self.stpmax, self.i, alpha1, phi0, derphi0, maxiter)

        ti.loop_config(serialize=True)
        for j in range(maxiter):
            if not something_else and not inf_stp:
                stp, phi1, derphi1, task = self.iterate(alpha1, phi1, derphi1, task)

                if ti.math.isinf(stp):
                    inf_stp = True
                    continue

                if task == TASK_FG:
                    alpha1 = stp
                    phi1 = phi(self.i, self.xk, self.pk, stp)
                    derphi1 = derphi(self.i, self.xk, self.pk, stp)
                else:
                    something_else = True
                    continue

                if j == maxiter - 1:
                    max_iter_hit = True

        # maxiter reached, the line search did not converge
        if max_iter_hit:
            task = TASK_MAX_ITER_WARNING
        elif inf_stp:
            task = TASK_MAX_INF_STP

        return stp, phi1, phi0, task

    @ti.func
    def iterate(self, stp, f, g, task):
        p5 = 0.5
        p66 = 0.66
        xtrapl = 1.1
        xtrapu = 4.0
        skip = False

        if task == TASK_START:
            if stp < self.stpmin:  # STP .LT. STPMIN
                task = TASK_ERROR_STP_LT_STPMIN
            if stp > self.stpmax:  # STP .GT. STPMAX
                task = TASK_ERROR_STP_GT_STPMAX
            if g >= 0:  # INITIAL G .GE. ZERO
                task = TASK_ERROR_IG_GE_ZERO
            if self.ftol < 0:  # FTOL .LT. ZERO
                task = TASK_ERROR_FTOL_LT_ZERO
            if self.gtol < 0:  # GTOL .LT. ZERO
                task = TASK_ERROR_GTOL_LT_ZERO
            if self.xtol < 0:  # XTOL .LT. ZERO
                task = TASK_ERROR_XTOL_LT_ZERO
            if self.stpmin < 0:  # STPMIN .LT. ZERO
                task = TASK_ERROR_STPMIN_LT_ZERO
            if self.stpmax < self.stpmin:  # STPMAX .LT. STPMIN
                task = TASK_ERROR_STPMAX_LT_STPMIN

            if task >= 200:  # if we're in the error range
                skip = True

            # Initialize local variables.
            if not skip:
                self.brackt = False
                self.stage = 1
                self.finit = f
                self.ginit = g
                self.gtest = self.ftol * self.ginit
                self.width = self.stpmax - self.stpmin
                self.width1 = self.width / p5

                # The variables stx, fx, gx contain the values of the step,
                # function, and derivative at the best step.
                # The variables sty, fy, gy contain the value of the step,
                # function, and derivative at sty.
                # The variables stp, f, g contain the values of the step,
                # function, and derivative at stp.

                self.stx = 0.0
                self.fx = self.finit
                self.gx = self.ginit
                self.sty = 0.0
                self.fy = self.finit
                self.gy = self.ginit
                self.stmin = 0
                self.stmax = stp + xtrapu * stp
                task = TASK_FG
                skip = True

        if not skip:
            # in the original Fortran this was a location to restore variables
            # we don't need to do that because they're attributes.

            # If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
            # algorithm enters the second stage.
            ftest = self.finit + stp * self.gtest

            if self.stage == 1 and f <= ftest and g >= 0:
                self.stage = 2

            # test for warnings
            if self.brackt and (stp <= self.stmin or stp >= self.stmax):
                task = TASK_WARNING_ROUNDING
                # print("WARNING: ROUNDING ERRORS PREVENT PROGRESS")
            if self.brackt and self.stmax - self.stmin <= self.xtol * self.stmax:
                task = TASK_WARNING_XTOL
                # print("WARNING: XTOL TEST SATISFIED")
            if stp == self.stpmax and f <= ftest and g <= self.gtest:
                task = TASK_WARNING_STP_EQ_STPMAX
            if stp == self.stpmin and (f > ftest or g >= self.gtest):
                task = TASK_WARNING_STP_EQ_STPMIN

            # test for convergence
            if f <= ftest and ti.abs(g) <= ti.abs(self.gtol * self.ginit):
                task = TASK_CONVERGENCE

            # test for termination
            if (task >= 100 and task < 200) or task == TASK_CONVERGENCE:
                # if we're in the warning range (100-199) or we've converged
                skip = True

            # A modified function is used to predict the step during the
            # first stage if a lower function value has been obtained but
            # the decrease is not sufficient.
            if not skip:
                if self.stage == 1 and f <= self.fx and f > ftest:
                    # Define the modified function and derivative values.
                    fm = f - stp * self.gtest
                    fxm = self.fx - self.stx * self.gtest
                    fym = self.fy - self.sty * self.gtest
                    gm = g - self.gtest
                    gxm = self.gx - self.gtest
                    gym = self.gy - self.gtest

                    # Call dcstep to update stx, sty, and to compute the new step.
                    # dcstep can have several operations which can produce NaN
                    # e.g. inf/inf. Filter these out.

                    self.stx, fxm, gxm, self.sty, fym, gym, stp, self.brackt = dcstep(
                        self.stx,
                        fxm,
                        gxm,
                        self.sty,
                        fym,
                        gym,
                        stp,
                        fm,
                        gm,
                        self.brackt,
                        self.stmin,
                        self.stmax,
                    )

                    # Reset the function and derivative values for f
                    self.fx = fxm + self.stx * self.gtest
                    self.fy = fym + self.sty * self.gtest
                    self.gx = gxm + self.gtest
                    self.gy = gym + self.gtest

                else:
                    # Call dcstep to update stx, sty, and to compute the new step.
                    # dcstep can have several operations which can produce NaN
                    # e.g. inf/inf. Filter these out.

                    (
                        self.stx,
                        self.fx,
                        self.gx,
                        self.sty,
                        self.fy,
                        self.gy,
                        stp,
                        self.brackt,
                    ) = dcstep(
                        self.stx,
                        self.fx,
                        self.gx,
                        self.sty,
                        self.fy,
                        self.gy,
                        stp,
                        f,
                        g,
                        self.brackt,
                        self.stmin,
                        self.stmax,
                    )

                # Decide if a bisection step is needed
                if self.brackt:
                    if ti.abs(self.sty - self.stx) >= p66 * self.width1:
                        stp = self.stx + p5 * (self.sty - self.stx)
                    self.width1 = self.width
                    self.width = ti.abs(self.sty - self.stx)

                # Set the minimum and maximum steps allowed for stp.
                if self.brackt:
                    self.stmin = ti.min(self.stx, self.sty)
                    self.stmax = ti.max(self.stx, self.sty)
                else:
                    self.stmin = stp + xtrapl * (stp - self.stx)
                    self.stmax = stp + xtrapu * (stp - self.stx)

                # Force the step to be within the bounds stpmax and stpmin.
                stp = clip(stp, self.stpmin, self.stpmax)

                # If further progress is not possible, let stp be the best
                # point obtained during the search.
                if (
                    self.brackt
                    and (stp <= self.stmin or stp >= self.stmax)
                    or (
                        self.brackt
                        and self.stmax - self.stmin <= self.xtol * self.stmax
                    )
                ):
                    stp = self.stx

                # Obtain another function and derivative
                task = TASK_FG
        return stp, f, g, task


@ti.func
def sign(x):
    # behaves like numpy sign, returning 1.0 for x >= 0, -1 else
    return ti.math.sign(x) * 2.0 - 1.0


@ti.func
def dcstep(
    stx: ti.f32,
    fx: ti.f32,
    dx: ti.f32,
    sty: ti.f32,
    fy: ti.f32,
    dy: ti.f32,
    stp: ti.f32,
    fp: ti.f32,
    dp: ti.f32,
    brackt: ti.u1,
    stpmin: ti.f32,
    stpmax: ti.f32,
):
    sgn_dp = sign(dp)
    sgn_dx = sign(dx)

    sgnd = sgn_dp * sgn_dx

    stpf = 0.0  # overwritten later

    # First case: A higher function value. The minimum is bracketed.
    # If the cubic step is closer to stx than the quadratic step, the
    # cubic step is taken, otherwise the average of the cubic and
    # quadratic steps is taken.
    if fp > fx:
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        s = ti.max(ti.abs(theta), ti.abs(dx), ti.abs(dp))
        gamma = s * ti.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
        if stp < stx:
            gamma *= -1
        p = (gamma - dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = p / q
        stpc = stx + r * (stp - stx)
        stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (stp - stx)
        if ti.abs(stpc - stx) <= ti.abs(stpq - stx):
            stpf = stpc
        else:
            stpf = stpc + (stpq - stpc) / 2.0
        brackt = True
    elif sgnd < 0.0:
        # Second case: A lower function value and derivatives of opposite
        # sign. The minimum is bracketed. If the cubic step is farther from
        # stp than the secant step, the cubic step is taken, otherwise the
        # secant step is taken.
        theta = 3 * (fx - fp) / (stp - stx) + dx + dp
        s = ti.max(ti.abs(theta), ti.abs(dx), ti.abs(dp))
        gamma = s * ti.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
        if stp > stx:
            gamma *= -1
        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dx
        r = p / q
        stpc = stp + r * (stx - stp)
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        if ti.abs(stpc - stp) > ti.abs(stpq - stp):
            stpf = stpc
        else:
            stpf = stpq
        brackt = True
    elif ti.abs(dp) < ti.abs(dx):
        # Third case: A lower function value, derivatives of the same sign,
        # and the magnitude of the derivative decreases.

        # The cubic step is computed only if the cubic tends to infinity
        # in the direction of the step or if the minimum of the cubic
        # is beyond stp. Otherwise the cubic step is defined to be the
        # secant step.
        theta = 3 * (fx - fp) / (stp - stx) + dx + dp
        s = ti.max(ti.abs(theta), ti.abs(dx), ti.abs(dp))

        # The case gamma = 0 only arises if the cubic does not tend
        # to infinity in the direction of the step.
        gamma = s * ti.sqrt(ti.max(0, (theta / s) ** 2 - (dx / s) * (dp / s)))
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = (gamma + (dx - dp)) + gamma
        r = p / q
        stpc = 0.0  # overwritten
        if r < 0 and gamma != 0:
            stpc = stp + r * (stx - stp)
        elif stp > stx:
            stpc = stpmax
        else:
            stpc = stpmin
        stpq = stp + (dp / (dp - dx)) * (stx - stp)

        if brackt:
            # A minimizer has been bracketed. If the cubic step is
            # closer to stp than the secant step, the cubic step is
            # taken, otherwise the secant step is taken.
            if ti.abs(stpc - stp) < ti.abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq

            if stp > stx:
                stpf = ti.min(stp + 0.66 * (sty - stp), stpf)
            else:
                stpf = ti.max(stp + 0.66 * (sty - stp), stpf)
        else:
            # A minimizer has not been bracketed. If the cubic step is
            # farther from stp than the secant step, the cubic step is
            # taken, otherwise the secant step is taken.
            if ti.abs(stpc - stp) > ti.abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            stpf = clip(stpf, stpmin, stpmax)

    else:
        # Fourth case: A lower function value, derivatives of the same sign,
        # and the magnitude of the derivative does not decrease. If the
        # minimum is not bracketed, the step is either stpmin or stpmax,
        # otherwise the cubic step is taken.
        if brackt:
            theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp
            s = ti.max(ti.abs(theta), ti.abs(dy), ti.abs(dp))
            gamma = s * ti.sqrt((theta / s) ** 2 - (dy / s) * (dp / s))
            if stp > sty:
                gamma = -gamma
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dy
            r = p / q
            stpc = stp + r * (sty - stp)
            stpf = stpc
        elif stp > stx:
            stpf = stpmax
        else:
            stpf = stpmin

    # Update the interval which contains a minimizer.
    stx_ret = stx
    sty_ret = sty
    fx_ret = fx
    fy_ret = fy
    dx_ret = dx
    dy_ret = dy

    if fp > fx:
        sty_ret = stp
        fy_ret = fp
        dy_ret = dp

    if fp <= fx and sgnd < 0.0:
        sty_ret = stx
        fy_ret = fx
        dy_ret = dx

    if fp <= fx:
        stx_ret = stp
        fx_ret = fp
        dx_ret = dp

    # Compute the new step.
    stp = stpf

    return stx_ret, fx_ret, dx_ret, sty_ret, fy_ret, dy_ret, stp, brackt


@ti.func
def minimize_bfgs(
    i: ti.i32,
    x0: VTYPE,
    gtol: ti.f32 = 1e-4,
    norm: ti.f32 = ti.math.inf,
    eps: ti.f32 = 1e-6,
    maxiter: ti.u16 = 100,
    maxfeval: ti.u16 = 1000,
    xrtol=1e-6,
    c1=1e-4,
    c2=0.9,
) -> res:
    old_fval = f(x0)
    gfk = fprime(x0)

    eye = ti.Matrix.identity(dt=ti.f32, n=N)
    Hk = eye

    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + gfk.norm() / 2

    xk = x0

    warnflag = FLAG_SUCCESS
    ki = 0
    task = 0
    feval = 0
    geval = 0
    gnorm = vecnorm(gfk, ord=norm)
    ti.loop_config(serialize=True)
    for _ in range(maxiter):
        pk = -Hk @ gfk
        alpha_k, fc, gc, old_fval, old_old_fval, gfkp1, task = line_search_wolfe1(
            i,
            xk,
            pk,
            gfk,
            old_fval,
            old_old_fval,
            amin=1e-10,
            amax=1e10,
            c1=c1,
            c2=c2,
            xtol=xrtol,
        )
        feval += fc
        geval += gc

        if feval >= maxfeval:
            warnflag = FLAG_MAX_FEVAL
            break

        if task >= 200:  # if we're in the error range
            warnflag = FLAG_ERROR
            break

        if task != TASK_CONVERGENCE:
            # Line search failed to find a better solution.
            warnflag = FLAG_PRECISION_LOSS
            break

        sk = alpha_k * pk
        xkp1 = xk + sk

        xk = xkp1

        yk = gfkp1 - gfk
        gfk = gfkp1

        gnorm = vecnorm(gfk, ord=norm)
        if gnorm <= gtol:
            break

        #  See Chapter 5 in  P.E. Frandsen, K. Jonasson, H.B. Nielsen,
        #  O. Tingleff: "Unconstrained Optimization", IMM, DTU.  1999.
        #  These notes are available here:
        #  http://www2.imm.dtu.dk/documents/ftp/publlec.html
        if alpha_k * vecnorm(pk) <= xrtol * (xrtol + vecnorm(xk)):
            break

        if ti.math.isinf(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = FLAG_PRECISION_LOSS
            break

        rhok_inv = ti.math.dot(yk, sk)
        # this was handled in numeric, let it remains for more safety
        # Cryptic comment above is preserved for posterity. Future reader:
        # consider change to condition below proposed in gh-1261/gh-17345.
        rhok: ti.f32 = 0.0  # to be overwritten
        if rhok_inv == 0.0:
            rhok = 1000.0
            print('Divide-by-zero encountered: rhok assumed large')
        else:
            rhok = 1.0 / rhok_inv

        A1 = eye - sk.outer_product(yk) * rhok
        A2 = eye - yk.outer_product(sk) * rhok
        Hk = A1 @ (Hk @ A2) + rhok * sk.outer_product(sk)
        ki += 1

    fval = old_fval

    if ki == maxiter:
        warnflag = FLAG_MAX_ITER
    elif ti.math.isnan(gnorm) or ti.math.isnan(fval) or ti.math.isnan(xk).any():
        warnflag = FLAG_NAN

    return res(
        fun=fval,
        xk=xk,
        x0=x0,
        status=warnflag,
        task=task,
        k=ki,
        grad=gfk,
        hess_inv=Hk,
        feval=feval,
        geval=geval,
    )
