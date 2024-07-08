import taichi as ti
from typing import Callable

ti.init(arch=ti.gpu)


f = None
def set_f(func: Callable) -> None:
    global f
    f = func

N = 2
NPART = 1

MTYPE = ti.types.matrix(n=N, m=N, dtype=ti.f32)
VTYPE = ti.types.vector(n=N, dtype=ti.f32)
PCOUNT_TYPE = ti.types.vector(n=NPART, dtype=ti.u16) # one for each particle
HIS = ti.field(dtype=MTYPE, shape=(NPART,)) # hessian inverses
GVALS = ti.field(dtype=VTYPE, shape=(NPART,)) # gradient values
# GRADS = ti.field(dtype=VTYPE, shape=(NPART,1)) # gradients
FCOUNT = PCOUNT_TYPE(ti.static([0] * NPART)) # objective funcion eval count
GCOUNT = PCOUNT_TYPE(ti.static([0] * NPART)) # gradient eval count

@ti.func
def fprime(x: VTYPE) -> VTYPE:
    return two_point_gradient(x)

@ti.func
def zero_vtype() -> VTYPE:
    v = VTYPE(ti.static([0.0] * N))
    return v

@ti.func
def two_point_gradient(x0: VTYPE, finite_difference_stepsize: ti.f32 = 1e-4) -> VTYPE:
    g = zero_vtype()
    fx0 = f(x0)

    p = zero_vtype()
    for pind in ti.static(range(N)):
        p[pind] = finite_difference_stepsize
        g[pind] = (f(x0 + p) - fx0) / finite_difference_stepsize
        p[pind] = 0.0
    return g

@ti.func
def matnorm(m: MTYPE, ord = ti.math.inf) -> ti.f32:
    v = ti.math.nan
    
    if ord == ti.math.inf:
        v = ti.abs(m).max()
    elif ord == -ti.math.inf:
        v = ti.abs(m).min()
    return v

@ti.func
def phi(i: int, xk: VTYPE, pk: VTYPE, s: ti.f32) -> ti.f32:
    FCOUNT[i] += 1
    return f(xk + s*pk)

@ti.func
def derphi(i: int, xk: VTYPE, pk: VTYPE, s: ti.f32) -> ti.f32:
    GVALS[i] = fprime(xk + s*pk)
    # print(xk, pk, s)
    GCOUNT[i] += 1
    return ti.math.dot(GVALS[i], pk)

@ti.func
def line_search_wolfe1(i: int,
                       xk: VTYPE,
                       pk: VTYPE, 
                       gfk: VTYPE,
                       old_fval: ti.f32, 
                       old_old_fval: ti.f32,
                       c1: ti.f32, 
                       c2: ti.f32, 
                       amin: ti.f32,
                       amax: ti.f32, 
                       xtol: ti.f32):
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
    FCOUNT[i] = 0
    GCOUNT[i] = 0

    derphi0 = ti.math.dot(gfk, pk)

    stp, fval, old_fval = scalar_search_wolfe1(i=i, xk=xk, pk=pk, phi0=old_fval, old_phi0=old_old_fval, derphi0=derphi0,
            c1=c1, c2=c2, amax=amax, amin=amin, xtol=xtol)

    return stp, FCOUNT[i], GCOUNT[i], fval, old_fval, GVALS[i]

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
                         xtol: ti.f32):
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

    alpha1 = ti.math.nan
    if derphi0 != 0:
        alpha1 = ti.min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)
        if alpha1 < 0:
            alpha1 = 1.0
    else:
        alpha1 = 1.0
    
    dcsrch = DCSRCH(xk=xk, pk=pk, ftol=c1, gtol=c2, xtol=xtol, stpmin=amin, stpmax=amax, i=i)
    stp, phi1, phi0, task = dcsrch.call(
        alpha1, phi0=phi0, derphi0=derphi0, maxiter=100
    )

    return stp, phi1, phi0

@ti.func
def clip(x: ti.f32, min_v: ti.f32, max_v: ti.f32) -> ti.f32:
    v = x
    if x < min_v:
        v = min_v
    if x > max_v:
        v = max_v
    return v

TASK_START = ti.cast(0, ti.u8)
TASK_WARNING = ti.cast(1, ti.u8)
TASK_FG = ti.cast(2, ti.u8)
TASK_ERROR = ti.cast(3, ti.u8)
TASK_CONVERGENCE = ti.cast(4, ti.u8)

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
        derphi1: ti.f32= derphi0

        task: ti.u8 = TASK_START
        inf_stp = False
        max_iter_hit = False
        something_else = False
        stp: ti.f32 = 0.0

        ti.loop_config(serialize=True)
        for j in range(maxiter):
            if not something_else and not inf_stp and not max_iter_hit:
                stp, phi1, derphi1, task = self.iterate(
                    alpha1, phi1, derphi1, task
                )
                
                if ti.math.isinf(stp):
                    inf_stp = True
                
                if not inf_stp:
                    if task == TASK_FG:
                        alpha1 = stp
                        phi1 = phi(self.i, self.xk, self.pk, stp)
                        derphi1 = derphi(self.i, self.xk, self.pk, stp)
                    else:
                        something_else = True

                if j == maxiter-1:
                    max_iter_hit = True

        # maxiter reached, the line search did not converge
        if max_iter_hit:
            task = TASK_WARNING
        elif inf_stp:
            task = TASK_ERROR

        return stp, phi1, phi0, task

    @ti.func
    def iterate(self, stp, f, g, task):
        p5 = 0.5
        p66 = 0.66
        xtrapl = 1.1
        xtrapu = 4.0
        skip = False

        if task == TASK_START:
            if stp < self.stpmin:
                print('fuck1')
                task = TASK_ERROR
            if stp > self.stpmax:
                print('fuck2')
                task = TASK_ERROR
            if g >= 0:
                print('fuck3')
                task = TASK_ERROR
            if self.ftol < 0:
                print('fuck4')
                task = TASK_ERROR
            if self.gtol < 0:
                task =TASK_ERROR
            if self.xtol < 0:
                task = TASK_ERROR
            if self.stpmin < 0:
                task = TASK_ERROR
            if self.stpmax < self.stpmin:
                print(self.ftol)
                print('fuck')
                task = TASK_ERROR

            if task == TASK_ERROR:
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
                task = TASK_WARNING
                print(101)
            if self.brackt and self.stmax - self.stmin <= self.xtol * self.stmax:
                task = TASK_WARNING
                print(102)
            if stp == self.stpmax and f <= ftest and g <= self.gtest:
                task = TASK_WARNING
                print(103)
            if stp == self.stpmin and (f > ftest or g >= self.gtest):
                task = TASK_WARNING
                print(104)


            # test for convergence
            if f <= ftest and abs(g) <= self.gtol * -self.ginit:
                task = TASK_CONVERGENCE

            # test for termination
            if task == TASK_WARNING or task == TASK_CONVERGENCE:
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

                    self.stx, fxm, gxm, self.sty, fym, gym, stp, self.brackt \
                    = dcstep(
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

                        (self.stx,
                        self.fx,
                        self.gx,
                        self.sty,
                        self.fy,
                        self.gy,
                        stp,
                        self.brackt) \
                        = dcstep(
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
def dcstep(stx: ti.f32, 
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
           stpmax: ti.f32):
    sgn_dp = sign(dp)
    sgn_dx = sign(dx)

    sgnd = sgn_dp * sgn_dx

    stpf = 0.0 # overwritten later

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
        gamma = s * ti.sqrt(max(0, (theta / s) ** 2 - (dx / s) * (dp / s)))
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = (gamma + (dx - dp)) + gamma
        r = p / q
        stpc = 0.0 # overwritten
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
