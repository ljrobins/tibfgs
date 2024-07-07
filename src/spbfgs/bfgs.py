import numpy as np
from typing import Callable, Literal

_epsilon = np.sqrt(np.finfo(float).eps)

_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                              'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.',
                   'nan': 'NaN result encountered.',
                   'out_of_bounds': 'The result is outside of the provided '
                                    'bounds.'}

class _LineSearchError(RuntimeError):
    pass

def finite_difference_gradient(f: Callable, 
                               x0: np.ndarray, 
                               finite_difference_stepsize: float) -> np.ndarray:
    g = np.zeros_like(x0)
    fx0 = f(x0)
    for pind in range(x0.size):
        p = np.zeros_like(x0)
        p[pind] = finite_difference_stepsize
        g[pind] = (f(x0 + p) - fx0) / finite_difference_stepsize
    return g

def vecnorm(x, ord=2):
    if ord == np.inf:
        return np.amax(np.abs(x))
    elif ord == -np.inf:
        return np.amin(np.abs(x))
    else:
        return np.sum(np.abs(x)**ord, axis=0)**(1.0 / ord)

def minimize_bfgs(fun, x0, args=(), jac=None, callback=None,
                   gtol=1e-5, norm=np.inf, eps=_epsilon * 100, maxiter=None,
                   disp=False, return_all=False, finite_diff_rel_step=None,
                   xrtol=0, c1=1e-4, c2=0.9,
                   hess_inv0=None):
    """
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Terminate successfully if gradient norm is less than `gtol`.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    finite_diff_rel_step : None or array_like, optional
        If ``jac in ['2-point', '3-point', 'cs']`` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x) * max(1, abs(x))``,
        possibly adjusted to fit into the bounds. For ``jac='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    xrtol : float, default: 0
        Relative tolerance for `x`. Terminate successfully if step size is
        less than ``xk * xrtol`` where ``xk`` is the current parameter vector.
    c1 : float, default: 1e-4
        Parameter for Armijo condition rule.
    c2 : float, default: 0.9
        Parameter for curvature condition rule.
    hess_inv0 : None or ndarray, optional
        Initial inverse hessian estimate, shape (n, n). If None (default) then
        the identity matrix is used.
    """
    retall = return_all
    
    assert x0.ndim == 1

    if maxiter is None:
        maxiter = len(x0) * 200

    f = fun
    myfprime = lambda x: finite_difference_gradient(f, x, finite_difference_stepsize=eps)

    old_fval = f(x0)
    gfk = myfprime(x0)

    k = 0
    N = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I if hess_inv0 is None else hess_inv0

    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2


    xk = x0
    if retall:
        allvecs = [x0]
    warnflag = 0
    gnorm = vecnorm(gfk, ord=norm)
    while (gnorm > gtol) and (k < maxiter):
        pk = -np.dot(Hk, gfk)
        try:
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     line_search_wolfe1(f, myfprime, xk, pk, gfk,
                                          old_fval, old_old_fval, amin=1e-100,
                                          amax=1e100, c1=c1, c2=c2)
            # if k == 0:
            #     print(xk, pk, gfk, old_fval, old_old_fval)
            #     print(alpha_k, fc, gc, old_fval, old_old_fval, gfkp1)
        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break

        sk = alpha_k * pk
        xkp1 = xk + sk

        if retall:
            allvecs.append(xkp1)
        xk = xkp1
        if gfkp1 is None:
            gfkp1 = myfprime(xkp1)

        yk = gfkp1 - gfk
        gfk = gfkp1
        k += 1
        intermediate_result = dict(x=xk, fun=old_fval)

        gnorm = vecnorm(gfk, ord=norm)
        if (gnorm <= gtol):
            break

        #  See Chapter 5 in  P.E. Frandsen, K. Jonasson, H.B. Nielsen,
        #  O. Tingleff: "Unconstrained Optimization", IMM, DTU.  1999.
        #  These notes are available here:
        #  http://www2.imm.dtu.dk/documents/ftp/publlec.html
        if (alpha_k*vecnorm(pk) <= xrtol*(xrtol + vecnorm(xk))):
            break

        if not np.isfinite(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = 2
            break

        rhok_inv = np.dot(yk, sk)
        # this was handled in numeric, let it remains for more safety
        # Cryptic comment above is preserved for posterity. Future reader:
        # consider change to condition below proposed in gh-1261/gh-17345.
        if rhok_inv == 0.:
            rhok = 1000.0
            if disp:
                msg = "Divide-by-zero encountered: rhok assumed large"
                _print_success_message_or_warn(True, msg)
        else:
            rhok = 1. / rhok_inv

        A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
        A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
        Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] *
                                                 sk[np.newaxis, :])

    fval = old_fval

    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']

    result = dict(fun=fval, jac=gfk, hess_inv=Hk,
                            status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result

def line_search_wolfe1(f, fprime, xk, pk, gfk=None,
                       old_fval=None, old_old_fval=None,
                       args=(), c1=1e-4, c2=0.9, amax=50, amin=1e-8,
                       xtol=1e-14):
    """
    As `scalar_search_wolfe1` but do a line search to direction `pk`

    Parameters
    ----------
    f : callable
        Function `f(x)`
    fprime : callable
        Gradient of `f`
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
    if gfk is None:
        gfk = fprime(xk)

    gval = [gfk]
    gc = [0]
    fc = [0]

    def phi(s):
        fc[0] += 1
        return f(xk + s*pk)

    def derphi(s):
        gval[0] = fprime(xk + s*pk)
        gc[0] += 1
        return np.dot(gval[0], pk)

    derphi0 = np.dot(gfk, pk)

    stp, fval, old_fval = scalar_search_wolfe1(
            phi, derphi, old_fval, old_old_fval, derphi0,
            c1=c1, c2=c2, amax=amax, amin=amin, xtol=xtol)

    return stp, fc[0], gc[0], fval, old_fval, gval[0]


def scalar_search_wolfe1(phi, derphi, phi0, old_phi0, derphi0,
                         c1=1e-4, c2=0.9,
                         amax=50, amin=1e-8, xtol=1e-14):
    """
    Scalar function search for alpha that satisfies strong Wolfe conditions

    alpha > 0 is assumed to be a descent direction.

    Parameters
    ----------
    phi : callable phi(alpha)
        Function at point `alpha`
    derphi : callable phi'(alpha)
        Objective function derivative. Returns a scalar.
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
    assert (c1 > 0) & (c1 < c2) & (c2 < 1)

    if old_phi0 is not None and derphi0 != 0:
        alpha1 = min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)
        if alpha1 < 0:
            alpha1 = 1.0
    else:
        alpha1 = 1.0

    maxiter = 100


    dcsrch = DCSRCH(phi, derphi, c1, c2, xtol, amin, amax)

    stp, phi1, phi0, task = dcsrch(
        alpha1, phi0=phi0, derphi0=derphi0, maxiter=maxiter
    )

    return stp, phi1, phi0

class DCSRCH:
    """
    Parameters
    ----------
    phi : callable phi(alpha)
        Function at point `alpha`
    derphi : callable phi'(alpha)
        Objective function derivative. Returns a scalar.
    ftol : float
        A nonnegative tolerance for the sufficient decrease condition.
    gtol : float
        A nonnegative tolerance for the curvature condition.
    xtol : float
        A nonnegative relative tolerance for an acceptable step. The
        subroutine exits with a warning if the relative difference between
        sty and stx is less than xtol.
    stpmin : float
        A nonnegative lower bound for the step.
    stpmax :
        A nonnegative upper bound for the step.

    Notes
    -----

    This subroutine finds a step that satisfies a sufficient
    decrease condition and a curvature condition.

    Each call of the subroutine updates an interval with
    endpoints stx and sty. The interval is initially chosen
    so that it contains a minimizer of the modified function

           psi(stp) = f(stp) - f(0) - ftol*stp*f'(0).

    If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
    interval is chosen so that it contains a minimizer of f.

    The algorithm is designed to find a step that satisfies
    the sufficient decrease condition

           f(stp) <= f(0) + ftol*stp*f'(0),

    and the curvature condition

           abs(f'(stp)) <= gtol*abs(f'(0)).

    If ftol is less than gtol and if, for example, the function
    is bounded below, then there is always a step which satisfies
    both conditions.

    If no step can be found that satisfies both conditions, then
    the algorithm stops with a warning. In this case stp only
    satisfies the sufficient decrease condition.

    A typical invocation of dcsrch has the following outline:

    Evaluate the function at stp = 0.0d0; store in f.
    Evaluate the gradient at stp = 0.0d0; store in g.
    Choose a starting step stp.

    task = 'START'
    10 continue
        call dcsrch(stp,f,g,ftol,gtol,xtol,task,stpmin,stpmax,
                   isave,dsave)
        if (task .eq. 'FG') then
           Evaluate the function and the gradient at stp
           go to 10
           end if

    NOTE: The user must not alter work arrays between calls.

    The subroutine statement is

        subroutine dcsrch(f,g,stp,ftol,gtol,xtol,stpmin,stpmax,
                         task,isave,dsave)
        where

    stp is a double precision variable.
        On entry stp is the current estimate of a satisfactory
            step. On initial entry, a positive initial estimate
            must be provided.
        On exit stp is the current estimate of a satisfactory step
            if task = 'FG'. If task = 'CONV' then stp satisfies
            the sufficient decrease and curvature condition.

    f is a double precision variable.
        On initial entry f is the value of the function at 0.
        On subsequent entries f is the value of the
            function at stp.
        On exit f is the value of the function at stp.

    g is a double precision variable.
        On initial entry g is the derivative of the function at 0.
        On subsequent entries g is the derivative of the
           function at stp.
        On exit g is the derivative of the function at stp.

    ftol is a double precision variable.
        On entry ftol specifies a nonnegative tolerance for the
           sufficient decrease condition.
        On exit ftol is unchanged.

    gtol is a double precision variable.
        On entry gtol specifies a nonnegative tolerance for the
           curvature condition.
        On exit gtol is unchanged.

    xtol is a double precision variable.
        On entry xtol specifies a nonnegative relative tolerance
          for an acceptable step. The subroutine exits with a
          warning if the relative difference between sty and stx
          is less than xtol.

        On exit xtol is unchanged.

    task is a character variable of length at least 60.
        On initial entry task must be set to 'START'.
        On exit task indicates the required action:

           If task(1:2) = 'FG' then evaluate the function and
           derivative at stp and call dcsrch again.

           If task(1:4) = 'CONV' then the search is successful.

           If task(1:4) = 'WARN' then the subroutine is not able
           to satisfy the convergence conditions. The exit value of
           stp contains the best point found during the search.

          If task(1:5) = 'ERROR' then there is an error in the
          input arguments.

        On exit with convergence, a warning or an error, the
           variable task contains additional information.

    stpmin is a double precision variable.
        On entry stpmin is a nonnegative lower bound for the step.
        On exit stpmin is unchanged.

    stpmax is a double precision variable.
        On entry stpmax is a nonnegative upper bound for the step.
        On exit stpmax is unchanged.

    isave is an integer work array of dimension 2.

    dsave is a double precision work array of dimension 13.

    Subprograms called

      MINPACK-2 ... dcstep
    MINPACK-1 Project. June 1983.
    Argonne National Laboratory.
    Jorge J. More' and David J. Thuente.

    MINPACK-2 Project. November 1993.
    Argonne National Laboratory and University of Minnesota.
    Brett M. Averick, Richard G. Carter, and Jorge J. More'.
    """

    def __init__(self, phi, derphi, ftol, gtol, xtol, stpmin, stpmax):
        self.stage = None
        self.ginit = None
        self.gtest = None
        self.gx = None
        self.gy = None
        self.finit = None
        self.fx = None
        self.fy = None
        self.stx = None
        self.sty = None
        self.stmin = None
        self.stmax = None
        self.width = None
        self.width1 = None

        # leave all assessment of tolerances/limits to the first call of
        # this object
        self.ftol = ftol
        self.gtol = gtol
        self.xtol = xtol
        self.stpmin = stpmin
        self.stpmax = stpmax

        self.phi = phi
        self.derphi = derphi

    def __call__(self, alpha1, phi0=None, derphi0=None, maxiter=100):
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
        if phi0 is None:
            phi0 = self.phi(0.0)
        if derphi0 is None:
            derphi0 = self.derphi(0.0)

        phi1 = phi0
        derphi1 = derphi0

        task = b"START"
        for i in range(maxiter):
            stp, phi1, derphi1, task = self._iterate(
                alpha1, phi1, derphi1, task
            )

            if not np.isfinite(stp):
                task = b"WARN"
                stp = None
                break

            if task[:2] == b"FG":
                alpha1 = stp
                phi1 = self.phi(stp)
                derphi1 = self.derphi(stp)
            else:
                break
        else:
            # maxiter reached, the line search did not converge
            stp = None
            task = b"WARNING: dcsrch did not converge within max iterations"

        if task[:5] == b"ERROR" or task[:4] == b"WARN":
            stp = None  # failed

        return stp, phi1, phi0, task

    def _iterate(self, stp, f, g, task):
        """
        Parameters
        ----------
        stp : float
            The current estimate of a satisfactory step. On initial entry, a
            positive initial estimate must be provided.
        f : float
            On first call f is the value of the function at 0. On subsequent
            entries f should be the value of the function at stp.
        g : float
            On initial entry g is the derivative of the function at 0. On
            subsequent entries g is the derivative of the function at stp.
        task : bytes
            On initial entry task must be set to 'START'.

        On exit with convergence, a warning or an error, the
           variable task contains additional information.


        Returns
        -------
        stp, f, g, task: tuple

            stp : float
                the current estimate of a satisfactory step if task = 'FG'. If
                task = 'CONV' then stp satisfies the sufficient decrease and
                curvature condition.
            f : float
                the value of the function at stp.
            g : float
                the derivative of the function at stp.
            task : bytes
                On exit task indicates the required action:

               If task(1:2) == b'FG' then evaluate the function and
               derivative at stp and call dcsrch again.

               If task(1:4) == b'CONV' then the search is successful.

               If task(1:4) == b'WARN' then the subroutine is not able
               to satisfy the convergence conditions. The exit value of
               stp contains the best point found during the search.

              If task(1:5) == b'ERROR' then there is an error in the
              input arguments.
        """
        p5 = 0.5
        p66 = 0.66
        xtrapl = 1.1
        xtrapu = 4.0

        if task[:5] == b"START":
            if stp < self.stpmin:
                task = b"ERROR: STP .LT. STPMIN"
            if stp > self.stpmax:
                task = b"ERROR: STP .GT. STPMAX"
            if g >= 0:
                task = b"ERROR: INITIAL G .GE. ZERO"
            if self.ftol < 0:
                task = b"ERROR: FTOL .LT. ZERO"
            if self.gtol < 0:
                task = b"ERROR: GTOL .LT. ZERO"
            if self.xtol < 0:
                task = b"ERROR: XTOL .LT. ZERO"
            if self.stpmin < 0:
                task = b"ERROR: STPMIN .LT. ZERO"
            if self.stpmax < self.stpmin:
                task = b"ERROR: STPMAX .LT. STPMIN"

            if task[:5] == b"ERROR":
                return stp, f, g, task

            # Initialize local variables.

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
            task = b"FG"
            return stp, f, g, task

        # in the original Fortran this was a location to restore variables
        # we don't need to do that because they're attributes.

        # If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
        # algorithm enters the second stage.
        ftest = self.finit + stp * self.gtest

        if self.stage == 1 and f <= ftest and g >= 0:
            self.stage = 2

        # test for warnings
        if self.brackt and (stp <= self.stmin or stp >= self.stmax):
            task = b"WARNING: ROUNDING ERRORS PREVENT PROGRESS"
        if self.brackt and self.stmax - self.stmin <= self.xtol * self.stmax:
            task = b"WARNING: XTOL TEST SATISFIED"
        if stp == self.stpmax and f <= ftest and g <= self.gtest:
            task = b"WARNING: STP = STPMAX"
        if stp == self.stpmin and (f > ftest or g >= self.gtest):
            task = b"WARNING: STP = STPMIN"

        # test for convergence
        if f <= ftest and abs(g) <= self.gtol * -self.ginit:
            task = b"CONVERGENCE"

        # test for termination
        if task[:4] == b"WARN" or task[:4] == b"CONV":
            return stp, f, g, task

        # A modified function is used to predict the step during the
        # first stage if a lower function value has been obtained but
        # the decrease is not sufficient.
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
            with np.errstate(invalid="ignore", over="ignore"):
                tup = dcstep(
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
                self.stx, fxm, gxm, self.sty, fym, gym, stp, self.brackt = tup

            # Reset the function and derivative values for f
            self.fx = fxm + self.stx * self.gtest
            self.fy = fym + self.sty * self.gtest
            self.gx = gxm + self.gtest
            self.gy = gym + self.gtest

        else:
            # Call dcstep to update stx, sty, and to compute the new step.
            # dcstep can have several operations which can produce NaN
            # e.g. inf/inf. Filter these out.

            with np.errstate(invalid="ignore", over="ignore"):
                tup = dcstep(
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
            (
                self.stx,
                self.fx,
                self.gx,
                self.sty,
                self.fy,
                self.gy,
                stp,
                self.brackt,
            ) = tup

        # Decide if a bisection step is needed
        if self.brackt:
            if abs(self.sty - self.stx) >= p66 * self.width1:
                stp = self.stx + p5 * (self.sty - self.stx)
            self.width1 = self.width
            self.width = abs(self.sty - self.stx)

        # Set the minimum and maximum steps allowed for stp.
        if self.brackt:
            self.stmin = min(self.stx, self.sty)
            self.stmax = max(self.stx, self.sty)
        else:
            self.stmin = stp + xtrapl * (stp - self.stx)
            self.stmax = stp + xtrapu * (stp - self.stx)

        # Force the step to be within the bounds stpmax and stpmin.
        stp = np.clip(stp, self.stpmin, self.stpmax)

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
        task = b"FG"
        return stp, f, g, task
    

def dcstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax):
    """
    Subroutine dcstep

    This subroutine computes a safeguarded step for a search
    procedure and updates an interval that contains a step that
    satisfies a sufficient decrease and a curvature condition.

    The parameter stx contains the step with the least function
    value. If brackt is set to .true. then a minimizer has
    been bracketed in an interval with endpoints stx and sty.
    The parameter stp contains the current step.
    The subroutine assumes that if brackt is set to .true. then

        min(stx,sty) < stp < max(stx,sty),

    and that the derivative at stx is negative in the direction
    of the step.

    The subroutine statement is

      subroutine dcstep(stx,fx,dx,sty,fy,dy,stp,fp,dp,brackt,
                        stpmin,stpmax)

    where

    stx is a double precision variable.
        On entry stx is the best step obtained so far and is an
          endpoint of the interval that contains the minimizer.
        On exit stx is the updated best step.

    fx is a double precision variable.
        On entry fx is the function at stx.
        On exit fx is the function at stx.

    dx is a double precision variable.
        On entry dx is the derivative of the function at
          stx. The derivative must be negative in the direction of
          the step, that is, dx and stp - stx must have opposite
          signs.
        On exit dx is the derivative of the function at stx.

    sty is a double precision variable.
        On entry sty is the second endpoint of the interval that
          contains the minimizer.
        On exit sty is the updated endpoint of the interval that
          contains the minimizer.

    fy is a double precision variable.
        On entry fy is the function at sty.
        On exit fy is the function at sty.

    dy is a double precision variable.
        On entry dy is the derivative of the function at sty.
        On exit dy is the derivative of the function at the exit sty.

    stp is a double precision variable.
        On entry stp is the current step. If brackt is set to .true.
          then on input stp must be between stx and sty.
        On exit stp is a new trial step.

    fp is a double precision variable.
        On entry fp is the function at stp
        On exit fp is unchanged.

    dp is a double precision variable.
        On entry dp is the derivative of the function at stp.
        On exit dp is unchanged.

    brackt is an logical variable.
        On entry brackt specifies if a minimizer has been bracketed.
            Initially brackt must be set to .false.
        On exit brackt specifies if a minimizer has been bracketed.
            When a minimizer is bracketed brackt is set to .true.

    stpmin is a double precision variable.
        On entry stpmin is a lower bound for the step.
        On exit stpmin is unchanged.

    stpmax is a double precision variable.
        On entry stpmax is an upper bound for the step.
        On exit stpmax is unchanged.

    MINPACK-1 Project. June 1983
    Argonne National Laboratory.
    Jorge J. More' and David J. Thuente.

    MINPACK-2 Project. November 1993.
    Argonne National Laboratory and University of Minnesota.
    Brett M. Averick and Jorge J. More'.

    """
    sgn_dp = np.sign(dp)
    sgn_dx = np.sign(dx)

    # sgnd = dp * (dx / abs(dx))
    sgnd = sgn_dp * sgn_dx

    # First case: A higher function value. The minimum is bracketed.
    # If the cubic step is closer to stx than the quadratic step, the
    # cubic step is taken, otherwise the average of the cubic and
    # quadratic steps is taken.
    if fp > fx:
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s * np.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
        if stp < stx:
            gamma *= -1
        p = (gamma - dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = p / q
        stpc = stx + r * (stp - stx)
        stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (stp - stx)
        if abs(stpc - stx) <= abs(stpq - stx):
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
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s * np.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
        if stp > stx:
            gamma *= -1
        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dx
        r = p / q
        stpc = stp + r * (stx - stp)
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        if abs(stpc - stp) > abs(stpq - stp):
            stpf = stpc
        else:
            stpf = stpq
        brackt = True
    elif abs(dp) < abs(dx):
        # Third case: A lower function value, derivatives of the same sign,
        # and the magnitude of the derivative decreases.

        # The cubic step is computed only if the cubic tends to infinity
        # in the direction of the step or if the minimum of the cubic
        # is beyond stp. Otherwise the cubic step is defined to be the
        # secant step.
        theta = 3 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))

        # The case gamma = 0 only arises if the cubic does not tend
        # to infinity in the direction of the step.
        gamma = s * np.sqrt(max(0, (theta / s) ** 2 - (dx / s) * (dp / s)))
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = (gamma + (dx - dp)) + gamma
        r = p / q
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
            if abs(stpc - stp) < abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq

            if stp > stx:
                stpf = min(stp + 0.66 * (sty - stp), stpf)
            else:
                stpf = max(stp + 0.66 * (sty - stp), stpf)
        else:
            # A minimizer has not been bracketed. If the cubic step is
            # farther from stp than the secant step, the cubic step is
            # taken, otherwise the secant step is taken.
            if abs(stpc - stp) > abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            stpf = np.clip(stpf, stpmin, stpmax)

    else:
        # Fourth case: A lower function value, derivatives of the same sign,
        # and the magnitude of the derivative does not decrease. If the
        # minimum is not bracketed, the step is either stpmin or stpmax,
        # otherwise the cubic step is taken.
        if brackt:
            theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp
            s = max(abs(theta), abs(dy), abs(dp))
            gamma = s * np.sqrt((theta / s) ** 2 - (dy / s) * (dp / s))
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
    if fp > fx:
        sty = stp
        fy = fp
        dy = dp
    else:
        if sgnd < 0:
            sty = stx
            fy = fx
            dy = dx
        stx = stp
        fx = fp
        dx = dp

    # Compute the new step.
    stp = stpf

    return stx, fx, dx, sty, fy, dy, stp, brackt
