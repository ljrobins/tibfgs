import tibfgs
import time
import numpy as np
import scipy
import polars as pl
import re

def copy_dataframe_as_latex(df: pl.DataFrame, format: str = 'latex'):
    # remember, requires \usepackage{booktabs}
    import pyperclip

    def replace_match(match):
        content = match.group(1)  # Get the captured content (inside the first {})
        if content == 'tabular':
            return '{' + content + '}'
        return (
            '{|' + '|'.join(f'{char}' for char in content) + '|}'
        )  # Transform "xyz" to "|x|y|z|"

    if format == 'latex':
        dfp = df.to_pandas()
        latex = dfp.to_latex(index=False, multicolumn_format='c', bold_rows=True)
        pattern = re.compile('\{([^{}]*)\}')

        latex = pattern.sub(
            replace_match, latex, count=2
        )  # Only replace the first match
        pyperclip.copy(latex)
    
def rosen_md_numpy(x: np.ndarray) -> float:
    f = 0.0
    for i in range(x.size-1):
        f += 100 * (x[i+1] - x[i]**2)**2 + (1-x[i])**2
    return f

np.random.seed(1)

NPART = 1000_000
NDIM = 8
x0 = 4 * np.random.rand(NPART, NDIM) - 2

# _ = tibfgs.minimize(tibfgs.ackley, x0)
# t1 = time.time()
# res_dict = tibfgs.minimize(tibfgs.ackley, x0)
# ti_per_sec = 1 / ((time.time() - t1) / NPART)
# print(ti_per_sec)

# print(res_dict)

n_scipy = 20

methods = [
    'BFGS',
    'Powell',
    'Nelder-Mead',
    'TNC',
    'SLSQP',
    'CG',
]
results_sp = []
for method in methods:
    # run once so the solver is warm
    res = scipy.optimize.minimize(
        fun=rosen_md_numpy,
        x0=x0[0],
        method=method,
    )
    t1 = time.time()
    n_iter = []
    n_fev = []
    for i in range(n_scipy):
        sol = scipy.optimize.minimize(
            fun=rosen_md_numpy,
            x0=x0[i],
            method=method,
        )
        n_iter.append(sol['nit'])
        n_fev.append(sol['nfev'])
    results_sp.append(
        {
            'Method': method,
            # 'Solutions / sec': round(1 / ((time.time() - t1) / n_scipy)),
            'Median iterations': round(np.median(n_iter)),
            'Median $f(\mathbf{x})$ evals': round(np.median(n_fev)),
        }
    )

df = pl.DataFrame(results_sp).sort('Median $f(\mathbf{x})$ evals')
print(df)
copy_dataframe_as_latex(df)
