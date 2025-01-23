import numpy as np
from task3 import SturmLiouvilleProblem, solve_and_plot


class LegendreEquation(SturmLiouvilleProblem):
    """
    Legendre equation: ((1-x²)y')' + λy = 0
    """

    def __init__(self):
        self._x_start = -0.999  # Avoid exact singularity
        self._x_end = 0.999

    def p(self, x):
        return 1 - x ** 2

    def q(self, x):
        return np.zeros_like(x)

    def w(self, x):
        return np.ones_like(x)

    @property
    def domain(self):
        return (self._x_start, self._x_end)


class BesselEquation(SturmLiouvilleProblem):
    """
    Modified Bessel equation with better scaling
    """

    def __init__(self):
        self._x_start = 0.1
        self._x_end = 1.0

    def p(self, x):
        return x

    def q(self, x):
        return np.zeros_like(x)

    def w(self, x):
        # Modified weight function for better eigenvalue distribution
        return x / 10.0  # Scaling factor to reduce eigenvalue magnitudes

    @property
    def domain(self):
        return (self._x_start, self._x_end)


class QuantumHarmonicOscillator(SturmLiouvilleProblem):
    """
    Quantum harmonic oscillator with scaled potential
    """

    def __init__(self):
        self._x_start = -4.0
        self._x_end = 4.0

    def p(self, x):
        return np.ones_like(x)

    def q(self, x):
        # Scale the potential to get eigenvalues in a reasonable range
        return 0.1 * x ** 2

    def w(self, x):
        return np.ones_like(x)

    @property
    def domain(self):
        return (self._x_start, self._x_end)


class MatthieuEquation(SturmLiouvilleProblem):
    """
    Modified Mathieu equation with smaller periodic term
    """

    def __init__(self, q=0.5):  # Reduced q parameter
        self.q = q
        self._x_start = 0.0
        self._x_end = np.pi

    def p(self, x):
        return np.ones_like(x)

    def q(self, x):
        # Reduced coefficient for periodic term
        return -self.q * np.cos(2 * x)

    def w(self, x):
        return np.ones_like(x)

    @property
    def domain(self):
        return (self._x_start, self._x_end)


class LameEquation(SturmLiouvilleProblem):
    """
    Modified Lamé equation with scaled parameters
    """

    def __init__(self, k=0.3, n=1):  # Reduced k parameter
        self.k = k
        self.n = n
        self._x_start = -0.999
        self._x_end = 0.999

    def p(self, x):
        return (1 - x ** 2) * (1 - self.k ** 2 * x ** 2)

    def q(self, x):
        # Scaled coefficient
        return -0.5 * self.n * (self.n + 1) * x ** 2

    def w(self, x):
        return np.ones_like(x)

    @property
    def domain(self):
        return (self._x_start, self._x_end)


def test_all_problems(save_dir='results'):
    """
    Test all implemented problems with improved parameters
    """
    problems = [
        ("Legendre", LegendreEquation()),
        ("Bessel", BesselEquation()),
        ("QuantumHO", QuantumHarmonicOscillator()),
        ("Mathieu", MatthieuEquation()),
        ("Lame", LameEquation())
    ]

    results = {}
    for name, problem in problems:
        print(f"\nSolving {name} equation...")
        try:
            eigenvalues, eigenfunctions, x_values = solve_and_plot(
                problem,
                n_eigenvalues=8,  # Explicitly request 8 eigenvalues
                max_search=1000,  # Increased search range
                save_dir=f"{save_dir}/{name}"
            )
            results[name] = {
                'eigenvalues': eigenvalues,
                'eigenfunctions': eigenfunctions,
                'x_values': x_values
            }
            print(f"{name} equation solved successfully")
            print(f"Found {len(eigenvalues)} eigenvalues: {eigenvalues}")
        except Exception as e:
            print(f"Error solving {name} equation: {str(e)}")

    return results


if __name__ == "__main__":
    # Test all problems with improved parameters
    results = test_all_problems()

    # Test individual problems
    print("\nTesting individual problems with increased max_search...")

    # Example of individual problem testing
    qho = QuantumHarmonicOscillator()
    solve_and_plot(qho, n_eigenvalues=8, max_search=1000, save_dir='results/qho_individual')

    mathieu = MatthieuEquation(q=0.5)  # Reduced q parameter
    solve_and_plot(mathieu, n_eigenvalues=8, max_search=1000, save_dir='results/mathieu_individual')