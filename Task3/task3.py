import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import os
from datetime import datetime

class SturmLiouvilleProblem(ABC):
    @abstractmethod
    def p(self, x):
        pass

    @abstractmethod
    def q(self, x):
        pass

    @abstractmethod
    def w(self, x):
        pass

    @property
    @abstractmethod
    def domain(self):
        pass


class ExampleProblem(SturmLiouvilleProblem):
    def __init__(self, m=1):
        self.m = m
        self._x_start = 0
        self._x_end = np.pi / 2

    def p(self, x):
        return np.cos(x) ** 4

    def q(self, x):
        eps = 1e-10
        sin_x = np.sin(x) + eps
        return (self.m ** 2 * np.cos(x) ** 2) / (2 * sin_x ** 2) - np.cos(x) / sin_x

    def w(self, x):
        return np.ones_like(x)

    @property
    def domain(self):
        return (self._x_start, self._x_end)


class SturmLiouvilleSolver:
    def __init__(self, problem: SturmLiouvilleProblem, num_points=300):
        self.problem = problem
        self.num_points = num_points
        self.x_start, self.x_end = problem.domain
        self.x = np.linspace(self.x_start, self.x_end, num_points)
        self.max_search = 500

    def system(self, x, y, lambda_):
        u, v = y
        p_x = max(self.problem.p(x), 1e-10)
        du_dx = v / p_x
        dv_dx = (self.problem.q(x) - lambda_ * self.problem.w(x)) * u
        return [du_dx, dv_dx]

    def solve_ivp_lambda(self, lambda_):
        y0 = [0, self.problem.p(self.x_start)]

        try:
            sol = solve_ivp(
                fun=lambda x, y: self.system(x, y, lambda_),
                t_span=(self.x_start, self.x_end),
                y0=y0,
                method='RK45',
                max_step=0.1,
                rtol=1e-6,
                atol=1e-8
            )

            if not sol.success:
                return float('inf')

            return sol.y[0, -1]

        except Exception as e:
            return float('inf')

    def find_eigenvalue(self, lambda_min, lambda_max):
        try:
            return brentq(
                self.solve_ivp_lambda,
                lambda_min,
                lambda_max,
                rtol=1e-6,
                maxiter=50
            )
        except ValueError:
            return None

    def compute_eigenfunction(self, lambda_):
        y0 = [0, self.problem.p(self.x_start)]
        x_eval = np.linspace(self.x_start, self.x_end, self.num_points)

        sol = solve_ivp(
            fun=lambda x, y: self.system(x, y, lambda_),
            t_span=(self.x_start, self.x_end),
            y0=y0,
            t_eval=x_eval,
            method='RK45',
            max_step=0.1,
            rtol=1e-6,
            atol=1e-8
        )

        eigenfunction = sol.y[0, :]
        weight = self.problem.w(x_eval)
        norm = np.sqrt(np.trapezoid(eigenfunction ** 2 * weight, x_eval))

        if norm > 1e-10:
            eigenfunction = eigenfunction / norm

        return x_eval, eigenfunction

    def find_first_n_eigenvalues(self, n, lambda_min=0, max_search=None):
        if max_search is None:
            max_search = self.max_search

        eigenvalues = []
        current_min = lambda_min
        step = 5
        consecutive_failures = 0
        max_consecutive_failures = 3

        while len(eigenvalues) < n and current_min < max_search:
            current_max = current_min + step
            eigenvalue = self.find_eigenvalue(current_min, current_max)

            if eigenvalue is not None:
                if not eigenvalues or abs(eigenvalue - eigenvalues[-1]) > 1e-6:
                    eigenvalues.append(eigenvalue)
                    print(f"Found eigenvalue: {eigenvalue:.6f}")
                    consecutive_failures = 0

                    if len(eigenvalues) > 1:
                        gap = eigenvalues[-1] - eigenvalues[-2]
                        step = max(3, min(gap * 0.8, gap * 1.2))

                    if len(eigenvalues) == n:
                        break

                current_min = eigenvalue + 0.1 * step
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    step *= 2
                    consecutive_failures = 0
                current_min = current_max

            step = min(step, max_search * 0.2)

        if len(eigenvalues) < n:
            print(
                f"Warning: Only found {len(eigenvalues)} eigenvalues. Try increasing max_search (current: {max_search}).")

        return eigenvalues

    def plot_results(self, eigenvalues, eigenfunctions, x_values, save_dir='results'):
        # Create results directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        n = len(eigenvalues)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot eigenvalues
        ax1.plot(range(1, n + 1), eigenvalues, 'bo-')
        ax1.set_xlabel('n')
        ax1.set_ylabel('λₙ')
        ax1.set_title('Eigenvalues')
        ax1.grid(True)

        # Plot eigenfunctions
        for i, ef in enumerate(eigenfunctions):
            ax2.plot(x_values, ef, label=f'n={i + 1}')

        ax2.set_xlabel('x')
        ax2.set_ylabel('u(x)')
        ax2.set_title('Eigenfunctions')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()

        # Save combined plot
        combined_filename = os.path.join(save_dir, f'combined_plot_{timestamp}.png')
        plt.savefig(combined_filename, dpi=300, bbox_inches='tight')

        # Create and save separate plots
        # Eigenvalues plot
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, n + 1), eigenvalues, 'bo-')
        plt.xlabel('n')
        plt.ylabel('λₙ')
        plt.title('Eigenvalues')
        plt.grid(True)
        eigenvalues_filename = os.path.join(save_dir, f'eigenvalues_{timestamp}.png')
        plt.savefig(eigenvalues_filename, dpi=300, bbox_inches='tight')

        # Eigenfunctions plot
        plt.figure(figsize=(8, 6))
        for i, ef in enumerate(eigenfunctions):
            plt.plot(x_values, ef, label=f'n={i + 1}')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.title('Eigenfunctions')
        plt.grid(True)
        plt.legend()
        eigenfunctions_filename = os.path.join(save_dir, f'eigenfunctions_{timestamp}.png')
        plt.savefig(eigenfunctions_filename, dpi=300, bbox_inches='tight')

        plt.close('all')

        print(f"Plots saved in {save_dir}:")
        print(f"1. Combined plot: {combined_filename}")
        print(f"2. Eigenvalues plot: {eigenvalues_filename}")
        print(f"3. Eigenfunctions plot: {eigenfunctions_filename}")

        return fig


def solve_and_plot(problem, n_eigenvalues=8, max_search=500, save_dir='results'):
    solver = SturmLiouvilleSolver(problem)

    print(f"Finding first {n_eigenvalues} eigenvalues...")
    eigenvalues = solver.find_first_n_eigenvalues(n_eigenvalues, max_search=max_search)
    print("\nEigenvalues found:", eigenvalues)

    print("\nComputing eigenfunctions...")
    eigenfunctions = []
    x_values = None

    for lambda_ in eigenvalues:
        x, ef = solver.compute_eigenfunction(lambda_)
        eigenfunctions.append(ef)
        if x_values is None:
            x_values = x

    fig = solver.plot_results(eigenvalues, eigenfunctions, x_values, save_dir=save_dir)
    plt.show()

    return eigenvalues, eigenfunctions, x_values


if __name__ == "__main__":
    print("Solving Example Problem...")
    problem = ExampleProblem(m=1)
    eigenvalues, eigenfunctions, x_values = solve_and_plot(problem, n_eigenvalues=8, max_search=500)