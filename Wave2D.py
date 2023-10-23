import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')

class Wave2D:
    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.x_axis = np.linspace(0, 1, N + 1)
        self.y_axis = np.linspace(0, 1, N + 1)
        self.h = 1/N
        self.xji, self.yij = np.meshgrid(self.x_axis, self.y_axis, indexing="ij")

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N + 1, N + 1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        dispersion_coeff = self.c * np.pi * np.sqrt(self.mx**2 + self.my**2)
        return dispersion_coeff
    
    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        c = self.c
        cfl = self.cfl
        dt = self.dt
        D = self.D
        
        Unp1, Un, Unm1 = np.zeros((3, N + 1, N + 1))
        u_exact = sp.lambdify((x,y,t), self.ue(mx, my))

        Unm1[:] = u_exact(self.xij, self.yij, 0)
        Un[:] = Unm1 + 0.5 * (c * dt)**2 * (D @ Unm1 + Unm1 @ D.T)

        self.Unm1 = Unm1
        self.Un = Un
        self.Unp1 = Unp1

    @property
    def dt(self):
        """Return the time step"""
        return (self.cfl * self.h) / self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        u_exact_func = sp.lambdify((x,y,t), self.ue(self.mx, self.my))
        u_exact = u_exact_func(self.xij, self.yij, t0)
        l2_error_norm = np.sqrt(self.h**2 * np.sum((u - u_exact)**2))
        return l2_error_norm
        

    def apply_bcs(self):
        self.Unp1[0] = 0
        self.Unp1[-1] = 0
        self.Unp1[:, 0] = 0
        self.Unp1[:, -1] = 0

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.mx = mx
        self.my = my
        self.c = c
        self.cfl = cfl

        self.create_mesh(N)
        self.D = (1 / self.h**2) * self.D2(N)
        self.initialize(N, mx, my)

        l2_err = []
        l2_err.append(self.l2_error(self.Un, seld.dt))
        
        if store_data > 0:
            plot_data = {0: self.Unm1.copy()}
        if store_data == 1:
            plot_data[self.dt] = self.Un.copy()

        for i in range(1, Nt):
            self.Unp1[:] = 2 * self.Un - self.Unm1 + (c * self.dt)**2 * (self.D @ self.Un + self.Un @ self.D.T)
            self.apply_bcs()

            self.Unm1[:] = self.Un
            self.Un[:] = self.Unp1

            if i % store_data == 0 and store_data != -1:
                plot_data[i] = self.Unm1.copy()

            l2_err.append(self.l2_error(self.Un, (i + 1) * self.dt))
        
        if store_data == -1:
            return self.h, l2_err
        else:
            return plot_data

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):
    def D2(self, N):
        raise NotImplementedError

    def ue(self, mx, my):
        raise NotImplementedError

    def apply_bcs(self):
        raise NotImplementedError

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    raise NotImplementedError

def main():
    test_convergence_wave2d()
    test_convergence_wave2d_neumann()
    test_exact_wave2d()

if __name__ == "__main__":
    main()
