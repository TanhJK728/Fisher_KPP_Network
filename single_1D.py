import numpy as np
import matplotlib.pyplot as plt
import time

class FisherKPPSolver1D:
    """
    A numerical solver for the Fisher-KPP equation on a 1D domain (single edge).
    Equation: u_t = D * u_xx + r * u * (1 - u)
    """
    def __init__(self, L, T, dx, dt, D, r):
        """
        Initialize the simulation parameters.
        :param L: Length of the domain (edge)
        :param T: Total simulation time
        :param dx: Spatial step size
        :param dt: Time step size
        :param D: Diffusion coefficient
        :param r: Growth rate
        """
        self.L = L
        self.T = T
        self.dx = dx
        self.dt = dt
        self.D = D
        self.r = r
        
        # Determine number of grid points
        self.Nx = int(L / dx) + 1
        self.Nt = int(T / dt) + 1
        self.x = np.linspace(0, L, self.Nx)
        
        # Initialize grid (u_current and u_next)
        self.u = np.zeros(self.Nx)
        
        # Check CFL Condition for stability
        # Stability requirement: dt <= dx^2 / (2D)
        stability_limit = dx**2 / (2 * D)
        if dt > stability_limit:
            print(f"The parameters are unstable: dt ({dt}) > Limit ({stability_limit:.4f})")
            print("The simulation might explode. Please reduce dt or increase dx.")
        else:
            print(f"Stability check passed. CFL Limit: {stability_limit:.4f}")

    def set_initial_condition(self, type="step"):
        """
        Set the initial population distribution u(x, 0).
        """
        if type == "step":
            # Step function: Population only exists on the left side
            self.u[self.x < 2.0] = 1.0
        elif type == "gaussian":
            # Gaussian bump in the middle
            self.u = np.exp(-0.5 * ((self.x - self.L/2)**2) / 0.5)
        elif type == "random":
             self.u = np.random.rand(self.Nx) * 0.1

    def solve_and_animate(self):
        """
        Run the time-stepping loop and animate the results.
        """
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot(self.x, self.u, 'b-', label='Population Density u(x,t)')
        ax.set_ylim(0, 1.2)
        ax.set_xlabel("Position on Edge")
        ax.set_ylabel("Density")
        ax.set_title("Fisher-KPP Simulation (1D)")
        ax.legend()
        
        # Time integration loop
        u_new = np.zeros(self.Nx)
        
        for n in range(self.Nt):
            # 1. Diffusion Term: D * (u_{i+1} - 2u_i + u_{i-1}) / dx^2
            # Inner points: u[1:-1]
            diffusion = self.D * (self.u[2:] - 2*self.u[1:-1] + self.u[:-2]) / (self.dx**2)
            
            # 2. Reaction Term: r * u * (1 - u)
            reaction = self.r * self.u[1:-1] * (1 - self.u[1:-1])
            
            # 3. Update inner points
            u_new[1:-1] = self.u[1:-1] + self.dt * (diffusion + reaction)
            
            # 4. Boundary Conditions (Neumann: No Flux)
            # u_x = 0  => u[0] = u[1], u[-1] = u[-2]
            u_new[0] = u_new[1]
            u_new[-1] = u_new[-2]
            
            # Update state
            self.u[:] = u_new[:]
            
            # Visualization (Update every 10 steps to speed up)
            if n % 10 == 0:
                line.set_ydata(self.u)
                ax.set_title(f"Fisher-KPP Simulation: t = {n*self.dt:.2f}s")
                plt.draw()
                plt.pause(0.001)
        
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    
    # Experiment: D ranges from 1 to 0, decreasing D leads to slower spread
    solver = FisherKPPSolver1D(L=20.0, T=15.0, dx=0.1, dt=0.002, D=1.0, r=1.0)
    
    # Initialize with a step function (population starts at left) or gaussian bump in the middle; type = "step" or type = "gaussian"
    solver.set_initial_condition(type="step")
    solver.solve_and_animate()
