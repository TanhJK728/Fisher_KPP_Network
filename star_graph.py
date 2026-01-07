import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class StarGraphSolver:
    """
    A numerical solver for the Fisher-KPP equation on a Star Graph (3 edges connected at a central node).
    The solver uses the Finite Difference Method (FDM) with explicit time stepping.
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
        self.D3 = 1.0
        
        # Grid dimensions
        self.Nx = int(L / dx) + 1
        self.Nt = int(T / dt) + 1
        
        # Initialize 3 edges (arrays)
        # Convention: Index 0 is the central Junction Node. Index -1 is the far boundary.
        self.u1 = np.zeros(self.Nx)
        self.u2 = np.zeros(self.Nx)
        self.u3 = np.zeros(self.Nx)
        
        # Spatial coordinate array
        self.x = np.linspace(0, L, self.Nx)

    def set_initial_condition(self):
        """
        Sets the initial population distribution.
        Scenario: An invasive population starts at the far end of Edge 1.
        """
        # Population exists only at the far end of Edge 1 (x > 8.0)
        self.u1[self.x > 8.0] = 1.0 
        
        # Edges 2 and 3 start empty
        self.u2[:] = 0.0
        self.u3[:] = 0.0

    def step_one_edge(self, u_array, D_local):
        """
        Computes the time evolution for the interior points of a single edge.
        Uses the explicit Finite Difference stencil: u_new = u + dt * (Diffusion + Reaction)
        """
        u = u_array
        # 1. Diffusion Term: D * d^2u/dx^2
        # Vectorized calculation for interior points u[1:-1]
        diffusion = D_local * (u[2:] - 2*u[1:-1] + u[:-2]) / (self.dx**2)
        
        # 2. Reaction Term: r * u * (1 - u)
        reaction = self.r * u[1:-1] * (1 - u[1:-1])
        
        # Update interior points
        u[1:-1] += self.dt * (diffusion + reaction)
        return u

    def solve_and_animate(self, filename="simulation.gif"):
        """
        Run the time-stepping loop and animate the results.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Edge 1 is plotted on the negative x-axis to visualize the 'input' flow.
        # Edges 2 and 3 are plotted on the positive x-axis.
        line1, = ax.plot(-self.x, self.u1, 'r-', label='Edge 1 (Input Branch)', lw=2)
        line2, = ax.plot(self.x, self.u2, 'b--', label='Edge 2 (Output Branch 1)')
        line3, = ax.plot(self.x, self.u3, 'g:', label='Edge 3 (Output Branch 2)')
        
        ax.set_ylim(-0.1, 1.2)
        ax.set_xlim(-self.L, self.L)
        ax.set_xlabel("Position (0 = Junction)")
        ax.set_ylabel("Population Density u(x,t)")
        ax.set_title("Fisher-KPP Dynamics on a Star Graph")
        ax.legend()
        ax.grid(True)

        total_frames = 200
        steps_per_frame = int(self.Nt / total_frames)
        
        def update(frame):
            for n in range(steps_per_frame):
                # 1. Update Interior Points for all edges
                self.u1 = self.step_one_edge(self.u1, D_local=self.D)
                self.u2 = self.step_one_edge(self.u2, D_local=self.D)
                self.u3 = self.step_one_edge(self.u3, D_local=self.D3)
                
                # 2. Apply Boundary Conditions at Far Ends (Neumann： No Flux)
                # Assumption: The population cannot leave the system from the far ends.
                self.u1[-1] = self.u1[-2] 
                self.u2[-1] = self.u2[-2]
                self.u3[-1] = self.u3[-2]
                
                # 3. Apply Junction Conditions (Kirchhoff Law)
                # Continuity: u1[0] = u2[0] = u3[0]
                # Flux Conservation: The sum of fluxes at the node must be zero.
                # In the discrete case with equal dx and D, the junction value becomes the average of its immediate neighbors.
                u_neighbors_sum = self.u1[1] + self.u2[1] + self.u3[1]
                u_junction_new = u_neighbors_sum / 3.0
                
                # Update the junction node for all edges
                self.u1[0] = u_junction_new
                self.u2[0] = u_junction_new
                self.u3[0] = u_junction_new
            
            # Update Plot every 20 frames for performance
            line1.set_ydata(self.u1)
            line2.set_ydata(self.u2)
            line3.set_ydata(self.u3)
            ax.set_title(f"Star Graph Spread: t = {frame * steps_per_frame * self.dt:.2f}s")
            return line1, line2, line3

        ani = animation.FuncAnimation(fig, update, frames=total_frames, blit=True)
        
        # Save GIF (Need pillow: pip install pillow to save gif)
        # fps=20
        ani.save(filename, writer='pillow', fps=20)
        plt.close()

if __name__ == "__main__":
    solver = StarGraphSolver(L=10.0, T=20.0, dx=0.1, dt=0.002, D = 1.0, r = 1.0)
    solver.set_initial_condition()
    solver.solve_and_animate()
