import numpy as np
import matplotlib.pyplot as plt

def compute_stream_function(u, v, omega):
    # Assuming periodic boundary conditions for simplicity
    # dx and dy are the grid spacing in the x and y directions, respectively
    
    # Assuming uniform grid spacing
    ny, nx = omega.shape
    print(f"nx = {nx}, ny = {ny}")
    dx, dy = 1.0 / (nx - 1), 1.0 / (ny - 1)

    # Compute vorticity from the velocity components
    vorticity = np.gradient(v, axis=1, edge_order=2) - np.gradient(u, axis=0, edge_order=2)

    # Ensure the vorticity is consistent with the provided vorticity (omega)
    vorticity = omega.copy() if omega is not None else vorticity

    # Create a stream function variable (initialized with zeros)
    stream_function = np.zeros((ny, nx))

    # Poisson equation: ∇^2Ψ = -ω
    # Solve for stream function Ψ using finite differences and an iterative method
    max_iterations = 100
    tolerance = 1e-6
    for _ in range(max_iterations):
        psi_old = stream_function.copy()
        print(f"interation: {_}")
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                stream_function[i, j] = 0.25 * (
                    stream_function[i + 1, j]
                    + stream_function[i - 1, j]
                    + stream_function[i, j + 1]
                    + stream_function[i, j - 1]
                    + dx * dy * vorticity[i, j]
                )

        # Check for convergence
        diff = np.linalg.norm(stream_function - psi_old)
        if diff < tolerance:
            break

    return stream_function

if __name__ == "__main__":

    # Example 2D velocity components (u, v), pressure (p), and vorticity (omega)
    nx, ny = 50, 50
    u = np.random.random((ny, nx))
    v = np.random.random((ny, nx))
    p = np.random.random((ny, nx))
    omega = np.random.random((ny, nx))

    # Compute the stream function (Ψ)
    stream_function = compute_stream_function(u, v, omega)

    # Plot the stream function
    plt.imshow(stream_function, cmap='plasma', origin='lower', extent=(0, 1, 0, 1))
    plt.title('Stream Function (Ψ)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar()
    plt.show()
