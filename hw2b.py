import math

def Secant(fcn, x0, x1, maxiter=10, xtol=1e-5):
    """
    Use the Secant Method to find the root of fcn(x), in the neighborhood of x0 and x1.

    Parameters:
    - fcn: The function for which to find the root.
    - x0, x1: Two initial guesses in the neighborhood of the root.
    - maxiter: Maximum number of iterations.
    - xtol: Tolerance for convergence.

    Returns:
    - The final estimate of the root.
    """

    # Initial values
    x_prev = x0
    x_curr = x1

    for iteration in range(maxiter):
        # Secant method formula
        f_prev = fcn(x_prev)
        f_curr = fcn(x_curr)

        if f_curr - f_prev == 0:
            raise ValueError("Secant method division by zero error.")

        x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)

        # Check for convergence
        if abs(x_next - x_curr) < xtol:
            return x_next

        # Update values for the next iteration
        x_prev = x_curr
        x_curr = x_next

    raise RuntimeError("Secant method did not converge within the specified number of iterations.")

def example_function(x):
    """Example function for demonstration purposes."""
    return x - 3*math.cos(x)

def main():
    """
    sets initial x0 and x1 and calls secant function with a callback to example_function for the function that we are
    using the secant method on
    :return: estimated root
    """
    # Initial guesses
    x0 = 1.0
    x1 = 2.0
    # Call the Secant function
    root_estimate = Secant(example_function, x0, x1)

    # Print the result
    print(f"Estimated root: {root_estimate}")

# Run the main function
if __name__ == "__main__":
    main()