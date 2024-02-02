from copy import deepcopy

def GaussSeidel(Aaug, x, Niter=15):
    """
    Implements the Gauss-Seidel method for solving a system of equations.
    :param Aaug: The augmented matrix from Ax=b -> [A|b]
    :param x: An initial guess for the x vector. if A is nxn, x is nx1
    :param Niter: Number of iterations to run the GS method
    :return: the solution vector x
    """
    n = len(x)

    for k in range(Niter):
        x_old = deepcopy(x)

        for i in range(n):
            sigma = sum(Aaug[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (Aaug[i][-1] - sigma) / Aaug[i][i]

    return x

def MakeDiagDom(Aaug):
    """
    Swaps rows to make the matrix diagonally dominant.
    :param Aaug: The matrix to modify
    :return: The modified matrix
    """
    n = len(Aaug)

    for i in range(n):
        max_row = max(range(i, n), key=lambda j: abs(Aaug[j][i]))
        Aaug[i], Aaug[max_row] = Aaug[max_row], Aaug[i]

    return Aaug

def main():
    # Example usage:
    A = [[1, -10, 2, 4],
         [3, 1, 4, 12],
         [9, 2, 3, 4],
         [-1, 2, 7, 3]]

    b = [2, 12, 21, 37]

    Aaug = [row + [bi] for row, bi in zip(A, b)]
    x_initial_guess = [0.0] * len(b)

    # Make the matrix diagonally dominant
    Aaug = MakeDiagDom(Aaug)

    # Solve using Gauss-Seidel
    solution = GaussSeidel(Aaug, x_initial_guess)

    print("Solution:", solution)

if __name__ == "__main__":
    main()
