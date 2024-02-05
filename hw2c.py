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
    # 4x4 A Matrix:
    A = [[1, -10, 2, 4],
         [3, 1, 4, 12],
         [9, 2, 3, 4],
         [-1, 2, 7, 3]]
    # 1x4 b matrix
    b = [2, 12, 21, 37]
    # 3x3 C matrix
    C = [[3, 1, -1],
         [1, 4, 1],
         [2, 1, 2]]
    # 1x3 d matrix
    d = [2, 12, 10]
    # creates the augmented matrices
    Aaug = [row + [bi] for row, bi in zip(A, b)]
    Caug = [row + [dj] for row, dj in zip(C, d)]
    # creates the initial guess matrix for gauss seidel
    x_initial_guess = [0.0] * len(b)
    x_initial_guess2 = [0.0] * len(d)
    # Make the matrix diagonally dominant
    Aaug = MakeDiagDom(Aaug)
    Caug = MakeDiagDom(Caug)
    # Solve using Gauss-Seidel
    solution = GaussSeidel(Aaug, x_initial_guess)
    solution2 = GaussSeidel(Caug, x_initial_guess2)
    # Print output to Console
    for i in range(len(solution2)):
        solution2[i] = round(solution2[i])
    for j in range(len(solution)):
        solution[j] = round(solution[j])

    print("Solution to 3x3 matrix: ", solution2)
    print("Solution to 4x4 matrix: ", solution)

if __name__ == "__main__":
    main()
