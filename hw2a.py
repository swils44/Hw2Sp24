def GaussianPDF(args):
    """
    Gaussian probability density function (PDF) callback function.

    Args:
        args (tuple): Tuple containing x, μ (population mean), and σ (population standard deviation).

    Returns:
        float: The probability density at the given x in the normal distribution.
    """
    x, mean, std_dev = args
    exponent = -0.5 * ((x - mean) / std_dev) ** 2
    return (1 / (std_dev * (2 * 3.14159) ** 0.5)) * 2.71828 ** exponent


def Probability(PDF, args, c, GT=True):
    """
    Calculate the probability of x being greater than or less than c using Simpson's 1/3 rule.

    Parameters:
        PDF (callable): Callback function for the Gaussian/normal probability density function.
        args (tuple): Tuple containing μ (population mean) and σ (population standard deviation).
        c (float): The upper limit of integration.
        GT (bool): Boolean indicating if we want the probability of x being greater than c (GT=True) or less than c (GT=False).

    Returns:
        float: Probability value.
    """
    mu, sigma = args
    n = 1000  # Number of intervals for Simpson's rule
    h = (c - mu + 5 * sigma) / n  # Interval width

    result = 0
    for i in range(n + 1):
        x = mu - 5 * sigma + i * h if not GT else c + i * h
        weight = 2 if i % 2 == 0 else 4  # Weight for Simpson's rule
        result += weight * PDF((x, mu, sigma))

    result = (h / 3) * result

    return result


def main():
    """
    Demonstrate the usage of the Probability function with GaussianPDF for two examples.
    """
    # Example 1: P(x<105|N(100,12.5))
    args1 = (100, 12.5)
    c1 = 105.0
    result1 = Probability(GaussianPDF, args1, c1, GT=False)
    print(f"P(x<{c1:.2f}|N({args1[0]},{args1[1]})) = {result1:.2f}")

    # Example 2: P(x>μ+2σ|N(100, 3))
    args2 = (100, 3)
    c2 = 100 + 2 * 3
    result2 = Probability(GaussianPDF, args2, c2, GT=False)
    print(f"P(x>{c2:.2f}|N({args2[0]},{args2[1]})) = {result2:.2f}")


# Run the main function
if __name__ == "__main__":
    main()
