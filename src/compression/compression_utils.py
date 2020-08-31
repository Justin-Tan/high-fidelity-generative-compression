import torch

# TODO unit test on simple CDFs
def estimate_tails(cdf, target, shape, dtype=torch.float32, extra_counts=10):
    """
    Estimates approximate tail quantiles.
    This runs a simple Adam iteration to determine tail quantiles. The
    objective is to find an `x` such that: [[[ cdf(x) == target ]]]

    Note that `cdf` is assumed to be monotonic. When each tail estimate has passed the 
    optimal value of `x`, the algorithm does `extra_counts` (default 10) additional 
    iterations and then stops.

    This operation is vectorized. The tensor shape of `x` is given by `shape`, and
    `target` must have a shape that is broadcastable to the output of `func(x)`.

    Arguments:
    cdf: A callable that computes cumulative distribution function, survival
         function, or similar.
    target: The desired target value.
    shape: The shape of the `tf.Tensor` representing `x`.
    Returns:
    A `torch.Tensor` representing the solution (`x`).
    """
    # A bit hacky
    lr, eps = 1e-2, 1e-8
    beta_1, beta_2 = 0.9, 0.99

    # Tails should be monotonically increasing
    tails = torch.zeros(shape, dtype=dtype, requires_grad=True)
    m = torch.zeros(shape, dtype=dtype)
    v = torch.ones(shape, dtype=dtype)
    counts = torch.zeros(shape, dtype=torch.int32)

    while torch.min(counts) < extra_counts:
        loss = abs(cdf(tails) - target)
        loss.backward(torch.ones_like(tails))

        grad = tails.grad

        with torch.no_grad():
            m = beta_1 * m + (1. - beta_1) * tails.grad
            v = beta_2 * v + (1. - beta_2) * torch.square(tails.grad)
            tails -= lr * m / (torch.sqrt(v) + eps)

        # Condition assumes tails init'd at zero
        counts = torch.where(torch.logical_or(counts > 0, tails.grad * tails > 0), 
            counts+1, counts)

        tails.grad.zero_()

    return tails


if __name__ == '__main__':

    quantile = 0.42

    norm_cdf = lambda x: 0.5 * (1. + torch.erf(x / np.sqrt(2)))
    tails = estimate_tails(norm_cdf, quantile, shape=10)

    print(f"OPT: {norm_cdf(torch.ones(1)*tails[0]).item()}, TRUE {quantile}")

    quantile = 0.87

    norm_cdf = torch.sigmoid
    tails = estimate_tails(norm_cdf, quantile, shape=10)

    print(f"OPT: {norm_cdf(torch.ones(1)*tails[0]).item()}, TRUE {quantile}")