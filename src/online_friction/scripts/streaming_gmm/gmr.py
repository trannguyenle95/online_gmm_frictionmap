import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import chi2, norm
from .utils import check_random_state
from .mvn import MVN, invert_indices, regression_coefficients

def predict(indices, X, means, covariances, priors,random_state):
        """Predict means of posteriors.

        Same as condition() but for multiple samples.

        Parameters
        ----------
        indices : array-like, shape (n_features_in,)
            Indices of dimensions that we want to condition.

        X : array-like, shape (n_samples, n_features_in)
            Values of the features that we know.

        Returns
        -------
        Y : array, shape (n_samples, n_features_out)
            Predicted means of missing values.
        """
        indices = np.asarray(indices, dtype=int)
        X = np.asarray(X)
        n_components = np.shape(priors)[0]

        n_samples = len(X)
        output_indices = invert_indices(means.shape[1], indices)
        regression_coeffs = np.empty((n_components, len(output_indices), len(indices)))

        marginal_norm_factors = np.empty(n_components)
        marginal_exponents = np.empty((n_samples,n_components))

        for k in range(n_components):
            regression_coeffs[k] = regression_coefficients(
                covariances[k], output_indices, indices)
            mvn = MVN(mean=means[k], covariance=covariances[k],
                      random_state=random_state)
            marginal_norm_factors[k], marginal_exponents[:, k] = \
                mvn.marginalize(indices).to_norm_factor_and_exponents(X)

        # posterior_means = mean_y + cov_xx^-1 * cov_xy * (x - mean_x)
        posterior_means = (
                means[:, output_indices][:, :, np.newaxis].T +
                np.einsum(
                    "ijk,lik->lji",
                    regression_coeffs,
                    X[:, np.newaxis] - means[:, indices]))

        new_priors = _safe_probability_density(
            priors * marginal_norm_factors, marginal_exponents)
        new_priors = new_priors.reshape(n_samples, 1,n_components)
        return np.sum(new_priors * posterior_means, axis=-1)


def _safe_probability_density(norm_factors, exponents):
    """Compute numerically safe probability densities of a GMM.

    The probability density of individual Gaussians in a GMM can be computed
    from a formula of the form
    q_k(X=x) = p_k(X=x) / sum_l p_l(X=x)
    where p_k(X=x) = c_k * exp(exponent_k) so that
    q_k(X=x) = c_k * exp(exponent_k) / sum_l c_l * exp(exponent_l)
    Instead of using computing this directly, we implement it in a numerically
    more stable version that works better for very small or large exponents
    that would otherwise lead to NaN or division by 0.
    The following expression is mathematically equal for any constant m:
    q_k(X=x) = c_k * exp(exponent_k - m) / sum_l c_l * exp(exponent_l - m),
    where we set m = max_l exponents_l.

    Parameters
    ----------
    norm_factors : array, shape (n_components,)
        Normalization factors of individual Gaussians

    exponents : array, shape (n_samples, n_components)
        Exponents of each combination of Gaussian and sample

    Returns
    -------
    p : array, shape (n_samples, n_components)
        Probability density of each sample
    """
    m = np.max(exponents, axis=1)[:, np.newaxis]
    p = norm_factors[np.newaxis] * np.exp(exponents - m)
    p /= np.sum(p, axis=1)[:, np.newaxis]
    return p