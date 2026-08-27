#include "bs_pricer.hpp"

#include <algorithm>
#include <cmath>

namespace qengine {

double norm_cdf(double x) { return 0.5 * std::erfc(-x * M_SQRT1_2); }

double norm_pdf(double x) { return std::exp(-0.5 * x * x) * 0.3989422804014327; /* 1/sqrt(2*pi) */ }

namespace {

// d1, d2 of the Black-Scholes-Merton formula. sigma and T are assumed > 0
// by the caller (guarded in bs_price/bs_greeks below).
inline void d1_d2(double S, double K, double T, double r, double q, double sigma, double& d1,
                   double& d2) {
    const double sqrtT = std::sqrt(T);
    d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    d2 = d1 - sigma * sqrtT;
}

}  // namespace

double bs_price(double S, double K, double T, double r, double q, double sigma, bool is_call) {
    if (T <= 0.0 || sigma <= 0.0) {
        // At expiry (or degenerate zero vol) the option is worth intrinsic value.
        const double fwd_diff = S * std::exp(-q * std::max(T, 0.0)) - K * std::exp(-r * std::max(T, 0.0));
        return is_call ? std::max(fwd_diff, 0.0) : std::max(-fwd_diff, 0.0);
    }
    double d1, d2;
    d1_d2(S, K, T, r, q, sigma, d1, d2);
    const double disc_q = std::exp(-q * T);
    const double disc_r = std::exp(-r * T);
    if (is_call) {
        return S * disc_q * norm_cdf(d1) - K * disc_r * norm_cdf(d2);
    }
    return K * disc_r * norm_cdf(-d2) - S * disc_q * norm_cdf(-d1);
}

Greeks bs_greeks(double S, double K, double T, double r, double q, double sigma, bool is_call) {
    Greeks g{0, 0, 0, 0, 0};
    if (T <= 0.0 || sigma <= 0.0) {
        // Degenerate: Greeks of a frozen intrinsic-value payoff.
        const double fwd_diff = S * std::exp(-q * std::max(T, 0.0)) - K * std::exp(-r * std::max(T, 0.0));
        const bool itm = is_call ? (fwd_diff > 0.0) : (fwd_diff < 0.0);
        g.delta = itm ? (is_call ? 1.0 : -1.0) : 0.0;
        return g;
    }

    double d1, d2;
    d1_d2(S, K, T, r, q, sigma, d1, d2);
    const double sqrtT = std::sqrt(T);
    const double disc_q = std::exp(-q * T);
    const double disc_r = std::exp(-r * T);
    const double pdf_d1 = norm_pdf(d1);

    g.gamma = disc_q * pdf_d1 / (S * sigma * sqrtT);
    g.vega = S * disc_q * pdf_d1 * sqrtT;  // per 1.00 change in sigma

    if (is_call) {
        g.delta = disc_q * norm_cdf(d1);
        const double theta_annual = -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrtT) -
                                     r * K * disc_r * norm_cdf(d2) + q * S * disc_q * norm_cdf(d1);
        g.theta = theta_annual / 365.0;
        g.rho = K * T * disc_r * norm_cdf(d2) / 100.0;
    } else {
        g.delta = disc_q * (norm_cdf(d1) - 1.0);
        const double theta_annual = -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrtT) +
                                     r * K * disc_r * norm_cdf(-d2) - q * S * disc_q * norm_cdf(-d1);
        g.theta = theta_annual / 365.0;
        g.rho = -K * T * disc_r * norm_cdf(-d2) / 100.0;
    }
    return g;
}

IVResult implied_vol(double price, double S, double K, double T, double r, double q, bool is_call,
                      double initial_guess, double tol, int max_iter, double vega_floor, double lo,
                      double hi) {
    // Reject prices outside no-arbitrage bounds up front (avoids chasing an
    // unreachable target with either Newton or bisection).
    const double intrinsic = bs_price(S, K, T, r, q, 1e-6, is_call);
    const double upper_bound = is_call ? S * std::exp(-q * T) : K * std::exp(-r * T);
    if (price < intrinsic - 1e-10 || price > upper_bound + 1e-10 || T <= 0.0) {
        return IVResult{0.0, 0, false, false};
    }

    double sigma = std::clamp(initial_guess, lo, hi);
    bool used_fallback = false;

    for (int i = 1; i <= max_iter; ++i) {
        const double model_price = bs_price(S, K, T, r, q, sigma, is_call);
        const double diff = model_price - price;
        if (std::fabs(diff) < tol) {
            return IVResult{sigma, i, true, used_fallback};
        }

        const Greeks g = bs_greeks(S, K, T, r, q, sigma, is_call);
        if (g.vega < vega_floor) {
            // Newton step is numerically unsafe here; hand off to bisection
            // for the remaining iteration budget.
            used_fallback = true;
            double blo = lo, bhi = hi;
            double f_lo = bs_price(S, K, T, r, q, blo, is_call) - price;
            for (int j = i; j <= max_iter; ++j) {
                const double mid = 0.5 * (blo + bhi);
                const double f_mid = bs_price(S, K, T, r, q, mid, is_call) - price;
                if (std::fabs(f_mid) < tol) {
                    return IVResult{mid, j, true, used_fallback};
                }
                if ((f_lo < 0.0) == (f_mid < 0.0)) {
                    blo = mid;
                    f_lo = f_mid;
                } else {
                    bhi = mid;
                }
            }
            return IVResult{0.5 * (blo + bhi), max_iter, false, used_fallback};
        }

        double next_sigma = sigma - diff / g.vega;
        next_sigma = std::clamp(next_sigma, lo, hi);
        sigma = next_sigma;
    }
    return IVResult{sigma, max_iter, false, used_fallback};
}

}  // namespace qengine
