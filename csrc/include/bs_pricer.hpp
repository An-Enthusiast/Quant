#pragma once
// Black-Scholes pricing, Greeks and a safeguarded Newton-Raphson implied-vol
// solver for European options on index futures/spot (Nifty/BankNifty).
//
// Pricing convention: continuously compounded risk-free rate r and a
// continuous dividend/carry yield q (for index options priced off the
// futures/forward, set q = r so the model prices off the forward F = S).

namespace qengine {

struct Greeks {
    double delta;
    double gamma;
    double theta;  // per calendar day (annual theta / 365)
    double vega;   // per 1.00 (100%) change in vol; divide by 100 for "per vol point"
    double rho;
};

struct IVResult {
    double iv;
    int iterations;
    bool converged;
    bool used_fallback;  // true if Newton-Raphson handed off to bisection
};

// Standard normal CDF/PDF, exposed for reuse/testing.
double norm_cdf(double x);
double norm_pdf(double x);

// European option price under Black-Scholes-Merton.
double bs_price(double S, double K, double T, double r, double q, double sigma, bool is_call);

// Analytical Greeks (Delta, Gamma, Theta, Vega, Rho).
Greeks bs_greeks(double S, double K, double T, double r, double q, double sigma, bool is_call);

// Implied volatility solver.
//
// Uses Newton-Raphson:
//     sigma_{n+1} = sigma_n - (BS(sigma_n) - price) / vega(sigma_n)
//
// Safeguard: if vega collapses below `vega_floor` (deep ITM/OTM, or T close
// to 0, where the Newton step becomes numerically unstable / can diverge),
// the solver falls back to bisection on [lo, hi] for the remaining budget
// of iterations, guaranteeing convergence for any arbitrage-free price.
IVResult implied_vol(double price, double S, double K, double T, double r, double q, bool is_call,
                      double initial_guess = 0.3, double tol = 1e-8, int max_iter = 100,
                      double vega_floor = 1e-8, double lo = 1e-4, double hi = 5.0);

}  // namespace qengine
