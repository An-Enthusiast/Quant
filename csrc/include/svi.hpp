#pragma once
// Raw SVI (Stochastic Volatility Inspired) smile parameterization and
// calibration, per Gatheral (2004) / Gatheral & Jacquier (2014).
//
// Total implied variance as a function of log-moneyness k = log(K/F):
//
//     w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + sigma^2) )
//
// with b >= 0, |rho| < 1, sigma > 0. Implied vol at k for maturity T is
// sqrt(w(k) / T).
#include <vector>

namespace qengine {

struct SVIParams {
    double a;
    double b;
    double rho;
    double m;
    double sigma;
};

// Total variance w(k) for a single log-moneyness point.
double svi_total_variance(const SVIParams& p, double k);

struct SVICalibrationResult {
    SVIParams params;
    double rmse;
    int iterations;
    bool converged;
};

// Fits SVI params to (k_i, w_i) pairs (log-moneyness, market total variance)
// via a hand-rolled Levenberg-Marquardt with a numerically-differenced
// Jacobian and box-constraint projection after each accepted step.
SVICalibrationResult svi_calibrate(const std::vector<double>& k, const std::vector<double>& w,
                                    SVIParams initial_guess, int max_iter = 200, double tol = 1e-10);

struct NoArbReport {
    bool butterfly_ok;       // sufficient condition for no butterfly arbitrage
    double min_density_proxy; // g(k) proxy at the vertex; should stay >= 0
};

// Static no-arbitrage sufficient conditions for a single SVI slice
// (Gatheral & Jacquier 2014, Lemma 2.2 style bounds):
//   b >= 0, |rho| < 1, sigma > 0
//   a + b * sigma * sqrt(1 - rho^2) >= 0       (min of w(k) is non-negative)
//   b * (1 + |rho|) <= 4                       (rules out steep-wing butterfly arb)
NoArbReport svi_check_butterfly(const SVIParams& p);

// Calendar-spread no-arbitrage check between two slices (near expiry T1 <
// far expiry T2): total variance must be non-decreasing in T at every k on
// the shared grid, i.e. w_far(k) >= w_near(k) for all k in k_grid.
bool svi_check_calendar(const SVIParams& near, const SVIParams& far, const std::vector<double>& k_grid);

}  // namespace qengine
