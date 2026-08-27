// Minimal assert-based smoke tests for the qengine C++ core, independent of
// pybind11/Python. Run via `ctest` from csrc/build, or the binary directly.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "bs_pricer.hpp"
#include "svi.hpp"

using namespace qengine;

static void test_put_call_parity() {
    const double S = 25000, K = 24800, T = 20.0 / 365.0, r = 0.065, q = 0.065, sigma = 0.14;
    const double call = bs_price(S, K, T, r, q, sigma, true);
    const double put = bs_price(S, K, T, r, q, sigma, false);
    const double lhs = call - put;
    const double rhs = S * std::exp(-q * T) - K * std::exp(-r * T);
    assert(std::fabs(lhs - rhs) < 1e-6);
}

static void test_iv_roundtrip() {
    const double S = 48000, K = 48500, T = 10.0 / 365.0, r = 0.065, q = 0.065, sigma = 0.22;
    const double price = bs_price(S, K, T, r, q, sigma, false);
    const IVResult res = implied_vol(price, S, K, T, r, q, false);
    assert(res.converged);
    assert(std::fabs(res.iv - sigma) < 1e-4);
}

static void test_iv_deep_otm_uses_fallback_or_converges() {
    // Deep-ish OTM, short-dated: vega is small (though not literally below
    // the floor for this strike/tenor). Correctness criterion is reprice
    // consistency, not exact recovery of sigma -- when a quoted price is
    // this close to zero, many sigmas reprice to the same
    // double-precision-indistinguishable price, so IV recovery is
    // inherently non-unique; what must hold is that plugging the recovered
    // IV back into bs_price reproduces the input price.
    const double S = 25000, K = 27000, T = 3.0 / 365.0, r = 0.065, q = 0.065, sigma = 0.18;
    const double price = bs_price(S, K, T, r, q, sigma, true);
    const IVResult res = implied_vol(price, S, K, T, r, q, true);
    assert(res.converged);
    const double repriced = bs_price(S, K, T, r, q, res.iv, true);
    assert(std::fabs(repriced - price) < 1e-6);
}

static void test_iv_fallback_path_triggers_and_converges() {
    // Force the vega floor to be unreachable-ly high so every step takes
    // the bisection fallback branch; the solver must still converge to a
    // reprice-consistent IV.
    const double S = 25000, K = 25200, T = 15.0 / 365.0, r = 0.065, q = 0.065, sigma = 0.16;
    const double price = bs_price(S, K, T, r, q, sigma, true);
    const IVResult res = implied_vol(price, S, K, T, r, q, true, /*initial_guess=*/0.3, /*tol=*/1e-8,
                                      /*max_iter=*/100, /*vega_floor=*/1e6);
    assert(res.used_fallback);
    assert(res.converged);
    const double repriced = bs_price(S, K, T, r, q, res.iv, true);
    assert(std::fabs(repriced - price) < 1e-6);
}

static void test_svi_recovers_seeded_params() {
    const SVIParams truth{0.02, 0.15, -0.3, 0.0, 0.2};
    std::vector<double> ks, ws;
    for (int i = -10; i <= 10; ++i) {
        const double k = i / 20.0;
        ks.push_back(k);
        ws.push_back(svi_total_variance(truth, k));
    }
    const SVICalibrationResult res =
        svi_calibrate(ks, ws, SVIParams{0.01, 0.1, 0.0, 0.0, 0.1});
    assert(res.rmse < 1e-6);
    const NoArbReport rep = svi_check_butterfly(res.params);
    assert(rep.butterfly_ok);
}

static void test_svi_calendar_no_arb() {
    const SVIParams near{0.01, 0.10, -0.2, 0.0, 0.15};
    const SVIParams far{0.02, 0.12, -0.2, 0.0, 0.18};
    std::vector<double> grid;
    for (int i = -10; i <= 10; ++i) grid.push_back(i / 20.0);
    assert(svi_check_calendar(near, far, grid));
    assert(!svi_check_calendar(far, near, grid));
}

int main() {
    test_put_call_parity();
    test_iv_roundtrip();
    test_iv_deep_otm_uses_fallback_or_converges();
    test_iv_fallback_path_triggers_and_converges();
    test_svi_recovers_seeded_params();
    test_svi_calendar_no_arb();
    std::printf("All qengine C++ tests passed.\n");
    return 0;
}
