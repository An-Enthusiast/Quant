#include "svi.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace qengine {

double svi_total_variance(const SVIParams& p, double k) {
    const double dk = k - p.m;
    return p.a + p.b * (p.rho * dk + std::sqrt(dk * dk + p.sigma * p.sigma));
}

namespace {

constexpr int kNumParams = 5;
using ParamVec = std::array<double, kNumParams>;

ParamVec to_vec(const SVIParams& p) { return {p.a, p.b, p.rho, p.m, p.sigma}; }
SVIParams to_params(const ParamVec& v) { return SVIParams{v[0], v[1], v[2], v[3], v[4]}; }

// Domain-derived bounds on (m, sigma), on top of the fixed algebraic box
// constraints (b >= 0, |rho| < 1, sigma > 0). Without these, a smile that
// looks locally close to linear over the *observed* strike range (common
// for short-dated slices where only a modest moneyness band is liquid) is
// under-identified: many (b, rho, m, sigma) combinations fit the observed
// points equally well by pushing the SVI "vertex" (m) far outside the data
// domain, which technically minimizes SSE but produces a meaningless,
// numerically fragile parameterization. Bounding m to (a small margin
// around) the observed k-range and sigma to a bounded multiple of its
// width -- the standard practical-calibration guardrail (see e.g. Zeliade
// Systems' SVI calibration notes) -- keeps the fit well-posed.
struct DomainBounds {
    double m_lo, m_hi, sigma_max;
};

ParamVec project(ParamVec v, const DomainBounds& db) {
    v[1] = std::max(v[1], 1e-8);                    // b >= 0
    v[2] = std::clamp(v[2], -0.999, 0.999);          // |rho| < 1
    v[3] = std::clamp(v[3], db.m_lo, db.m_hi);       // m within (a margin around) the data domain
    v[4] = std::clamp(v[4], 1e-6, db.sigma_max);     // 0 < sigma <= bounded multiple of domain width
    return v;
}

double sse(const ParamVec& v, const std::vector<double>& k, const std::vector<double>& w) {
    const SVIParams p = to_params(v);
    double s = 0.0;
    for (size_t i = 0; i < k.size(); ++i) {
        const double r = svi_total_variance(p, k[i]) - w[i];
        s += r * r;
    }
    return s;
}

// Solve a small symmetric positive-definite system A x = b via Gaussian
// elimination with partial pivoting (kNumParams is fixed and tiny, so a
// hand-rolled solver avoids pulling in a linear-algebra dependency).
bool solve_linear(std::array<std::array<double, kNumParams>, kNumParams> A, ParamVec b, ParamVec& x) {
    for (int col = 0; col < kNumParams; ++col) {
        int pivot = col;
        double best = std::fabs(A[col][col]);
        for (int row = col + 1; row < kNumParams; ++row) {
            if (std::fabs(A[row][col]) > best) {
                best = std::fabs(A[row][col]);
                pivot = row;
            }
        }
        if (best < 1e-14) return false;
        if (pivot != col) {
            std::swap(A[pivot], A[col]);
            std::swap(b[pivot], b[col]);
        }
        for (int row = col + 1; row < kNumParams; ++row) {
            const double factor = A[row][col] / A[col][col];
            for (int c = col; c < kNumParams; ++c) A[row][c] -= factor * A[col][c];
            b[row] -= factor * b[col];
        }
    }
    for (int row = kNumParams - 1; row >= 0; --row) {
        double s = b[row];
        for (int c = row + 1; c < kNumParams; ++c) s -= A[row][c] * x[c];
        x[row] = s / A[row][row];
    }
    return true;
}

}  // namespace

SVICalibrationResult svi_calibrate(const std::vector<double>& k, const std::vector<double>& w,
                                    SVIParams initial_guess, int max_iter, double tol) {
    const size_t n = k.size();
    double k_min = k.empty() ? -1.0 : k[0];
    double k_max = k.empty() ? 1.0 : k[0];
    for (double ki : k) {
        k_min = std::min(k_min, ki);
        k_max = std::max(k_max, ki);
    }
    const double range = std::max(k_max - k_min, 1e-3);
    const DomainBounds db{k_min - range, k_max + range, 2.0 * range};

    ParamVec x = project(to_vec(initial_guess), db);
    double lambda = 1e-3;
    double cost = sse(x, k, w);
    int iter = 0;
    bool converged = false;

    // Step size for central-difference numerical Jacobian, per parameter.
    const ParamVec h_step = {1e-5, 1e-5, 1e-6, 1e-5, 1e-6};

    for (; iter < max_iter; ++iter) {
        // Residuals and numerical Jacobian (n x kNumParams).
        std::vector<double> resid(n);
        const SVIParams px = to_params(x);
        for (size_t i = 0; i < n; ++i) resid[i] = svi_total_variance(px, k[i]) - w[i];

        std::vector<std::array<double, kNumParams>> J(n);
        for (int p = 0; p < kNumParams; ++p) {
            ParamVec xp = x, xm = x;
            xp[p] += h_step[p];
            xm[p] -= h_step[p];
            const SVIParams pp = to_params(project(xp, db));
            const SVIParams pm = to_params(project(xm, db));
            const double denom = 2.0 * h_step[p];
            for (size_t i = 0; i < n; ++i) {
                J[i][p] = (svi_total_variance(pp, k[i]) - svi_total_variance(pm, k[i])) / denom;
            }
        }

        // Normal equations: (J^T J + lambda * diag(J^T J)) delta = -J^T r
        std::array<std::array<double, kNumParams>, kNumParams> JTJ{};
        ParamVec JTr{};
        for (int a = 0; a < kNumParams; ++a) {
            for (int b = 0; b < kNumParams; ++b) {
                double s = 0.0;
                for (size_t i = 0; i < n; ++i) s += J[i][a] * J[i][b];
                JTJ[a][b] = s;
            }
            double s = 0.0;
            for (size_t i = 0; i < n; ++i) s += J[i][a] * resid[i];
            JTr[a] = s;
        }

        bool step_accepted = false;
        for (int attempt = 0; attempt < 12 && !step_accepted; ++attempt) {
            auto A = JTJ;
            for (int d = 0; d < kNumParams; ++d) A[d][d] += lambda * std::max(JTJ[d][d], 1e-10);
            ParamVec neg_JTr{};
            for (int d = 0; d < kNumParams; ++d) neg_JTr[d] = -JTr[d];

            ParamVec delta{};
            if (!solve_linear(A, neg_JTr, delta)) {
                lambda *= 10.0;
                continue;
            }
            ParamVec candidate = x;
            for (int d = 0; d < kNumParams; ++d) candidate[d] += delta[d];
            candidate = project(candidate, db);
            const double new_cost = sse(candidate, k, w);

            if (new_cost < cost) {
                x = candidate;
                const double improvement = cost - new_cost;
                cost = new_cost;
                lambda = std::max(lambda * 0.5, 1e-12);
                step_accepted = true;
                if (improvement < tol) {
                    converged = true;
                }
            } else {
                lambda *= 10.0;
            }
        }
        if (!step_accepted || converged) {
            if (step_accepted) ++iter;
            break;
        }
    }

    const double rmse = std::sqrt(std::max(cost, 0.0) / std::max<size_t>(n, 1));
    return SVICalibrationResult{to_params(x), rmse, iter + 1, converged || rmse < 1e-4};
}

NoArbReport svi_check_butterfly(const SVIParams& p) {
    NoArbReport rep{};
    const double vertex_min = p.a + p.b * p.sigma * std::sqrt(std::max(1.0 - p.rho * p.rho, 0.0));
    const bool basic_ok = p.b >= 0.0 && std::fabs(p.rho) < 1.0 && p.sigma > 0.0;
    const bool wing_ok = p.b * (1.0 + std::fabs(p.rho)) <= 4.0 + 1e-9;
    rep.min_density_proxy = vertex_min;
    rep.butterfly_ok = basic_ok && wing_ok && vertex_min >= -1e-9;
    return rep;
}

bool svi_check_calendar(const SVIParams& near, const SVIParams& far, const std::vector<double>& k_grid) {
    for (double k : k_grid) {
        if (svi_total_variance(far, k) < svi_total_variance(near, k) - 1e-9) {
            return false;
        }
    }
    return true;
}

}  // namespace qengine
