// pybind11 bindings exposing the qengine C++ pricing/calibration primitives
// to Python as the `qengine` extension module. Consumed by
// core/pricer_bindings.py (falls back to a pure-Python/Numba implementation
// if this extension isn't built).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bs_pricer.hpp"
#include "svi.hpp"

namespace py = pybind11;

PYBIND11_MODULE(qengine, m) {
    m.doc() = "Low-latency C++ Black-Scholes / SVI engine for the NSE options market maker";

    py::class_<qengine::Greeks>(m, "Greeks")
        .def_readonly("delta", &qengine::Greeks::delta)
        .def_readonly("gamma", &qengine::Greeks::gamma)
        .def_readonly("theta", &qengine::Greeks::theta)
        .def_readonly("vega", &qengine::Greeks::vega)
        .def_readonly("rho", &qengine::Greeks::rho);

    py::class_<qengine::IVResult>(m, "IVResult")
        .def_readonly("iv", &qengine::IVResult::iv)
        .def_readonly("iterations", &qengine::IVResult::iterations)
        .def_readonly("converged", &qengine::IVResult::converged)
        .def_readonly("used_fallback", &qengine::IVResult::used_fallback);

    m.def("bs_price", &qengine::bs_price, py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"),
          py::arg("q"), py::arg("sigma"), py::arg("is_call"), "Black-Scholes-Merton price");

    m.def("bs_greeks", &qengine::bs_greeks, py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"),
          py::arg("q"), py::arg("sigma"), py::arg("is_call"), "Analytical Delta/Gamma/Theta/Vega/Rho");

    m.def("implied_vol", &qengine::implied_vol, py::arg("price"), py::arg("S"), py::arg("K"),
          py::arg("T"), py::arg("r"), py::arg("q"), py::arg("is_call"), py::arg("initial_guess") = 0.3,
          py::arg("tol") = 1e-8, py::arg("max_iter") = 100, py::arg("vega_floor") = 1e-8,
          py::arg("lo") = 1e-4, py::arg("hi") = 5.0,
          "Safeguarded Newton-Raphson (bisection fallback) implied volatility solver");

    py::class_<qengine::SVIParams>(m, "SVIParams")
        .def(py::init<double, double, double, double, double>(), py::arg("a"), py::arg("b"),
             py::arg("rho"), py::arg("m"), py::arg("sigma"))
        .def_readwrite("a", &qengine::SVIParams::a)
        .def_readwrite("b", &qengine::SVIParams::b)
        .def_readwrite("rho", &qengine::SVIParams::rho)
        .def_readwrite("m", &qengine::SVIParams::m)
        .def_readwrite("sigma", &qengine::SVIParams::sigma);

    py::class_<qengine::SVICalibrationResult>(m, "SVICalibrationResult")
        .def_readonly("params", &qengine::SVICalibrationResult::params)
        .def_readonly("rmse", &qengine::SVICalibrationResult::rmse)
        .def_readonly("iterations", &qengine::SVICalibrationResult::iterations)
        .def_readonly("converged", &qengine::SVICalibrationResult::converged);

    py::class_<qengine::NoArbReport>(m, "NoArbReport")
        .def_readonly("butterfly_ok", &qengine::NoArbReport::butterfly_ok)
        .def_readonly("min_density_proxy", &qengine::NoArbReport::min_density_proxy);

    m.def("svi_total_variance", &qengine::svi_total_variance, py::arg("params"), py::arg("k"));

    m.def("svi_calibrate", &qengine::svi_calibrate, py::arg("k"), py::arg("w"),
          py::arg("initial_guess"), py::arg("max_iter") = 200, py::arg("tol") = 1e-10,
          "Levenberg-Marquardt SVI calibration to (log-moneyness, total-variance) pairs");

    m.def("svi_check_butterfly", &qengine::svi_check_butterfly, py::arg("params"),
          "Static butterfly no-arbitrage sufficient-condition check for one SVI slice");

    m.def("svi_check_calendar", &qengine::svi_check_calendar, py::arg("near"), py::arg("far"),
          py::arg("k_grid"), "Calendar-spread no-arbitrage check between two SVI slices");
}
