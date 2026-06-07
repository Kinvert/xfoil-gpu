// Deterministic scalar XFOIL surrogate kernel in C++.
//
// The implementation mirrors `native_cpu_kernels.py` and is intentionally
// conservative in behavior so Python and C++ outputs remain close enough for
// CPU baseline benchmarking and a future CUDA migration.

#include <Python.h>

#include <cmath>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

struct KernelGeometryFeatures {
    double n = 0.0;
    double camber = 0.0;
    double thickness = 0.0;
    double curvature = 0.0;
    double thickness_ratio = 0.0;
    double chord = 0.0;
};

KernelGeometryFeatures geometry_features(const std::vector<double>& xs, const std::vector<double>& ys) {
    KernelGeometryFeatures feat{};
    const auto n = static_cast<int>(xs.size());
    feat.n = static_cast<double>(n);

    if (n < 3) {
        return feat;
    }

    double x_min = xs[0];
    double x_max = xs[0];
    double y_min = ys[0];
    double y_max = ys[0];

    for (int i = 1; i < n; ++i) {
        if (xs[i] < x_min) x_min = xs[i];
        if (xs[i] > x_max) x_max = xs[i];
        if (ys[i] < y_min) y_min = ys[i];
        if (ys[i] > y_max) y_max = ys[i];
    }

    feat.chord = std::max(x_max - x_min, 1e-12);
    feat.camber = 0.5 * (y_max + y_min);
    feat.thickness = y_max - y_min;
    feat.thickness_ratio = feat.thickness / feat.chord;

    double curv_sum = 0.0;
    int segments = 0;
    for (int i = 1; i < n - 1; ++i) {
        const double x0 = xs[i - 1];
        const double y0 = ys[i - 1];
        const double x1 = xs[i];
        const double y1 = ys[i];
        const double x2 = xs[i + 1];
        const double y2 = ys[i + 1];

        const double dx1 = x1 - x0;
        const double dx2 = x2 - x1;
        if (std::fabs(dx1) < 1e-12 || std::fabs(dx2) < 1e-12) {
            continue;
        }

        const double s1 = std::atan2(y1 - y0, dx1);
        const double s2 = std::atan2(y2 - y1, dx2);
        curv_sum += std::fabs(s2 - s1);
        ++segments;
    }

    feat.curvature = curv_sum / std::max(1, segments);
    return feat;
}

PyObject* build_features_dict(const KernelGeometryFeatures& f) {
    PyObject* features = PyDict_New();
    if (!features) return nullptr;

    PyObject* value = nullptr;
    if ((value = PyFloat_FromDouble(f.n))) {
        PyDict_SetItemString(features, "n", value);
        Py_DECREF(value);
    }
    if ((value = PyFloat_FromDouble(f.camber))) {
        PyDict_SetItemString(features, "camber", value);
        Py_DECREF(value);
    }
    if ((value = PyFloat_FromDouble(f.thickness))) {
        PyDict_SetItemString(features, "thickness", value);
        Py_DECREF(value);
    }
    if ((value = PyFloat_FromDouble(f.curvature))) {
        PyDict_SetItemString(features, "curvature", value);
        Py_DECREF(value);
    }
    if ((value = PyFloat_FromDouble(f.thickness_ratio))) {
        PyDict_SetItemString(features, "thickness_ratio", value);
        Py_DECREF(value);
    }
    if ((value = PyFloat_FromDouble(f.chord))) {
        PyDict_SetItemString(features, "chord", value);
        Py_DECREF(value);
    }
    return features;
}

bool parse_points(PyObject* points_obj, std::vector<double>* xs, std::vector<double>* ys) {
    if (!PySequence_Check(points_obj)) {
        PyErr_SetString(PyExc_TypeError, "points must be a sequence");
        return false;
    }

    const Py_ssize_t n = PySequence_Size(points_obj);
    if (n < 0) return false;
    xs->reserve(static_cast<size_t>(n));
    ys->reserve(static_cast<size_t>(n));

    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* point = PySequence_GetItem(points_obj, i);
        if (!point) return false;

        if (!PySequence_Check(point) || PySequence_Size(point) != 2) {
            Py_DECREF(point);
            PyErr_SetString(PyExc_TypeError, "each point must be a 2-tuple/list");
            return false;
        }

        PyObject* px = PySequence_GetItem(point, 0);
        PyObject* py = PySequence_GetItem(point, 1);
        Py_DECREF(point);
        if (!px || !py) {
            Py_XDECREF(px);
            Py_XDECREF(py);
            return false;
        }

        const double x = PyFloat_AsDouble(px);
        const double y = PyFloat_AsDouble(py);
        Py_DECREF(px);
        Py_DECREF(py);
        if (PyErr_Occurred()) {
            return false;
        }
        xs->push_back(x);
        ys->push_back(y);
    }
    return true;
}

bool append_warning(PyObject* warnings, const char* text) {
    PyObject* item = PyUnicode_FromString(text);
    if (!item) {
        return false;
    }
    const int rc = PyList_Append(warnings, item);
    Py_DECREF(item);
    return rc == 0;
}

}  // namespace

static PyObject* estimate_aero(PyObject* /* self */, PyObject* args, PyObject* kwargs) {
    PyObject* points_obj = nullptr;
    double query_alpha_deg = 0.0;
    double query_reynolds = 1.0;
    double query_mach = 0.0;
    int n_panels = 80;
    int iterations_requested = 120;
    double stall_alpha_deg = 18.0;
    double residual_floor = 1e-10;
    int iterations_to_converge = 120;

    static const char* const kwlist[] = {
        "points",
        "query_alpha_deg",
        "query_reynolds",
        "query_mach",
        "n_panels",
        "iterations",
        "stall_alpha_deg",
        "residual_floor",
        "iterations_to_converge",
        nullptr,
    };

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "Odddiiddi", const_cast<char**>(const_cast<const char* const*>(kwlist)),
            &points_obj,
            &query_alpha_deg,
            &query_reynolds,
            &query_mach,
            &n_panels,
            &iterations_requested,
            &stall_alpha_deg,
            &residual_floor,
            &iterations_to_converge)) {
        return nullptr;
    }

    if (n_panels < 1) {
        n_panels = 1;
    }
    if (iterations_requested < 1) {
        iterations_requested = 1;
    }

    std::vector<double> xs;
    std::vector<double> ys;
    if (!parse_points(points_obj, &xs, &ys)) {
        return nullptr;
    }

    const auto features = geometry_features(xs, ys);
    const double alpha_rad = query_alpha_deg * kPi / 180.0;
    const double camber = features.camber;
    const double thickness_ratio = features.thickness_ratio;
    const double curvature = features.curvature;
    const double stall_alpha = std::max(1.0, std::fabs(query_alpha_deg));
    double stall_ratio = (stall_alpha - stall_alpha_deg) / 20.0;
    if (stall_ratio < 0.0) {
        stall_ratio = 0.0;
    } else if (stall_ratio > 1.0) {
        stall_ratio = 1.0;
    }

    double camber_norm = camber * 6.0;
    if (camber_norm < -0.6) camber_norm = -0.6;
    if (camber_norm > 0.6) camber_norm = 0.6;

    double panel_scale = static_cast<double>(n_panels);
    if (panel_scale < 1.0) panel_scale = 1.0;
    panel_scale = std::max(1.0, panel_scale / std::max(20.0, features.n));

    double cl = 2.0 * M_PI * alpha_rad;
    cl *= 1.0 + 0.25 * camber_norm;
    cl *= 1.0 + 0.08 * std::tanh(curvature);
    cl *= 1.0 + 0.20 * (std::min(2.0, panel_scale - 1.0) / 4.0);
    cl *= 1.0 - 0.70 * stall_ratio;

    if (query_mach > 0.0) {
        cl *= 1.0 - 0.12 * query_mach;
    }
    if (query_reynolds > 0.0) {
        const double re_scale = std::log10(std::max(query_reynolds, 1.0));
        cl *= 1.0 + 0.002 * std::max(0.0, re_scale - 5.0);
    }

    double cd = 0.004 + 0.018 * thickness_ratio;
    cd += 0.0025 * std::fabs(camber_norm);
    cd += 0.0012 * thickness_ratio * (1.0 + curvature);
    cd *= 1.0 + 0.5 * query_mach;
    cd *= 1.0 + 0.03 * stall_ratio;
    if (query_reynolds > 0.0) {
        cd *= 1.0 + 0.15 / std::sqrt(query_reynolds / 1.0e5 + 1.0);
    }

    double cm = -0.02 * (1.0 + camber_norm) - 0.003 * std::sin(alpha_rad * 2.0);
    cm -= 0.01 * thickness_ratio;

    const int iterations_used = std::min(iterations_requested, std::max(1, iterations_to_converge));
    double residual = std::max(residual_floor, std::exp(-0.04 * iterations_used));
    residual *= 1.0 + 0.8 * stall_ratio;

    const bool iterations_failed = (stall_ratio > 0.9) || (iterations_requested < static_cast<int>(0.25 * iterations_to_converge));
    const std::string status = iterations_failed ? "native_warnings" : "ok";

    PyObject* output = PyDict_New();
    if (!output) {
        return nullptr;
    }

    PyObject* warnings = PyList_New(0);
    if (!warnings) {
        Py_DECREF(output);
        return nullptr;
    }
    if (iterations_failed) {
        if (!append_warning(warnings, "proxy_low_iterations")) {
            Py_DECREF(warnings);
            Py_DECREF(output);
            return nullptr;
        }
    }
    if (stall_ratio > 0.55) {
        if (!append_warning(warnings, "proxy_stall_like_response")) {
            Py_DECREF(warnings);
            Py_DECREF(output);
            return nullptr;
        }
    }
    if (n_panels < 16) {
        if (!append_warning(warnings, "low_panel_resolution")) {
            Py_DECREF(warnings);
            Py_DECREF(output);
            return nullptr;
        }
    }

    PyObject* features_dict = build_features_dict(features);
    if (!features_dict) {
        Py_DECREF(warnings);
        Py_DECREF(output);
        return nullptr;
    }

    bool ok = true;
    ok &= PyDict_SetItemString(output, "cl", PyFloat_FromDouble(cl)) == 0;
    ok &= PyDict_SetItemString(output, "cd", PyFloat_FromDouble(cd)) == 0;
    ok &= PyDict_SetItemString(output, "cm", PyFloat_FromDouble(cm)) == 0;
    ok &= PyDict_SetItemString(output, "status", PyUnicode_FromString(status.c_str())) == 0;
    ok &= PyDict_SetItemString(output, "residual", PyFloat_FromDouble(residual)) == 0;
    ok &= PyDict_SetItemString(output, "iterations_used", PyLong_FromLong(iterations_used)) == 0;
    ok &= PyDict_SetItemString(output, "iterations_failed", PyBool_FromLong(iterations_failed)) == 0;
    ok &= PyDict_SetItemString(output, "warnings", warnings) == 0;
    ok &= PyDict_SetItemString(output, "features", features_dict) == 0;

    Py_DECREF(warnings);
    Py_DECREF(features_dict);

    if (!ok) {
        Py_DECREF(output);
        return nullptr;
    }
    return output;
}

static PyMethodDef NativeCpuMethods[] = {
    {"estimate_aero", (PyCFunction)estimate_aero, METH_VARARGS | METH_KEYWORDS, "Estimate aerodynamic coefficients."},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef native_cpu_cpp_module = {
    PyModuleDef_HEAD_INIT,
    "_native_cpu_cpp",
    "Native CPU kernel accelerator",
    -1,
    NativeCpuMethods,
};

PyMODINIT_FUNC PyInit__native_cpu_cpp(void) {
    return PyModule_Create(&native_cpu_cpp_module);
}
