#pragma once

#include <iostream>
#include <string>

namespace nanospline {
template <typename SplineType, typename Matrix>
void to_svg(std::ostream &out,
    SplineType &curve,
    const Matrix &offset,
    typename SplineType::Scalar scaling = 1,
    bool export_ctrl = true,
    const std::string &color = "red",
    typename SplineType::Scalar line_width = 2.0,
    const std::string &ctrl_color = "888888",
    typename SplineType::Scalar ctrl_radius = 0.2,
    int num_samples = 1e3)
{
    // TODO: maybe approximate this curve into many cubic beziers
    using Scalar = typename SplineType::Scalar;
    const auto &control_points = curve.get_control_points();
    assert(control_points.cols() == 2);
    const auto num_control_points = control_points.rows();

    const auto min_t = curve.get_domain_lower_bound();
    const auto max_t = curve.get_domain_upper_bound();
    out << "<path d=\"";
    for (int i = 0; i < num_samples + 1; i++) {
        auto t = min_t + Scalar(i) / Scalar(num_samples) * (max_t - min_t);
        t = std::max(min_t, std::min(max_t, t));
        const auto p = curve.evaluate(t);
        if (i == 0) {
            out << "M" << (p[0] + offset[0]) * scaling << "," << (p[1] + offset[1]) * scaling
                << " ";
        } else {
            out << "L" << (p[0] + offset[0]) * scaling << "," << (p[1] + offset[1]) * scaling
                << " ";
        }
    }
    out << "\" fill=\"none\" stroke=\"" << color << "\" stroke-width=\"" << line_width << "\"/>"
        << std::endl;

    if (!export_ctrl) return;

    out << "<g fill=\"#" << ctrl_color << "\">" << std::endl;
    for (int i = 0; i < num_control_points; i++) {
        const auto &p = control_points.row(i);
        out << "<circle cx=\"" << (p[0] + offset[0]) * scaling << "\" cy=\""
            << (p[1] + offset[1]) * scaling << "\" r=\"" << ctrl_radius * scaling << "\"/>"
            << std::endl;
    }
    out << "</g>" << std::endl;

    return;
}

template <typename SplineType, typename Matrix>
void to_eps(std::ostream &out,
    SplineType &curve,
    const Matrix &offset,
    typename SplineType::Scalar scaling = 1,
    bool export_ctrl = true,
    const std::string &color = "255 0 0",
    typename SplineType::Scalar line_width = 0.2,
    const std::string &ctrl_color = "88 88 88",
    typename SplineType::Scalar ctrl_radius = 1,
    int num_samples = 1e3)
{
    // TODO: maybe approximate this curve into many cubic beziers
    using Scalar = typename SplineType::Scalar;
    const auto &control_points = curve.get_control_points();
    assert(control_points.cols() == 2);
    const auto num_control_points = control_points.rows();

    const auto min_t = curve.get_domain_lower_bound();
    const auto max_t = curve.get_domain_upper_bound();
    out << color << " setrgbcolor\n";
    out << line_width << " setlinewidth\n";

    for (int i = 0; i < num_samples + 1; i++) {
        auto t = min_t + Scalar(i) / Scalar(num_samples) * (max_t - min_t);
        t = std::max(min_t, std::min(max_t, t));
        const auto p = curve.evaluate(t);
        if (i == 0) {
            out << (p[0] + offset[0]) * scaling << " " << (p[1] + offset[1]) * scaling
                << " moveto\n";
        } else {
            out << (p[0] + offset[0]) * scaling << " " << (p[1] + offset[1]) * scaling
                << " lineto\n";
        }
    }
    out << "stoke" << std::endl;

    if (!export_ctrl) return;

    out << ctrl_color << " setrgbcolor\n";
    for (int i = 0; i < num_control_points; i++) {
        const auto &p = control_points.row(i);
        out << (p[0] + offset[0]) * scaling << " " << (p[1] + offset[1]) * scaling << " "
            << ctrl_radius * scaling << " 0 360" << std::endl;
    }
    out << "fill" << std::endl;

    return;
}
} // namespace nanospline
