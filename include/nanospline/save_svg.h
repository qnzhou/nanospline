#pragma once

#include <nanospline/VectorExport.h>
#include <Eigen/Core>
#include <fstream>
#include <limits>
#include <string>


namespace nanospline {

template <typename SplineType>
void save_svg(const std::string& filename, const SplineType& curve)
{
    const auto& control_points = curve.get_control_points();
    assert(control_points.cols() == 2);
    const auto bbox_min = control_points.colwise().minCoeff().eval();
    const auto bbox_max = control_points.colwise().maxCoeff().eval();
    const auto bbox_dim = bbox_max - bbox_min;
    const auto margin = bbox_dim.maxCoeff() / 10;
    const auto s = 1000.0 / bbox_dim.maxCoeff();
    const auto r = bbox_dim.norm() / 100;
    const auto o = -bbox_min.array() + margin;

    std::ofstream fout(filename.c_str());
    fout << "<?xml version=\"1.0\" standalone=\"no\"?>" << std::endl;
    fout << "<svg width=\"12cm\" height=\"6cm\" "
         << "viewBox=\"" << 0 << " " << 0 << " " << (bbox_dim[0] + 2 * margin) * s << " "
         << (bbox_dim[1] + 2 * margin) * s << "\" "
         << "xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">" << std::endl;

    fout << "<rect x=\"" << 0 << "\" y=\"" << 0 << "\" width=\"" << (bbox_dim[0] + 2 * margin) * s
         << "\" height=\"" << (bbox_dim[1] + 2 * margin) * s
         << "\" fill=\"none\" stroke=\"blue\" stroke-width=\"1\" />" << std::endl;

    constexpr int num_samples = 1e3;
    to_svg(fout, curve, o, s, true, "red", 0.2, "888888", r, num_samples);

    fout << "</svg>" << std::endl;
    fout.close();
}

template <typename SplineType>
void save_svg(const std::string& filename, const std::vector<SplineType>& curves)
{
    using Scalar = typename SplineType::Scalar;

    Eigen::Matrix<Scalar, 1, 2> bbox_min;
    Eigen::Matrix<Scalar, 1, 2> bbox_max;
    bbox_min.setConstant(std::numeric_limits<Scalar>::max());
    bbox_max.setConstant(std::numeric_limits<Scalar>::lowest());

    for (const auto& curve : curves) {
        const auto& control_points = curve.get_control_points();
        assert(control_points.cols() == 2);
        const auto curr_bbox_min = control_points.colwise().minCoeff().eval();
        const auto curr_bbox_max = control_points.colwise().maxCoeff().eval();
        bbox_min = bbox_min.array().min(curr_bbox_min.array());
        bbox_max = bbox_max.array().max(curr_bbox_max.array());
    }

    const auto bbox_dim = bbox_max - bbox_min;
    const auto margin = bbox_dim.maxCoeff() / 10;
    const auto s = 1000.0 / bbox_dim.maxCoeff();
    const auto r = bbox_dim.norm() / 100;
    const auto o = -bbox_min.array() + margin;

    std::ofstream fout(filename.c_str());
    fout << "<?xml version=\"1.0\" standalone=\"no\"?>" << std::endl;
    fout << "<svg width=\"12cm\" height=\"6cm\" "
         << "viewBox=\"" << 0 << " " << 0 << " " << (bbox_dim[0] + 2 * margin) * s << " "
         << (bbox_dim[1] + 2 * margin) * s << "\" "
         << "xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">" << std::endl;
    fout << "<rect x=\"" << 0 << "\" y=\"" << 0 << "\" width=\"" << (bbox_dim[0] + 2 * margin) * s
         << "\" height=\"" << (bbox_dim[1] + 2 * margin) * s
         << "\" fill=\"none\" stroke=\"blue\" stroke-width=\"1\" />" << std::endl;

    constexpr int num_samples = 1e3;
    for (const auto& curve : curves) {
        to_svg(fout, curve, o, s, true, "red", 0.2, "888888", r, num_samples);
    }

    fout << "</svg>" << std::endl;
    fout.close();
}

} // namespace nanospline
