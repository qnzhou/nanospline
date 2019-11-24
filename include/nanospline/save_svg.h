#pragma once

#include <Eigen/Core>
#include <string>
#include <fstream>

namespace nanospline {

template<typename SplineType>
void save_svg(const std::string& filename, const SplineType& curve) {
    using Scalar = typename SplineType::Scalar;
    const auto& control_points = curve.get_control_points();
    assert(control_points.cols() == 2);
    const auto num_control_points = control_points.rows();
    const auto bbox_min = control_points.colwise().minCoeff().eval();
    const auto bbox_max = control_points.colwise().maxCoeff().eval();
    const auto bbox_dim = bbox_max - bbox_min;
    const auto margin = bbox_dim.maxCoeff() / 10;
    const auto s = 1000.0 / bbox_dim.maxCoeff();
    const auto r = bbox_dim.norm() / 100 * s;
    const auto o = -bbox_min.array() + margin;

    std::ofstream fout(filename.c_str());
    fout << "<?xml version=\"1.0\" standalone=\"no\"?>" << std::endl;
    fout << "<svg width=\"12cm\" height=\"6cm\" "
        << "viewBox=\"" 
        << 0 << " " << 0 << " "
        << (bbox_dim[0]+2*margin)*s << " " << (bbox_dim[1]+2*margin)*s << "\" "
        << "xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">" << std::endl;

    fout << "<rect x=\"" << 0 << "\" y=\"" << 0
        << "\" width=\"" << (bbox_dim[0]+2*margin) * s
        << "\" height=\"" << (bbox_dim[1]+2*margin) * s
        << "\" fill=\"none\" stroke=\"blue\" stroke-width=\"1\" />" << std::endl;

    constexpr int num_samples = 1e3;
    const auto min_t = curve.get_domain_lower_bound();
    const auto max_t = curve.get_domain_upper_bound();
    fout << "<path d=\"";
    for (int i=0; i<num_samples+1; i++) {
        auto t = min_t + Scalar(i)/Scalar(num_samples) * (max_t-min_t);
        t = std::max(min_t, std::min(max_t, t));
        const auto p = curve.evaluate(t);
        if (i==0) {
            fout << "M" << (p[0]+o[0])*s << "," << (p[1]+o[1])*s << " ";
        } else {
            fout << "L" << (p[0]+o[0])*s << "," << (p[1]+o[1])*s << " ";
        }
    }
    fout << "\" fill=\"none\" stroke=\"red\" strock-width=\""
        << (int)std::round(0.2 * r) << "\"/>" << std::endl;

    fout << "<g fill=\"#888888\">" << std::endl;
    for (int i=0; i<num_control_points; i++) {
        const auto& p = control_points.row(i);
        fout << "<circle cx=\"" << (p[0]+o[0])*s
            <<"\" cy=\"" << (p[1]+o[1])*s << "\" r=\""
            << r << "\"/>" << std::endl;
    }
    fout << "</g>" << std::endl;

    fout << "</svg>" << std::endl;
    fout.close();
}

}
