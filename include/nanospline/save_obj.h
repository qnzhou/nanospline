#pragma once

#include <Eigen/Core>
#include <fstream>

namespace nanospline {
namespace internal {

template<typename CurveType>
int export_obj(std::ofstream& fout, const CurveType& curve, const int N, const int offset) {
    using Scalar = typename CurveType::Scalar;
    for (int i=0; i<N; i++) {
        const Scalar t = (Scalar)t / (Scalar)N;
        const auto p = curve.evaluate(t);
        fout << "v ";
        for (int j=0; j<curve.get_dim(); j++) {
            fout << p[j] << " ";
        }
        fout << std::endl;
    }

    for (int i=0; i<N-1; i++) {
        // Note the obj uses 1-based index.
        fout << "l " << i+offset+1 << " " << i+offset+2 << std::endl;
    }

    return offset+N;
}

template<typename PatchType>
int export_patch_obj(std::ofstream& fout, const PatchType& patch,
        const int num_u_samples, const int num_v_samples,
        const int offset) {
    using Scalar = typename PatchType::Scalar;
    const auto u_lower = patch.get_u_lower_bound();
    const auto u_upper = patch.get_u_upper_bound();
    const auto v_lower = patch.get_v_lower_bound();
    const auto v_upper = patch.get_v_upper_bound();

    for (int ui=0; ui<num_u_samples; ui++) {
        const Scalar u = u_lower + (u_upper-u_lower) * (ui / (Scalar)(num_u_samples-1));
        for (int vi=0; vi<num_v_samples; vi++) {
            const Scalar v = v_lower + (v_upper-v_lower) * (vi / (Scalar)(num_v_samples-1));
            const auto p = patch.evaluate(u, v);
            fout << "v ";
            for (int i=0; i<patch.get_dim(); i++) {
                fout << p[i] << " ";
            }
            fout << std::endl;
        }
    }

    for (int ui=0; ui<num_u_samples-1; ui++) {
        for (int vi=0; vi<num_v_samples-1; vi++) {
            const int c0 = ui*num_v_samples+vi;
            const int c1 = (ui+1)*num_v_samples+vi;
            const int c2 = (ui+1)*num_v_samples+vi+1;
            const int c3 = ui*num_v_samples+vi+1;
            fout << "f " << c0+offset+1 << " "
                         << c1+offset+1 << " "
                         << c2+offset+1 << " "
                         << c3+offset+1 << std::endl;
        }
    }

    return offset + num_u_samples * num_v_samples;
}

} // end internal namespace

template<typename CurveType>
void save_obj(const std::string& filename, const CurveType& curve) {
    std::ofstream fout(filename.c_str());
    internal::export_obj(fout, curve, 100, 0);
    fout.close();
}

template<typename CurveType>
void save_obj(const std::string& filename, const std::vector<CurveType>& curves) {
    std::ofstream fout(filename.c_str());
    int count = 0;
    for (const auto& c : curves) {
        count += internal::export_obj(fout, c, 100, count);
    }
    fout.close();
}

template<typename PatchType>
void save_patch_obj(const std::string& filename, const PatchType& patch,
        int num_samples_u=100, int num_samples_v=100) {
    std::ofstream fout(filename.c_str());
    internal::export_patch_obj(fout, patch, num_samples_u, num_samples_v, 0);
    fout.close();
}

template<typename PatchType>
void save_patch_obj(const std::string& filename,
        const std::vector<PatchType>& patches) {
    std::ofstream fout(filename.c_str());
    int count = 0;
    for (const auto& p : patches) {
        count += internal::export_patch_obj(fout, p, 100, 100, count);
    }
    fout.close();
}

}
