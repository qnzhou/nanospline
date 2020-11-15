#pragma once

#include <nanospline/CurveBase.h>
#include <nanospline/PatchBase.h>
#include <nanospline/enums.h>

#include <MshIO/mshio.h>

#include <Eigen/Core>
#include <fstream>

namespace nanospline {

namespace internal {

using namespace mshio;

template <typename CurveType>
void add_curve_sampled(MshSpec& spec, const CurveType& curve, const size_t N, int tag = 1)
{
    using Scalar = typename CurveType::Scalar;
    const size_t node_offset = spec.nodes.max_node_tag;
    const size_t element_offset = spec.elements.max_element_tag;

    spec.nodes.entity_blocks.emplace_back();
    spec.nodes.num_entity_blocks += 1;
    spec.nodes.num_nodes += N + 1;
    spec.nodes.min_node_tag = static_cast<size_t>(tag);
    spec.nodes.max_node_tag += N + 1;

    auto& node_block = spec.nodes.entity_blocks.back();
    node_block.entity_dim = 1;
    node_block.entity_tag = tag;
    node_block.parametric = 1;
    node_block.num_nodes_in_block = N + 1;

    spec.elements.entity_blocks.emplace_back();
    spec.elements.num_entity_blocks += 1;
    spec.elements.num_elements += N;
    spec.elements.min_element_tag = 1;
    spec.elements.max_element_tag += N;

    auto& element_block = spec.elements.entity_blocks.back();
    element_block.entity_dim = 1;
    element_block.entity_tag = tag;
    element_block.element_type = 1;
    element_block.num_elements_in_block = N;

    const Scalar t_min = curve.get_domain_lower_bound();
    const Scalar t_max = curve.get_domain_upper_bound();

    for (size_t i = 0; i <= N; i++) {
        const Scalar r = (Scalar)i / (Scalar)N;
        const Scalar t = t_min * (1 - r) + t_max * r;
        auto p = curve.evaluate(t);

        node_block.tags.push_back(node_offset + i + 1);
        node_block.data.push_back(p[0]);
        node_block.data.push_back(p[1]);
        if (curve.get_dim() == 2) {
            node_block.data.push_back(0);
        } else {
            node_block.data.push_back(p[2]);
        }
        node_block.data.push_back(t);

        if (i != N) {
            element_block.data.push_back(element_offset + i + 1);
            element_block.data.push_back(node_offset + i + 1);
            element_block.data.push_back(node_offset + i + 2);
        }
    }
}

template <typename CurveType>
void add_curve(MshSpec& spec, const CurveType& curve, int tag = 1)
{
    auto& curves = spec.curves;
    curves.emplace_back();
    auto& curve_spec = curves.back();

    const int dim = curve.get_dim();
    const int degree = curve.get_degree();
    const int num_control_points = curve.get_num_control_points();
    const int num_knots = curve.get_num_knots();
    const int num_weights = curve.get_num_weights();

    if (dim != 3 && dim != 2) {
        throw not_implemented_error("Msh format only support 2D or 3D data.");
    }

    curve_spec.curve_tag = static_cast<size_t>(tag);
    if (num_knots == 0) {
        curve_spec.curve_type = static_cast<size_t>(
            (num_weights == 0) ? CurveEnum::BEZIER : CurveEnum::RATIONAL_BEZIER);
    } else {
        curve_spec.curve_type =
            static_cast<size_t>((num_weights == 0) ? CurveEnum::BSPLINE : CurveEnum::NURBS);
    }
    curve_spec.curve_degree = static_cast<size_t>(degree);
    curve_spec.num_control_points = static_cast<size_t>(num_control_points);
    curve_spec.num_knots = static_cast<size_t>(num_knots);
    curve_spec.with_weights = num_weights > 0;
    curve_spec.data.reserve(static_cast<size_t>(num_control_points * 3 + num_weights + num_knots));

    for (size_t i = 0; i < static_cast<size_t>(num_control_points); i++) {
        const auto p = curve.get_control_point(static_cast<int>(i)).template cast<double>();
        for (size_t j = 0; j < dim; j++) {
            curve_spec.data.push_back(p[static_cast<Eigen::Index>(j)]);
        }
        if (dim == 2) curve_spec.data.push_back(0);
        if (num_weights > 0) {
            curve_spec.data.push_back(static_cast<double>(curve.get_weight(static_cast<int>(i))));
        }
    }
    for (size_t i = 0; i < static_cast<size_t>(num_knots); i++) {
        curve_spec.data.push_back(static_cast<double>(curve.get_knot(static_cast<int>(i))));
    }
}

template <typename PatchType>
void add_patch_sampled(MshSpec& spec,
    const PatchType& patch,
    const size_t num_u_samples,
    const size_t num_v_samples,
    int tag = 1)
{
    using Scalar = typename PatchType::Scalar;
    const size_t node_offset = spec.nodes.max_node_tag;
    const size_t element_offset = spec.elements.max_element_tag;

    const size_t N = (num_u_samples + 1) * (num_v_samples + 1);
    const size_t M = num_u_samples * num_v_samples;

    spec.nodes.entity_blocks.emplace_back();
    spec.nodes.num_entity_blocks += 1;
    spec.nodes.num_nodes += N;
    spec.nodes.min_node_tag = 1;
    spec.nodes.max_node_tag += N;

    auto& node_block = spec.nodes.entity_blocks.back();
    node_block.entity_dim = 2;
    node_block.entity_tag = tag;
    node_block.parametric = 1;
    node_block.num_nodes_in_block = N;

    spec.elements.entity_blocks.emplace_back();
    spec.elements.num_entity_blocks += 1;
    spec.elements.num_elements += M;
    spec.elements.min_element_tag = 1;
    spec.elements.max_element_tag += M;

    auto& element_block = spec.elements.entity_blocks.back();
    element_block.entity_dim = 2;
    element_block.entity_tag = tag;
    element_block.element_type = 3;
    element_block.num_elements_in_block = M;

    const Scalar u_min = patch.get_u_lower_bound();
    const Scalar u_max = patch.get_u_upper_bound();
    const Scalar v_min = patch.get_v_lower_bound();
    const Scalar v_max = patch.get_v_upper_bound();

    for (size_t i = 0; i <= num_u_samples; i++) {
        const Scalar ru = (Scalar)i / (Scalar)num_u_samples;
        const Scalar u = u_min * (1 - ru) + u_max * ru;
        for (size_t j = 0; j <= num_v_samples; j++) {
            const Scalar rv = (Scalar)j / (Scalar)num_v_samples;
            const Scalar v = v_min * (1 - rv) + v_max * rv;

            auto p = patch.evaluate(u, v);

            node_block.tags.push_back(node_offset + i * (num_v_samples + 1) + j + 1);
            node_block.data.push_back(p[0]);
            node_block.data.push_back(p[1]);
            if (patch.get_dim() == 2) {
                node_block.data.push_back(0);
            } else {
                node_block.data.push_back(p[2]);
            }
            node_block.data.push_back(u);
            node_block.data.push_back(v);

            if (i != num_u_samples && j != num_v_samples) {
                element_block.data.push_back(element_offset + i * (num_v_samples) + j + 1);
                element_block.data.push_back(node_offset + i * (num_v_samples + 1) + j + 1);
                element_block.data.push_back(node_offset + (i + 1) * (num_v_samples + 1) + j + 1);
                element_block.data.push_back(node_offset + (i + 1) * (num_v_samples + 1) + j + 2);
                element_block.data.push_back(node_offset + i * (num_v_samples + 1) + j + 2);
            }
        }
    }
}

template <typename PatchType>
void add_patch(MshSpec& spec, const PatchType& patch, int tag = 1)
{
    spec.patches.emplace_back();
    auto& patch_spec = spec.patches.back();

    const auto dim = patch.get_dim();
    const auto degree_u = patch.get_degree_u();
    const auto degree_v = patch.get_degree_v();
    const auto num_control_points = patch.num_control_points();
    const auto num_control_points_u = patch.num_control_points_u();
    const auto num_control_points_v = patch.num_control_points_v();
    const auto num_weights_u = patch.get_num_weights_u();
    const auto num_weights_v = patch.get_num_weights_v();
    const auto num_knots_u = patch.get_num_knots_u();
    const auto num_knots_v = patch.get_num_knots_v();

    patch_spec.patch_tag = static_cast<size_t>(tag);
    if (num_knots_u == 0 && num_knots_v == 0) {
        patch_spec.patch_type = static_cast<size_t>((num_weights_u == 0 && num_weights_v == 0)
                                                        ? PatchEnum::BEZIER
                                                        : PatchEnum::RATIONAL_BEZIER);
    } else {
        patch_spec.patch_type = static_cast<size_t>(
            (num_weights_u == 0 && num_weights_v == 0) ? PatchEnum::BSPLINE : PatchEnum::NURBS);
    }
    patch_spec.degree_u = static_cast<size_t>(degree_u);
    patch_spec.degree_v = static_cast<size_t>(degree_v);
    patch_spec.num_control_points = static_cast<size_t>(num_control_points);
    patch_spec.num_u_knots = static_cast<size_t>(num_knots_u);
    patch_spec.num_v_knots = static_cast<size_t>(num_knots_v);
    patch_spec.with_weights = num_weights_u > 0;

    patch_spec.data.reserve(static_cast<size_t>(
        num_control_points * 3 + num_weights_u * num_weights_v + num_knots_u + num_knots_v));

    for (int i = 0; i < num_control_points_u; i++) {
        for (int j = 0; j < num_control_points_v; j++) {
            const auto p = patch.get_control_point(i, j).template cast<double>().eval();
            for (int k = 0; k < dim; k++) {
                patch_spec.data.push_back(p[static_cast<Eigen::Index>(k)]);
            }
            if (dim == 2) {
                patch_spec.data.push_back(0);
            }
            if (patch_spec.with_weights) {
                patch_spec.data.push_back(static_cast<double>(patch.get_weight(i, j)));
            }
        }
    }
    for (int i = 0; i < num_knots_u; i++) {
        patch_spec.data.push_back(patch.get_knot_u(i));
    }
    for (int i = 0; i < num_knots_v; i++) {
        patch_spec.data.push_back(patch.get_knot_v(i));
    }
}

} // namespace internal

template <typename Scalar, int dim=3>
void save_msh(const std::string& filename,
    const std::vector<CurveBase<Scalar, dim>*>& curves,
    const std::vector<PatchBase<Scalar, dim>*>& patches,
    bool binary=true)
{
    using namespace mshio;

    MshSpec spec;
    spec.mesh_format.version = "4.1";
    spec.mesh_format.file_type = binary?1:0;

    int tag = 1;
    for (auto& curve : curves) {
        internal::add_curve(spec, *curve, tag);
        internal::add_curve_sampled(spec, *curve, 100, tag);
        tag++;
    }
    tag = 1;
    for (auto& patch : patches) {
        internal::add_patch_sampled(spec,
            *patch,
            static_cast<size_t>(5 * patch->num_control_points_u()),
            static_cast<size_t>(5 * patch->num_control_points_v()),
            tag);
        internal::add_patch(spec, *patch, tag);
        tag++;
    }

    save_msh(filename, spec);
}


} // namespace nanospline
