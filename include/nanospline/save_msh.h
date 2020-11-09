#pragma once

#include <nanospline/CurveBase.h>
#include <nanospline/PatchBase.h>

#include <MshIO/mshio.h>

#include <Eigen/Core>
#include <fstream>

namespace nanospline {

namespace internal {

using namespace mshio;

template <typename CurveType>
void add_curve(MshSpec& spec, const CurveType& curve, const size_t N, int tag = 1)
{
    using Scalar = typename CurveType::Scalar;
    const size_t node_offset = spec.nodes.max_node_tag;
    const size_t element_offset = spec.elements.max_element_tag;

    spec.nodes.entity_blocks.emplace_back();
    spec.nodes.num_entity_blocks += 1;
    spec.nodes.num_nodes += N + 1;
    spec.nodes.min_node_tag = tag;
    spec.nodes.max_node_tag += N + 1;

    auto& node_block = spec.nodes.entity_blocks.back();
    node_block.entity_dim = 1;
    node_block.entity_tag = 1;
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

template <typename PatchType>
void add_patch(MshSpec& spec,
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

            if (i != num_v_samples && j != num_v_samples) {
                element_block.data.push_back(element_offset + i * (num_v_samples) + j + 1);
                element_block.data.push_back(node_offset + i * (num_v_samples + 1) + j + 1);
                element_block.data.push_back(node_offset + i * (num_v_samples + 1) + j + 2);
                element_block.data.push_back(node_offset + (i + 1) * (num_v_samples + 1) + j + 2);
                element_block.data.push_back(node_offset + (i + 1) * (num_v_samples + 1) + j + 1);
            }
        }
    }
}

} // namespace internal

template <typename Scalar, int dim>
void save_msh(const std::string& filename,
    const std::vector<CurveBase<Scalar, dim>*>& curves,
    const std::vector<PatchBase<Scalar, dim>*>& patches)
{
    using namespace mshio;

    MshSpec spec;
    spec.mesh_format.version = "4.1";
    spec.mesh_format.file_type = 0;

    int tag = 0;
    for (auto& curve : curves) {
        internal::add_curve(spec, *curve, 100, tag);
        tag++;
    }
    for (auto& patch : patches) {
        internal::add_patch(spec,
            *patch,
            5 * patch->get_num_control_points_u(),
            5 * patch->get_num_control_points_v(),
            tag);
        tag++;
    }

    save_msh(filename, spec);
}


} // namespace nanospline
