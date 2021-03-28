#pragma once

#include <nanospline/BSpline.h>
#include <nanospline/BSplinePatch.h>
#include <nanospline/Bezier.h>
#include <nanospline/BezierPatch.h>
#include <nanospline/CurveBase.h>
#include <nanospline/NURBS.h>
#include <nanospline/NURBSPatch.h>
#include <nanospline/PatchBase.h>
#include <nanospline/RationalBezier.h>
#include <nanospline/RationalBezierPatch.h>
#include <nanospline/enums.h>

#include <mshio/mshio.h>

#include <Eigen/Core>

#include <cassert>
#include <memory>
#include <vector>

namespace nanospline {

namespace internal {

inline Eigen::Matrix<double, Eigen::Dynamic, 3> extract_control_points(
    const mshio::Curve& curve_spec)
{
    const size_t entry_size = curve_spec.with_weights ? 4 : 3;
    Eigen::Matrix<double, Eigen::Dynamic, 3> control_points(curve_spec.num_control_points, 3);
    for (size_t i = 0; i < curve_spec.num_control_points; i++) {
        control_points(static_cast<Eigen::Index>(i), 0) = curve_spec.data[i * entry_size];
        control_points(static_cast<Eigen::Index>(i), 1) = curve_spec.data[i * entry_size + 1];
        control_points(static_cast<Eigen::Index>(i), 2) = curve_spec.data[i * entry_size + 2];
    }
    return control_points;
}

inline Eigen::Matrix<double, Eigen::Dynamic, 3> extract_control_points(
    const mshio::Patch& patch_spec)
{
    const size_t entry_size = patch_spec.with_weights ? 4 : 3;
    Eigen::Matrix<double, Eigen::Dynamic, 3> control_points(patch_spec.num_control_points, 3);
    for (size_t i = 0; i < patch_spec.num_control_points; i++) {
        control_points(static_cast<Eigen::Index>(i), 0) = patch_spec.data[i * entry_size];
        control_points(static_cast<Eigen::Index>(i), 1) = patch_spec.data[i * entry_size + 1];
        control_points(static_cast<Eigen::Index>(i), 2) = patch_spec.data[i * entry_size + 2];
    }
    return control_points;
}


inline Eigen::Matrix<double, Eigen::Dynamic, 1> extract_weights(const mshio::Curve& curve_spec)
{
    constexpr size_t entry_size = 4;
    Eigen::Matrix<double, Eigen::Dynamic, 1> weights(curve_spec.num_control_points, 1);
    for (size_t i = 0; i < curve_spec.num_control_points; i++) {
        weights[static_cast<Eigen::Index>(i)] = curve_spec.data[i * entry_size + 3];
    }
    return weights;
}

inline Eigen::Matrix<double, Eigen::Dynamic, 1> extract_weights(const mshio::Patch& patch_spec)
{
    constexpr size_t entry_size = 4;
    Eigen::Matrix<double, Eigen::Dynamic, 1> weights(patch_spec.num_control_points, 1);
    for (size_t i = 0; i < patch_spec.num_control_points; i++) {
        weights[static_cast<Eigen::Index>(i)] = patch_spec.data[i * entry_size + 3];
    }
    return weights;
}

inline Eigen::Matrix<double, Eigen::Dynamic, 1> extract_knots(const mshio::Curve& curve_spec)
{
    const size_t entry_size = curve_spec.with_weights ? 4 : 3;
    Eigen::Matrix<double, Eigen::Dynamic, 1> knots(curve_spec.num_knots, 1);
    for (size_t i = 0; i < curve_spec.num_knots; i++) {
        knots[static_cast<Eigen::Index>(i)] =
            curve_spec.data[curve_spec.num_control_points * entry_size + i];
    }
    return knots;
}

inline Eigen::Matrix<double, Eigen::Dynamic, 1> extract_knots_u(const mshio::Patch& patch_spec)
{
    const size_t entry_size = patch_spec.with_weights ? 4 : 3;
    const size_t index_offset = patch_spec.num_control_points * entry_size;
    Eigen::Matrix<double, Eigen::Dynamic, 1> knots(patch_spec.num_u_knots, 1);
    for (size_t i = 0; i < patch_spec.num_u_knots; i++) {
        knots[static_cast<Eigen::Index>(i)] = patch_spec.data[index_offset + i];
    }
    return knots;
}

inline Eigen::Matrix<double, Eigen::Dynamic, 1> extract_knots_v(const mshio::Patch& patch_spec)
{
    const size_t entry_size = patch_spec.with_weights ? 4 : 3;
    const size_t index_offset = patch_spec.num_control_points * entry_size + patch_spec.num_u_knots;
    Eigen::Matrix<double, Eigen::Dynamic, 1> knots(patch_spec.num_v_knots, 1);
    for (size_t i = 0; i < patch_spec.num_v_knots; i++) {
        knots[static_cast<Eigen::Index>(i)] = patch_spec.data[index_offset + i];
    }
    return knots;
}


template <typename Scalar, int dim>
std::unique_ptr<CurveBase<Scalar, dim>> load_Bezier_curve(const mshio::Curve& curve_spec)
{
    assert(curve_spec.num_knots == 0);
    assert(!curve_spec.with_weights);
    assert(curve_spec.num_control_points == (curve_spec.curve_degree + 1));
    assert(curve_spec.data.size() == curve_spec.num_control_points * 3);

    auto curve = std::make_unique<Bezier<Scalar, dim, -1>>();
    curve->set_control_points(extract_control_points(curve_spec));
    return curve;
}

template <typename Scalar, int dim>
std::unique_ptr<CurveBase<Scalar, dim>> load_BSpline_curve(const mshio::Curve& curve_spec)
{
    assert(!curve_spec.with_weights);
    assert(curve_spec.num_knots == curve_spec.num_control_points + curve_spec.curve_degree + 1);
    assert(curve_spec.data.size() == curve_spec.num_control_points * 3 + curve_spec.num_knots);

    auto curve = std::make_unique<BSpline<Scalar, dim, -1>>();

    curve->set_control_points(extract_control_points(curve_spec));
    curve->set_knots(extract_knots(curve_spec));
    return curve;
}

template <typename Scalar, int dim>
std::unique_ptr<CurveBase<Scalar, dim>> load_rational_Bezier_curve(const mshio::Curve& curve_spec)
{
    assert(curve_spec.with_weights);
    assert(curve_spec.num_knots == 0);
    assert(curve_spec.data.size() == curve_spec.num_control_points * 4);

    auto curve = std::make_unique<RationalBezier<Scalar, dim, -1>>();

    curve->set_control_points(extract_control_points(curve_spec));
    curve->set_weights(extract_weights(curve_spec));
    curve->initialize();
    return curve;
}

template <typename Scalar, int dim>
std::unique_ptr<CurveBase<Scalar, dim>> load_NURBS_curve(const mshio::Curve& curve_spec)
{
    assert(curve_spec.with_weights);
    assert(curve_spec.num_knots == curve_spec.num_control_points + curve_spec.curve_degree + 1);
    assert(curve_spec.data.size() == curve_spec.num_control_points * 4 + curve_spec.num_knots);

    auto curve = std::make_unique<NURBS<Scalar, dim, -1>>();

    curve->set_control_points(extract_control_points(curve_spec));
    curve->set_knots(extract_knots(curve_spec));
    curve->set_weights(extract_weights(curve_spec));
    curve->initialize();
    return curve;
}

template <typename Scalar, int dim>
std::unique_ptr<PatchBase<Scalar, dim>> load_Bezier_patch(const mshio::Patch& patch_spec)
{
    assert(patch_spec.num_control_points == (patch_spec.degree_u + 1) * (patch_spec.degree_v + 1));
    assert(patch_spec.num_u_knots == 0);
    assert(patch_spec.num_v_knots == 0);
    assert(!patch_spec.with_weights);

    auto patch = std::make_unique<BezierPatch<Scalar, dim, -1, -1>>();

    patch->set_control_grid(extract_control_points(patch_spec).template cast<Scalar>());
    patch->set_degree_u(static_cast<int>(patch_spec.degree_u));
    patch->set_degree_v(static_cast<int>(patch_spec.degree_v));
    patch->initialize();
    return patch;
}

template <typename Scalar, int dim>
std::unique_ptr<PatchBase<Scalar, dim>> load_rational_Bezier_patch(const mshio::Patch& patch_spec)
{
    assert(patch_spec.num_control_points == (patch_spec.degree_u + 1) * (patch_spec.degree_v + 1));
    assert(patch_spec.num_u_knots == 0);
    assert(patch_spec.num_v_knots == 0);
    assert(patch_spec.with_weights);

    auto patch = std::make_unique<RationalBezierPatch<Scalar, dim, -1, -1>>();

    patch->set_control_grid(extract_control_points(patch_spec).template cast<Scalar>());
    patch->set_weights(extract_weights(patch_spec).template cast<Scalar>());
    patch->set_degree_u(static_cast<int>(patch_spec.degree_u));
    patch->set_degree_v(static_cast<int>(patch_spec.degree_v));
    patch->initialize();
    return patch;
}

template <typename Scalar, int dim>
std::unique_ptr<PatchBase<Scalar, dim>> load_BSpline_patch(const mshio::Patch& patch_spec)
{
    assert(patch_spec.num_control_points == (patch_spec.num_u_knots - patch_spec.degree_u - 1) *
                                                (patch_spec.num_v_knots - patch_spec.degree_v - 1));
    assert(!patch_spec.with_weights);

    auto patch = std::make_unique<BSplinePatch<Scalar, dim, -1, -1>>();

    patch->set_control_grid(extract_control_points(patch_spec).template cast<Scalar>());
    patch->set_knots_u(extract_knots_u(patch_spec).template cast<Scalar>());
    patch->set_knots_v(extract_knots_v(patch_spec).template cast<Scalar>());
    patch->set_degree_u(static_cast<int>(patch_spec.degree_u));
    patch->set_degree_v(static_cast<int>(patch_spec.degree_v));
    patch->initialize();
    return patch;
}

template <typename Scalar, int dim>
std::unique_ptr<PatchBase<Scalar, dim>> load_NURBS_patch(const mshio::Patch& patch_spec)
{
    assert(patch_spec.num_control_points == (patch_spec.num_u_knots - patch_spec.degree_u - 1) *
                                                (patch_spec.num_v_knots - patch_spec.degree_v - 1));
    assert(patch_spec.with_weights);

    auto patch = std::make_unique<NURBSPatch<Scalar, dim, -1, -1>>();

    patch->set_control_grid(extract_control_points(patch_spec).template cast<Scalar>());
    patch->set_weights(extract_weights(patch_spec).template cast<Scalar>());
    patch->set_knots_u(extract_knots_u(patch_spec).template cast<Scalar>());
    patch->set_knots_v(extract_knots_v(patch_spec).template cast<Scalar>());
    patch->set_degree_u(static_cast<int>(patch_spec.degree_u));
    patch->set_degree_v(static_cast<int>(patch_spec.degree_v));
    patch->initialize();
    return patch;
}

template <typename Scalar, int dim>
auto load_curve(const mshio::Curve& curve_spec) -> std::unique_ptr<CurveBase<Scalar, dim>>
{
    switch (curve_spec.curve_type) {
    case static_cast<size_t>(CurveEnum::BEZIER): return load_Bezier_curve<Scalar, dim>(curve_spec);
    case static_cast<size_t>(CurveEnum::BSPLINE):
        return load_BSpline_curve<Scalar, dim>(curve_spec);
    case static_cast<size_t>(CurveEnum::RATIONAL_BEZIER):
        return load_rational_Bezier_curve<Scalar, dim>(curve_spec);
    case static_cast<size_t>(CurveEnum::NURBS): return load_NURBS_curve<Scalar, dim>(curve_spec);
    default:
        throw invalid_setting_error("Unknown curve type " + std::to_string(curve_spec.curve_type));
    }
}

template <typename Scalar, int dim>
auto load_patch(const mshio::Patch& patch_spec) -> std::unique_ptr<PatchBase<Scalar, dim>>
{
    switch (patch_spec.patch_type) {
    case static_cast<size_t>(PatchEnum::BEZIER): return load_Bezier_patch<Scalar, dim>(patch_spec);
    case static_cast<size_t>(PatchEnum::BSPLINE):
        return load_BSpline_patch<Scalar, dim>(patch_spec);
    case static_cast<size_t>(PatchEnum::RATIONAL_BEZIER):
        return load_rational_Bezier_patch<Scalar, dim>(patch_spec);
    case static_cast<size_t>(PatchEnum::NURBS): return load_NURBS_patch<Scalar, dim>(patch_spec);
    default:
        throw invalid_setting_error("Unknown patch type " + std::to_string(patch_spec.patch_type));
    }
}

template <typename Scalar, int dim = 3>
auto load_curves(const mshio::MshSpec& spec) -> std::vector<std::unique_ptr<CurveBase<Scalar, dim>>>
{
    static_assert(dim == 3, "Msh format only support 3D curves.");
    std::vector<std::unique_ptr<CurveBase<Scalar, dim>>> curves;
    curves.reserve(spec.curves.size());

    for (const auto& curve_spec : spec.curves) {
        curves.push_back(load_curve<Scalar, dim>(curve_spec));
    }

    return curves;
}

template <typename Scalar, int dim = 3>
auto load_patches(const mshio::MshSpec& spec)
    -> std::vector<std::unique_ptr<PatchBase<Scalar, dim>>>
{
    static_assert(dim == 3, "Msh format only support 3D patches.");
    std::vector<std::unique_ptr<PatchBase<Scalar, dim>>> patches;
    patches.reserve(spec.patches.size());

    for (const auto& patch_spec : spec.patches) {
        patches.push_back(load_patch<Scalar, dim>(patch_spec));
    }

    return patches;
}

} // namespace internal

template<typename Scalar, int dim=3>
auto load_msh(const std::string& filename)
    -> std::tuple<std::vector<std::unique_ptr<CurveBase<Scalar, dim>>>,
        std::vector<std::unique_ptr<PatchBase<Scalar, dim>>>>
{
    const auto spec = mshio::load_msh(filename);
    auto curves = internal::load_curves<Scalar, dim>(spec);
    auto patches = internal::load_patches<Scalar, dim>(spec);
    return {std::move(curves), std::move(patches)};
}

} // namespace nanospline
