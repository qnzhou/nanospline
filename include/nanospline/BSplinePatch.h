#pragma once

#include <nanospline/BSpline.h>
#include <nanospline/PatchBase.h>

using std::vector;

namespace nanospline {

template <typename _Scalar, int _dim = 3, int _degree_u = 3, int _degree_v = 3>
class BSplinePatch final : public PatchBase<_Scalar, _dim>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static_assert(_dim > 0, "Dimension must be positive.");

    using Base = PatchBase<_Scalar, _dim>;
    using ThisType = BSplinePatch<_Scalar, _dim, _degree_u, _degree_v>;
    using Scalar = typename Base::Scalar;
    using UVPoint = typename Base::UVPoint;
    using Point = typename Base::Point;
    using ControlGrid = typename Base::ControlGrid;
    using KnotVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using IsoCurveU = BSpline<Scalar, _dim, _degree_u>;
    using IsoCurveV = BSpline<Scalar, _dim, _degree_v>;

public:
    static BSplinePatch<_Scalar, _dim, _degree_u, _degree_v> ZeroPatch()
    {
        BSplinePatch<_Scalar, _dim, _degree_u, _degree_v> patch;
        const int degree_u = _degree_u > 0 ? _degree_u : 0;
        const int degree_v = _degree_v > 0 ? _degree_v : 0;
        patch.set_degree_u(degree_u);
        patch.set_degree_v(degree_v);
        ControlGrid grid((degree_u + 1) * (degree_v + 1), _dim);
        grid.setZero();
        patch.swap_control_grid(grid);
        KnotVector knots_u(2 * (degree_u + 1));
        knots_u.segment(0, degree_u + 1).setZero();
        knots_u.segment(degree_u + 1, degree_u + 1).setOnes();
        KnotVector knots_v(2 * (degree_v + 1));
        knots_v.segment(0, degree_v + 1).setZero();
        knots_v.segment(degree_v + 1, degree_v + 1).setOnes();
        patch.set_knots_u(knots_u);
        patch.set_knots_v(knots_v);
        patch.initialize();
        return patch;
    }

public:
    BSplinePatch()
    {
        Base::set_degree_u(_degree_u);
        Base::set_degree_v(_degree_v);
    }

    BSplinePatch(const ThisType& other)
        : PatchBase<_Scalar, _dim>(other)
        , m_knots_u(other.m_knots_u)
        , m_knots_v(other.m_knots_v)
        , m_du_patch(nullptr)
        , m_dv_patch(nullptr)
    {}

    BSplinePatch(ThisType&& other)
        : PatchBase<_Scalar, _dim>(other)
        , m_knots_u(std::move(other.m_knots_u))
        , m_knots_v(std::move(other.m_knots_v))
        , m_du_patch(nullptr)
        , m_dv_patch(nullptr)
    {}

    ThisType& operator=(const ThisType& other)
    {
        Base::m_control_grid = other.m_control_grid;
        Base::m_degree_u = other.m_degree_u;
        Base::m_degree_v = other.m_degree_v;
        m_knots_u = other.m_knots_u;
        m_knots_v = other.m_knots_v;
        return *this;
    }

    ThisType& operator=(ThisType&& other)
    {
        swap_control_grid(other.m_control_grid);
        Base::m_degree_u = other.m_degree_u;
        Base::m_degree_v = other.m_degree_v;
        m_knots_u.swap(other.m_knots_u);
        m_knots_v.swap(other.m_knots_v);
        return *this;
    }

public:
    Point evaluate(Scalar u, Scalar v) const override
    {
        if (Base::m_degree_v > Base::m_degree_u) {
            auto iso_curve_v = compute_iso_curve_v(u);
            return iso_curve_v.evaluate(v);
        } else {
            auto iso_curve_u = compute_iso_curve_u(v);
            return iso_curve_u.evaluate(u);
        }
    }

    Point evaluate_derivative_u(Scalar u, Scalar v) const override
    {
        auto iso_curve_u = compute_iso_curve_u(v);
        return iso_curve_u.evaluate_derivative(u);
    }

    Point evaluate_derivative_v(Scalar u, Scalar v) const override
    {
        auto iso_curve_v = compute_iso_curve_v(u);
        return iso_curve_v.evaluate_derivative(v);
    }

    Point evaluate_2nd_derivative_uu(Scalar u, Scalar v) const override
    {
        auto iso_curve_u = compute_iso_curve_u(v);
        return iso_curve_u.evaluate_2nd_derivative(u);
    }

    Point evaluate_2nd_derivative_vv(Scalar u, Scalar v) const override
    {
        auto iso_curve_v = compute_iso_curve_v(u);
        return iso_curve_v.evaluate_2nd_derivative(v);
    }

    Point evaluate_2nd_derivative_uv(Scalar u, Scalar v) const override
    {
        auto duv_patch = compute_duv_patch();
        return duv_patch.evaluate(u, v);
    }

    void initialize() override
    {
        const auto num_v_knots = m_knots_v.size();
        const auto num_u_knots = m_knots_u.size();
        const int degree_u = Base::get_degree_u();
        const int degree_v = Base::get_degree_v();
        if (Base::m_control_grid.rows() !=
            (num_u_knots - degree_u - 1) * (num_v_knots - degree_v - 1)) {
            throw invalid_setting_error("Control grid size mismatch uv degrees");
        }
    }

    Scalar get_u_lower_bound() const override
    {
        const auto degree_u = Base::get_degree_u();
        return m_knots_u[degree_u];
    }

    Scalar get_v_lower_bound() const override
    {
        const auto degree_v = Base::get_degree_v();
        return m_knots_v[degree_v];
    }

    Scalar get_u_upper_bound() const override
    {
        const auto num_knots = get_num_knots_u();
        const auto degree_u = Base::get_degree_u();
        return m_knots_u[num_knots - degree_u - 1];
    }

    Scalar get_v_upper_bound() const override
    {
        const auto num_knots = get_num_knots_v();
        const auto degree_v = Base::get_degree_v();
        return m_knots_v[num_knots - degree_v - 1];
    }

    UVPoint get_control_point_preimage(int i, int j) const override
    {
        // Computes the Greville abscissae for control point (i,j)
        UVPoint preimage;
        preimage.setZero();
        for (int k = 0; k < Base::get_degree_u(); k++) {
            preimage(0) += m_knots_u[1 + i + k];
        }
        preimage(0) = preimage(0) / Scalar(Base::get_degree_u());
        for (int k = 0; k < Base::get_degree_v(); k++) {
            preimage(1) += m_knots_v[1 + j + k];
        }
        preimage(1) = preimage(1) / Scalar(Base::get_degree_v());
        return preimage;
    }

public:
    const KnotVector& get_knots_u() const { return m_knots_u; }

    const KnotVector& get_knots_v() const { return m_knots_v; }

    template <typename Derived>
    void set_knots_u(const Eigen::PlainObjectBase<Derived>& knots)
    {
        m_knots_u = knots;
    }

    template <typename Derived>
    void set_knots_v(const Eigen::PlainObjectBase<Derived>& knots)
    {
        m_knots_v = knots;
    }

    template <typename Derived>
    void set_knots_u(Eigen::PlainObjectBase<Derived>&& knots)
    {
        m_knots_u.swap(knots);
    }

    template <typename Derived>
    void set_knots_v(Eigen::PlainObjectBase<Derived>&& knots)
    {
        m_knots_v.swap(knots);
    }

    IsoCurveU compute_iso_curve_u(Scalar v) const
    {
        const int num_ctrl_pts_u = num_control_points_u();
        typename IsoCurveU::ControlPoints control_points_u(num_ctrl_pts_u, _dim);

        std::vector<IsoCurveV> iso_curves_v = get_all_iso_curves_v();

        for (int i = 0; i < num_ctrl_pts_u; i++) {
            const auto& iso_curve = iso_curves_v[static_cast<size_t>(i)];
            control_points_u.row(i) = iso_curve.evaluate(v);
        }

        IsoCurveU iso_curve_u;
        iso_curve_u.set_control_points(std::move(control_points_u));
        iso_curve_u.set_knots(m_knots_u);
        return iso_curve_u;
    }

    IsoCurveV compute_iso_curve_v(Scalar u) const
    {
        const int num_ctrl_pts_v = num_control_points_v();
        typename IsoCurveV::ControlPoints control_points_v(num_ctrl_pts_v, _dim);
        std::vector<IsoCurveU> iso_curves_u = get_all_iso_curves_u();

        for (int j = 0; j < num_ctrl_pts_v; j++) {
            const auto& iso_curve = iso_curves_u[static_cast<size_t>(j)];
            control_points_v.row(j) = iso_curve.evaluate(u);
        }

        IsoCurveV iso_curve_v;
        iso_curve_v.set_control_points(std::move(control_points_v));
        iso_curve_v.set_knots(m_knots_v);
        return iso_curve_v;
    }

    BSplinePatch<_Scalar, _dim, -1, -1> compute_du_patch() const
    {
        const int num_u_knots = get_num_knots_u();
        const int num_v_knots = get_num_knots_v();
        const int degree_u = Base::get_degree_u();
        const int degree_v = Base::get_degree_v();

        if (degree_u == 0) {
            return BSplinePatch<_Scalar, _dim, -1, -1>::ZeroPatch();
        }

        ControlGrid du_grid((num_u_knots - degree_u - 2) * (num_v_knots - degree_v - 1), _dim);
        for (int i = 0; i < num_u_knots - degree_u - 2; i++) {
            for (int j = 0; j < num_v_knots - degree_v - 1; j++) {
                const int row_id = i * (num_v_knots - degree_v - 1) + j;
                du_grid.row(row_id) = degree_u *
                                      (get_control_point(i + 1, j) - get_control_point(i, j)) /
                                      (m_knots_u[i + degree_u + 1] - m_knots_u[i + 1]);
            }
        }

        BSplinePatch<_Scalar, _dim, -1, -1> du_patch;
        du_patch.set_degree_u(degree_u - 1);
        du_patch.set_degree_v(degree_v);
        du_patch.swap_control_grid(du_grid);
        du_patch.set_knots_u(m_knots_u.segment(1, num_u_knots - 2).eval());
        du_patch.set_knots_v(m_knots_v);
        du_patch.initialize();
        return du_patch;
    }

    BSplinePatch<_Scalar, _dim, -1, -1> compute_dv_patch() const
    {
        const int num_u_knots = get_num_knots_u();
        const int num_v_knots = get_num_knots_v();
        const int degree_u = Base::get_degree_u();
        const int degree_v = Base::get_degree_v();

        if (degree_v == 0) {
            return BSplinePatch<_Scalar, _dim, -1, -1>::ZeroPatch();
        }

        ControlGrid dv_grid((num_u_knots - degree_u - 1) * (num_v_knots - degree_v - 2), _dim);
        for (int i = 0; i < num_u_knots - degree_u - 1; i++) {
            for (int j = 0; j < num_v_knots - degree_v - 2; j++) {
                const int row_id = i * (num_v_knots - degree_v - 2) + j;
                dv_grid.row(row_id) = degree_v *
                                      (get_control_point(i, j + 1) - get_control_point(i, j)) /
                                      (m_knots_v[j + degree_v + 1] - m_knots_v[j + 1]);
            }
        }

        BSplinePatch<_Scalar, _dim, -1, -1> dv_patch;
        dv_patch.set_degree_u(degree_u);
        dv_patch.set_degree_v(degree_v - 1);
        dv_patch.swap_control_grid(dv_grid);
        dv_patch.set_knots_u(m_knots_u);
        dv_patch.set_knots_v(m_knots_v.segment(1, num_v_knots - 2).eval());
        dv_patch.initialize();
        return dv_patch;
    }

    BSplinePatch<_Scalar, _dim, -1, -1> compute_duv_patch() const
    {
        return compute_du_patch().compute_dv_patch();
    }


    Point get_control_point(int ui, int vj) const override
    {
        // const auto degree_v = Base::get_degree_v();
        // const auto row_size = m_knots_v.size() - degree_v - 1;
        // return Base::m_control_grid.row(ui*row_size + vj);
        return Base::m_control_grid.row(Base::control_point_linear_index(ui, vj));
    }

    int num_control_points_u() const override
    {
        const auto num_u_knots = get_num_knots_u();
        const int degree_u = Base::get_degree_u();
        return num_u_knots - degree_u - 1;
    }

    int num_control_points_v() const override
    {
        const int num_v_knots = get_num_knots_v();
        const int degree_v = Base::get_degree_v();
        return num_v_knots - degree_v - 1;
    }

private:
    // Hopefully we won't have more than 2 billion knots...
    // If so this will break.
    int get_num_knots_u() const { return static_cast<int>(m_knots_u.rows()); }

    int get_num_knots_v() const { return static_cast<int>(m_knots_v.rows()); }

    // Construct the implicit isocurves determined by each rows of control points
    // i.e. fixed values of v. This copies these rows into
    // individual matrices of control points and returns the resulting
    // curves
    std::vector<IsoCurveU> get_all_iso_curves_u() const
    {
        const int num_control_pts_u = num_control_points_u();
        const int num_control_pts_v = num_control_points_v();

        typename IsoCurveV::ControlPoints control_points_v(num_control_pts_v, _dim);

        std::vector<IsoCurveU> iso_curves;
        iso_curves.reserve((size_t)num_control_pts_v);
        for (int vj = 0; vj < num_control_pts_v; vj++) {
            typename IsoCurveU::ControlPoints control_points_u(num_control_pts_u, _dim);
            for (int ui = 0; ui < num_control_pts_u; ui++) {
                control_points_u.row(ui) = get_control_point(ui, vj);
            }

            IsoCurveU iso_curve_u;
            iso_curve_u.set_control_points(std::move(control_points_u));
            iso_curve_u.set_knots(m_knots_u);
            iso_curves.push_back(std::move(iso_curve_u));
        }
        return iso_curves;
    }

    // Construct the implicit isocurves determined by each column of control points
    // i.e. fixed values of u. This copies these columns into
    // individual matrices of control points and returns the resulting
    // curves
    std::vector<IsoCurveV> get_all_iso_curves_v() const
    {
        const int num_control_pts_u = num_control_points_u();
        const int num_control_pts_v = num_control_points_v();

        typename IsoCurveU::ControlPoints control_points_u(num_control_pts_u, _dim);
        std::vector<IsoCurveV> iso_curves;
        iso_curves.reserve((size_t)num_control_pts_u);
        for (int ui = 0; ui < num_control_pts_u; ui++) {
            typename IsoCurveV::ControlPoints control_points_v(num_control_pts_v, _dim);

            for (int vj = 0; vj < num_control_pts_v; vj++) {
                control_points_v.row(vj) = get_control_point(ui, vj);
            }

            IsoCurveV iso_curve_v;
            iso_curve_v.set_control_points(std::move(control_points_v));
            iso_curve_v.set_knots(m_knots_v);
            iso_curves.push_back(std::move(iso_curve_v));
        }
        return iso_curves;
    }

public:
    // Split a patch into two patches along the vertical line at u
    std::vector<ThisType> split_u(Scalar u) const
    {
        if (!Base::in_domain_u(u)) {
            invalid_setting_error("Parameter not inside of the domain.");
        }
        if (Base::is_endpoint_u(u)) { // no split required
            return std::vector<ThisType>{*this};
        }

        const size_t num_split_patches = 2; // splitting one patch into two

        // split each isocurve
        std::vector<IsoCurveU> iso_curves_u = get_all_iso_curves_u();
        std::vector<std::vector<IsoCurveU>> split_iso_curves;
        split_iso_curves.reserve(static_cast<size_t>(num_control_points_v()));

        for (int vj = 0; vj < num_control_points_v(); vj++) {
            IsoCurveU& iso_curve = iso_curves_u[static_cast<size_t>(vj)];
            std::vector<IsoCurveU> iso_curve_parts = nanospline::split(iso_curve, u);
            split_iso_curves.push_back(std::move(iso_curve_parts));
        }

        // all split curves have the same knot sequence, degree, and number of
        // control points; just grab the values from one of them
        const std::vector<IsoCurveU>& reference_split_curves = split_iso_curves[0];

        // Initialize control grids of final split patches
        std::vector<ControlGrid> split_control_pts_u;
        for (const auto& ref_curve : reference_split_curves) {
            int num_ctrl_pts = static_cast<int>(ref_curve.get_control_points().rows());

            split_control_pts_u.push_back(ControlGrid(num_ctrl_pts * num_control_points_v(), _dim));
        }
        for (int vj = 0; vj < num_control_points_v(); vj++) {
            const std::vector<IsoCurveU>& iso_curve_parts =
                split_iso_curves[static_cast<size_t>(vj)];
            // Copy control points of each split curve to its proper place in
            // the final control grid
            for (size_t ci = 0; ci < num_split_patches; ci++) {
                const auto& ctrl_pts = iso_curve_parts[ci].get_control_points();
                const int num_split_ctrl_pts_u = static_cast<int>(ctrl_pts.rows());

                for (int ui = 0; ui < num_split_ctrl_pts_u; ui++) {
                    int index = Base::control_point_linear_index(ui, vj);
                    split_control_pts_u[ci].row(index) = ctrl_pts.row(ui);
                }
            }
        }

        // Initialize resulting split curves
        std::vector<ThisType> results(num_split_patches, ThisType());
        for (size_t ci = 0; ci < num_split_patches; ci++) {
            const IsoCurveU& ref_curve = reference_split_curves[ci];
            ThisType split_patch;
            split_patch.set_control_grid(split_control_pts_u[ci]);
            split_patch.set_knots_u(ref_curve.get_knots());
            split_patch.set_knots_v(get_knots_v());

            if (_degree_u < 0 || _degree_v < 0) {
                split_patch.set_degree_u(ref_curve.get_degree());
                split_patch.set_degree_v(Base::get_degree_v());
            }
            results[ci] = split_patch;
        }
        return results;
    }

    std::vector<ThisType> split_v(Scalar v) const
    {
        if (!Base::in_domain_v(v)) {
            throw invalid_setting_error("Parameter not inside of the domain.");
        }
        if (Base::is_endpoint_v(v)) { // no split required
            return std::vector<ThisType>{*this};
        }

        const size_t num_split_patches = 2; // splitting one patch into two
        // split each isocurve
        std::vector<IsoCurveV> iso_curves_v = get_all_iso_curves_v();
        std::vector<std::vector<IsoCurveV>> split_iso_curves;
        split_iso_curves.reserve(static_cast<size_t>(num_control_points_u()));

        for (int ui = 0; ui < num_control_points_u(); ui++) {
            IsoCurveV iso_curve = iso_curves_v[static_cast<size_t>(ui)];
            std::vector<IsoCurveV> iso_curve_parts = nanospline::split(iso_curve, v);
            split_iso_curves.push_back(std::move(iso_curve_parts));
        }

        // all split curves have the same knot sequence, degree, and number of
        // control points; just grab the values from one of them
        const std::vector<IsoCurveV>& reference_split_curves = split_iso_curves[0];

        // Initialize control grids of final split patches
        std::vector<ControlGrid> split_control_pts_v;
        for (const auto& ref_curve : reference_split_curves) {
            int num_ctrl_pts = static_cast<int>(ref_curve.get_control_points().rows());

            split_control_pts_v.push_back(ControlGrid(num_ctrl_pts * num_control_points_u(), _dim));
        }
        for (int ui = 0; ui < num_control_points_u(); ui++) {
            const std::vector<IsoCurveV>& iso_curve_parts =
                split_iso_curves[static_cast<size_t>(ui)];
            // Copy control points of each split curve to its proper place in
            // the final control grid
            for (size_t ci = 0; ci < num_split_patches; ci++) {
                const auto& ctrl_pts = iso_curve_parts[ci].get_control_points();
                const int num_split_ctrl_pts_v = static_cast<int>(ctrl_pts.rows());

                for (int vj = 0; vj < num_split_ctrl_pts_v; vj++) {
                    // stride size is num_split_ctrl_pts_v, NOT num_control_points_v()
                    // control grid is V-major and splitting a spline can result in a
                    // varied number of control points.
                    int index = ui * (num_split_ctrl_pts_v) + vj;

                    split_control_pts_v[ci].row(index) = ctrl_pts.row(vj);
                }
            }
        }

        // Initialize resulting split patches
        vector<ThisType> results(num_split_patches, ThisType());
        for (size_t ci = 0; ci < num_split_patches; ci++) {
            const IsoCurveV& ref_curve = reference_split_curves[ci];
            ThisType split_patch;
            split_patch.set_control_grid(split_control_pts_v[ci]);
            split_patch.set_knots_v(ref_curve.get_knots());
            split_patch.set_knots_u(get_knots_u());

            if (_degree_u < 0 || _degree_v < 0) {
                split_patch.set_degree_v(ref_curve.get_degree());
                split_patch.set_degree_u(Base::get_degree_u());
            }
            results[ci] = split_patch;
        }
        return results;
    }

    std::vector<ThisType> split(Scalar u, Scalar v) const
    {
        if (!Base::in_domain(u, v)) {
            throw invalid_setting_error("Parameter not inside of the domain.");
        }
        if (Base::is_endpoint_u(u) && Base::is_endpoint_v(v)) { // no splitting required
            return std::vector<ThisType>{*this};

        } else if (Base::is_endpoint_u(u)) { // no need to split along u
            return split_v(v);

        } else if (Base::is_endpoint_v(v)) { // no need to split along v
            return split_u(u);

        } else {
            // 1. split in u direction
            std::vector<ThisType> split_patches = split_u(u);
            ThisType less_than_u = split_patches[0];
            ThisType greater_than_u = split_patches[1];

            // 2. Split each subpatch patch at v
            std::vector<ThisType> split_patches_less_than_u = less_than_u.split_v(v);

            std::vector<ThisType> split_patches_greater_than_u = greater_than_u.split_v(v);

            // 3. Reorder. The patch is split into 4 patches returned in the
            // following order:
            //
            //   |-----------|
            //   |  1  |  3  |
            // ^ |-----*-----| *= split point (u,v)
            // | |  0  |  2  |
            // v |-----------|
            //    u ->

            ThisType less_than_u_less_than_v = split_patches_less_than_u[0];
            ThisType less_than_u_greater_than_v = split_patches_less_than_u[1];
            ThisType greater_than_u_less_than_v = split_patches_greater_than_u[0];
            ThisType greater_than_u_greater_than_v = split_patches_greater_than_u[1];

            return {less_than_u_less_than_v,
                less_than_u_greater_than_v,
                greater_than_u_less_than_v,
                greater_than_u_greater_than_v};
        }
    }

    ThisType subpatch(Scalar u_min, Scalar u_max, Scalar v_min, Scalar v_max) const
    {
        if (u_min > u_max) {
            throw invalid_setting_error("u_min must be smaller than u_max");
        }
        if (!Base::in_domain_u(u_min) || !Base::in_domain_u(u_max)) {
            throw invalid_setting_error("Invalid range in subcurve: u");
        }

        if (v_min > v_max) {
            throw invalid_setting_error("v_min must be smaller than v_max");
        }
        if (!Base::in_domain_v(v_min) || !Base::in_domain_v(v_max)) {
            throw invalid_setting_error("Invalid range in subcurve: v");
        }

        // Note: conditionals are for bound checking to avoid redundant splits
        // 1. Split patch in u direction at u_min; take the patch on >= u_min
        ThisType greater_than_umin = Base::is_endpoint_u(u_min) ? *this : split_u(u_min)[1];

        // 2. Split patch on >= u_min at u_max in u dir; take the patch <=
        // u_max
        ThisType greater_than_umin_less_umax =
            Base::is_endpoint_u(u_max) ? greater_than_umin : greater_than_umin.split_u(u_max)[0];

        // 3. Split patch on u_min <= u <= umax at v_min  in the v dir, take
        // the patch >= v_min
        ThisType greater_than_vmin = Base::is_endpoint_v(v_min)
                                         ? greater_than_umin_less_umax
                                         : greater_than_umin_less_umax.split_v(v_min)[1];

        // 4. Split patch  at v_max in the v dir, take the patch <= v_max
        ThisType greater_than_vmin_less_vmax =
            Base::is_endpoint_v(v_max) ? greater_than_vmin : greater_than_vmin.split_v(v_max)[0];

        return greater_than_vmin_less_vmax;
    }

    UVPoint approximate_inverse_evaluate(const Point& p,
        const int num_samples,
        const Scalar min_u,
        const Scalar max_u,
        const Scalar min_v,
        const Scalar max_v,
        const int level = 15) const override
    {
        // Only two control points at the endpoints, so finding the closest
        // point doesn't restrict the search at all; default to the parent
        // class function based on sampling where resolution isn't an issue
        if (Base::get_degree_u() < 2 || Base::get_degree_v() < 2) {
            return Base::approximate_inverse_evaluate(
                p, num_samples, min_u, max_u, min_v, max_v, level);
        }

        // 1. find closest control point
        auto closest_control_pt_index = Base::find_closest_control_point(p);
        int i_min = closest_control_pt_index.first;
        int j_min = closest_control_pt_index.second;

        if (level <= 0) {
            return get_control_point_preimage(i_min, j_min);

        } else {
            // 2. Control points c_{i+/-1,j+/-1} bound the domain containing our
            // desired initial guess; find subdomain corresponding to control
            // point subdomain boundary
            // Conditionals for bound checking
            UVPoint uv_min = get_control_point_preimage(
                i_min > 0 ? i_min - 1 : i_min, j_min > 0 ? j_min - 1 : j_min);
            UVPoint uv_max =
                get_control_point_preimage(i_min < num_control_points_u() - 1 ? i_min + 1 : i_min,
                    j_min < num_control_points_v() - 1 ? j_min + 1 : j_min);

            // 3. split a subcurve to find closest ctrl points on subdomain
            ThisType patch = subpatch(uv_min(0), uv_max(0), uv_min(1), uv_max(1));

            // 4. repeat recursively
            UVPoint uv = patch.approximate_inverse_evaluate(
                p, num_samples, uv_min(0), uv_max(0), uv_min(1), uv_max(1), level - 1);

            // no need to remap coordinates for splines
            return uv;
        }
    }

    static Eigen::MatrixXd form_least_squares_matrix(int num_control_pts_u,
        int num_control_pts_v,
        Eigen::MatrixXd knots_u,
        Eigen::MatrixXd knots_v,
        Eigen::MatrixXd parameters)
    {
        const int num_control_pts = num_control_pts_u * num_control_pts_v;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> basis_func_control_pts(num_control_pts, 1);
        BSplinePatch<Scalar, 1, _degree_u, _degree_v> basis_function;
        basis_func_control_pts.setZero();
        basis_function.set_knots_u(knots_u);
        basis_function.set_knots_v(knots_v);

        // Suppose we are fitting samples p_0, ..., p_n of a function f, with
        // parameter values (u_0,v_0) ..., (u_n,v_n). If B_q^n(u,v) is the
        // jth basis function value at (u,v), then
        // least_squares_matrix(i,q) = B_q^n(u_i, v_i),
        // where q is the linearized index over basis elements:
        // q = i*num_control_pts_v + j
        const int num_constraints = int(parameters.rows());
        Eigen::MatrixXd least_squares_matrix =
            Eigen::MatrixXd::Zero(num_constraints, num_control_pts);

        for (int j = 0; j < num_control_pts_u; j++) {
            for (int k = 0; k < num_control_pts_v; k++) {
                int index = j * num_control_pts_v + k; // TODO bug prone
                basis_func_control_pts.row(index) << 1.;
                basis_function.set_control_grid(basis_func_control_pts);
                basis_function.initialize();
                for (int i = 0; i < num_constraints; i++) {
                    Scalar u = parameters(i, 0);
                    Scalar v = parameters(i, 1);
                    Scalar bezier_value = basis_function.evaluate(u, v)(0);
                    least_squares_matrix(i, index) = bezier_value;
                }
                basis_func_control_pts.row(index) << 0.;
            }
        }
        return least_squares_matrix;
    }

    static ThisType fit(Eigen::MatrixXd parameters,
        Eigen::MatrixXd values,
        int num_control_pts_u,
        int num_control_pts_v,
        Eigen::MatrixXd knots_u,
        Eigen::MatrixXd knots_v)
    {
        ThisType least_squares_fit;
        least_squares_fit.set_knots_u(knots_u);
        least_squares_fit.set_knots_v(knots_v);

        assert(parameters.rows() == values.rows());
        assert(parameters.cols() == 2);
        assert(values.cols() == least_squares_fit.get_dim());
        // Form least squares matrix
        Eigen::MatrixXd least_squares_matrix = form_least_squares_matrix(
            num_control_pts_u, num_control_pts_v, knots_u, knots_v, parameters);

        // 2. Least squares solve via SVD
        Eigen::MatrixXd fit_control_points =
            least_squares_matrix.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(values);

        // 3. Store control points in least_squares_fit and return
        least_squares_fit.set_control_grid(fit_control_points);
        least_squares_fit.set_knots_u(knots_u);
        least_squares_fit.set_knots_v(knots_v);
        least_squares_fit.initialize();
        return least_squares_fit;
    }

    void deform(Eigen::MatrixXd parameters, Eigen::MatrixXd changes_in_values)
    {
        ThisType least_squares_fit = ThisType::fit(parameters,
            changes_in_values,
            num_control_points_u(),
            num_control_points_v(),
            get_knots_u(),
            get_knots_v());

        ControlGrid changes_in_control_points = least_squares_fit.get_control_grid();
        ControlGrid updated_control_points = Base::m_control_grid + changes_in_control_points;
        Base::set_control_grid(updated_control_points);
        initialize();
    }

protected:
    KnotVector m_knots_u;
    KnotVector m_knots_v;

private:
    std::unique_ptr<BSplinePatch<_Scalar, _dim, -1, -1>> m_du_patch;
    std::unique_ptr<BSplinePatch<_Scalar, _dim, -1, -1>> m_dv_patch;
};

} // namespace nanospline
