#pragma once

#include <nanospline/Bezier.h>
#include <nanospline/PatchBase.h>

namespace nanospline {

template <typename _Scalar, int _dim = 3, int _degree_u = 3, int _degree_v = 3>
class BezierPatch final : public PatchBase<_Scalar, _dim>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static_assert(_dim > 0, "Dimension must be positive.");

    using Base = PatchBase<_Scalar, _dim>;
    using ThisType = BezierPatch<_Scalar, _dim, _degree_u, _degree_v>;
    using Scalar = typename Base::Scalar;
    using Point = typename Base::Point;
    using UVPoint = typename Base::UVPoint;
    using ControlGrid = typename Base::ControlGrid;
    using IsoCurveU = Bezier<Scalar, _dim, _degree_u>;
    using IsoCurveV = Bezier<Scalar, _dim, _degree_v>;

public:
    static BezierPatch<_Scalar, _dim, _degree_u, _degree_v> ZeroPatch()
    {
        BezierPatch<_Scalar, _dim, _degree_u, _degree_v> patch;
        patch.set_degree_u(_degree_u > 0 ? _degree_u : 0);
        patch.set_degree_v(_degree_v > 0 ? _degree_v : 0);
        ControlGrid grid((patch.get_degree_u() + 1) * (patch.get_degree_v() + 1), _dim);
        grid.setZero();
        patch.swap_control_grid(grid);
        patch.initialize();
        return patch;
    }

public:
    BezierPatch()
    {
        Base::set_degree_u(_degree_u);
        Base::set_degree_v(_degree_v);
    }

public:
    Point evaluate(Scalar u, Scalar v) const override
    {
        auto iso_curve_v = compute_iso_curve_v(u);
        return iso_curve_v.evaluate(v);
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
        const int degree_u = Base::get_degree_u();
        const int degree_v = Base::get_degree_v();
        if (Base::m_control_grid.rows() != (degree_u + 1) * (degree_v + 1)) {
            throw invalid_setting_error("Control grid size mismatch uv degrees");
        }
    }

    Scalar get_u_lower_bound() const override { return 0.0; }

    Scalar get_v_lower_bound() const override { return 0.0; }

    Scalar get_u_upper_bound() const override { return 1.0; }

    Scalar get_v_upper_bound() const override { return 1.0; }

    UVPoint get_control_point_preimage(int i, int j) const override
    {
        UVPoint preimage;
        preimage << Scalar(i) / Scalar(Base::get_degree_u()),
            Scalar(j) / Scalar(Base::get_degree_v());
        return preimage;
    }


public:
    IsoCurveU compute_iso_curve_u(Scalar v) const
    {
        typename IsoCurveU::ControlPoints control_points_u(num_control_points_u(), _dim);

        std::vector<IsoCurveV> iso_curves_v = get_all_iso_curves_v();
        for (int i = 0; i < num_control_points_u(); i++) {
            IsoCurveV iso_curve_v = iso_curves_v[size_t(i)];
            control_points_u.row(i) = iso_curve_v.evaluate(v);
        }

        IsoCurveU iso_curve_u;
        iso_curve_u.set_control_points(std::move(control_points_u));
        return iso_curve_u;
    }

    IsoCurveV compute_iso_curve_v(Scalar u) const
    {
        typename IsoCurveV::ControlPoints control_points_v(num_control_points_v(), _dim);

        std::vector<IsoCurveU> iso_curves_u = get_all_iso_curves_u();
        for (int i = 0; i < num_control_points_v(); i++) {
            IsoCurveU iso_curve = iso_curves_u[size_t(i)];
            control_points_v.row(i) = iso_curve.evaluate(u);
        }
        IsoCurveV iso_curve_v;
        iso_curve_v.set_control_points(std::move(control_points_v));
        return iso_curve_v;
    }

    BezierPatch<_Scalar, _dim, -1, -1> compute_du_patch() const
    {
        const int degree_u = Base::get_degree_u();
        const int degree_v = Base::get_degree_v();

        if (degree_u == 0) {
            return BezierPatch<_Scalar, _dim, -1, -1>::ZeroPatch();
        }

        ControlGrid du_grid(degree_u * (degree_v + 1), _dim);
        for (int i = 0; i <= degree_u - 1; i++) {
            for (int j = 0; j <= degree_v; j++) {
                const int row_id = i * (degree_v + 1) + j;
                du_grid.row(row_id) =
                    degree_u * (get_control_point(i + 1, j) - get_control_point(i, j));
            }
        }

        BezierPatch<_Scalar, _dim, -1, -1> du_patch;
        du_patch.set_degree_u(degree_u - 1);
        du_patch.set_degree_v(degree_v);
        du_patch.swap_control_grid(du_grid);
        du_patch.initialize();
        return du_patch;
    }

    BezierPatch<_Scalar, _dim, -1, -1> compute_dv_patch() const
    {
        const int degree_u = Base::get_degree_u();
        const int degree_v = Base::get_degree_v();

        if (degree_v == 0) {
            return BezierPatch<_Scalar, _dim, -1, -1>::ZeroPatch();
        }

        ControlGrid dv_grid((degree_u + 1) * degree_v, _dim);
        for (int i = 0; i <= degree_u; i++) {
            for (int j = 0; j <= degree_v - 1; j++) {
                const int row_id = i * degree_v + j;
                dv_grid.row(row_id) =
                    degree_v * (get_control_point(i, j + 1) - get_control_point(i, j));
            }
        }

        BezierPatch<_Scalar, _dim, -1, -1> dv_patch;
        dv_patch.set_degree_u(degree_u);
        dv_patch.set_degree_v(degree_v - 1);
        dv_patch.swap_control_grid(dv_grid);
        dv_patch.initialize();
        return dv_patch;
    }

    BezierPatch<_Scalar, _dim, -1, -1> compute_duv_patch() const
    {
        const int degree_u = Base::get_degree_u();
        const int degree_v = Base::get_degree_v();

        if (degree_u == 0 || degree_v == 0) {
            return BezierPatch<_Scalar, _dim, -1, -1>::ZeroPatch();
        }

        ControlGrid duv_grid(degree_u * degree_v, _dim);
        for (int i = 0; i <= degree_u - 1; i++) {
            for (int j = 0; j <= degree_v - 1; j++) {
                const int row_id = i * degree_v + j;
                const Point pu0 =
                    degree_u * (get_control_point(i + 1, j) - get_control_point(i, j));
                const Point pu1 =
                    degree_u * (get_control_point(i + 1, j + 1) - get_control_point(i, j + 1));
                duv_grid.row(row_id) = degree_v * (pu1 - pu0);
            }
        }

        BezierPatch<_Scalar, _dim, -1, -1> duv_patch;
        duv_patch.set_degree_u(degree_u - 1);
        duv_patch.set_degree_v(degree_v - 1);
        duv_patch.swap_control_grid(duv_grid);
        duv_patch.initialize();
        return duv_patch;
    }


    Point get_control_point(int ui, int vj) const override
    {
        return Base::m_control_grid.row(Base::control_point_linear_index(ui, vj));
    }
    int num_control_points_u() const override
    {
        const int degree_u = Base::get_degree_u();
        return degree_u + 1;
    }
    int num_control_points_v() const override
    {
        const int degree_v = Base::get_degree_v();
        return degree_v + 1;
    }

private:
    // Construct the implicit isocurves determined by each column of control points
    // i.e. fixed values of u. This copies these columns into
    // individual matrices of control points and returns the resulting
    // curves
    std::vector<IsoCurveV> get_all_iso_curves_v() const
    {
        std::vector<IsoCurveV> iso_curves;
        for (int ui = 0; ui < num_control_points_u(); ui++) {
            typename IsoCurveV::ControlPoints control_points_v(num_control_points_v(), _dim);
            for (int vj = 0; vj < num_control_points_v(); vj++) {
                control_points_v.row(vj) = get_control_point(ui, vj);
            }

            IsoCurveV iso_curve_v;
            iso_curve_v.set_control_points(std::move(control_points_v));
            iso_curves.push_back(iso_curve_v);
        }
        return iso_curves;
    }

    // Construct the implicit isocurves determined by each rows of control points
    // i.e. fixed values of v. This copies these rows into
    // individual matrices of control points and returns the resulting
    // curves
    std::vector<IsoCurveU> get_all_iso_curves_u() const
    {
        std::vector<IsoCurveU> iso_curves;
        for (int vj = 0; vj < num_control_points_v(); vj++) {
            typename IsoCurveU::ControlPoints control_points_u(num_control_points_u(), _dim);

            for (int ui = 0; ui < num_control_points_u(); ui++) {
                control_points_u.row(ui) = get_control_point(ui, vj);
            }

            IsoCurveU iso_curve_u;
            iso_curve_u.set_control_points(std::move(control_points_u));
            iso_curves.push_back(iso_curve_u);
        }
        return iso_curves;
    }

public:
    // Split a patch into two patches along the vertical line at u
    std::vector<ThisType> split_u(Scalar u) const
    {
        if (!Base::in_domain_u(u)) {
            throw invalid_setting_error("Parameter not inside of the domain.");
        }
        if (Base::is_endpoint_u(u)) { // no split required
            return std::vector<ThisType>{*this};
        }

        const int num_split_patches = 2; // splitting one patch into two
        std::vector<IsoCurveU> iso_curves_u = get_all_iso_curves_u();

        // Initialize control grids of final split patches
        std::vector<ControlGrid> split_control_pts_u(
            num_split_patches, ControlGrid(Base::num_control_points(), _dim));

        for (int vj = 0; vj < num_control_points_v(); vj++) {
            // split each isocurve
            IsoCurveU iso_curve = iso_curves_u[static_cast<size_t>(vj)];
            std::vector<IsoCurveU> split_iso_curves = nanospline::split(iso_curve, u);

            // Copy control points of each split curve to its proper place in
            // the final control grid
            for (size_t ci = 0; ci < num_split_patches; ci++) {
                const auto &ctrl_pts = split_iso_curves[ci].get_control_points();

                for (int ui = 0; ui < num_control_points_u(); ui++) {
                    int index = Base::control_point_linear_index(ui, vj);
                    split_control_pts_u[ci].row(index) = ctrl_pts.row(ui);
                }
            }
        }

        ThisType patch_less_than_u;
        ThisType patch_greater_than_u;
        patch_less_than_u.set_control_grid(split_control_pts_u[0]);
        patch_greater_than_u.set_control_grid(split_control_pts_u[1]);

        return {patch_less_than_u, patch_greater_than_u};
    }

    // Split a patch into two patches along the horizontal line at v
    std::vector<ThisType> split_v(Scalar v) const
    {
        if (!Base::in_domain_v(v)) {
            throw invalid_setting_error("Parameter not inside of the domain.");
        }
        if (Base::is_endpoint_v(v)) { // no split required
            return std::vector<ThisType>{*this};
        }
        const int num_split_patches = 2; // splitting one patch into two

        std::vector<IsoCurveV> iso_curves_v = get_all_iso_curves_v();

        std::vector<ControlGrid> split_control_pts_v(
            num_split_patches, ControlGrid(Base::num_control_points(), _dim));

        for (int ui = 0; ui < num_control_points_u(); ui++) {
            // split each isocurve
            IsoCurveV iso_curve = iso_curves_v[static_cast<size_t>(ui)];
            std::vector<IsoCurveV> split_iso_curves = nanospline::split(iso_curve, v);

            // Copy control points of each split curve to its proper place in
            // the final control grid
            for (size_t ci = 0; ci < num_split_patches; ci++) {
                auto ctrl_pts = split_iso_curves[ci].get_control_points();

                for (int vj = 0; vj < num_control_points_v(); vj++) {
                    int index = Base::control_point_linear_index(ui, vj);
                    split_control_pts_v[ci].row(index) = ctrl_pts.row(vj);
                }
            }
        }

        ThisType patch_less_than_v;
        ThisType patch_greater_than_v;
        patch_less_than_v.set_control_grid(split_control_pts_v[0]);
        patch_greater_than_v.set_control_grid(split_control_pts_v[1]);

        return {patch_less_than_v, patch_greater_than_v};
    }

public:
    std::vector<ThisType> split(Scalar u, Scalar v) const
    {
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

    UVPoint approximate_inverse_evaluate(const Point &p,
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
            return Base::approximate_inverse_evaluate(p, num_samples, min_u, max_u, min_v, max_v);
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

            // 5. remap solution up through affine subdomain transformations
            uv(0) = (uv_max(0) - uv_min(0)) * uv(0) + uv_min(0);
            uv(1) = (uv_max(1) - uv_min(1)) * uv(1) + uv_min(1);

            return uv;
        }
    }

    static Eigen::MatrixXd form_least_squares_matrix(Eigen::MatrixXd parameters)
    {
        ThisType __;
        const int num_control_pts_u = _degree_u + 1;
        const int num_control_pts_v = _degree_v + 1;
        const int num_control_pts = num_control_pts_u * num_control_pts_v;

        Eigen::MatrixXd bezier_control_pts(num_control_pts, 1);
        BezierPatch<Scalar, 1, _degree_u, _degree_v> bezier;
        bezier_control_pts.setZero();

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
                int index = __.control_point_linear_index(j, k);
                bezier_control_pts.row(index) << 1.;
                bezier.set_control_grid(bezier_control_pts);
                bezier.initialize();
                for (int i = 0; i < num_constraints; i++) {
                    Scalar u = parameters(i, 0);
                    Scalar v = parameters(i, 1);
                    Scalar bezier_value = bezier.evaluate(u, v)(0);
                    least_squares_matrix(i, index) = bezier_value;
                }
                bezier_control_pts.row(index) << 0.;
            }
        }
        return least_squares_matrix;
    }

    static ThisType fit(Eigen::MatrixXd parameters, Eigen::MatrixXd values)
    {
        ThisType least_squares_fit;

        assert(parameters.rows() == values.rows());
        assert(parameters.cols() == 2);
        assert(values.cols() == least_squares_fit.get_dim());
        const int num_control_pts = (_degree_u + 1) * (_degree_v + 1);

        // 1. Form least squares matrix .
        Eigen::MatrixXd least_squares_matrix = form_least_squares_matrix(parameters);

        // 2. Least squares solve via SVD
        Eigen::Matrix<Scalar, num_control_pts, _dim> fit_control_points =
            least_squares_matrix.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(values);

        // 3. Store control points in least_squares_fit and return
        least_squares_fit.set_control_grid(fit_control_points);
        least_squares_fit.initialize();
        return least_squares_fit;
    }

    void deform(Eigen::MatrixXd parameters, Eigen::MatrixXd changes_in_values)
    {
        ThisType least_squares_fit = ThisType::fit(parameters, changes_in_values);

        ControlGrid changes_in_control_points = least_squares_fit.get_control_grid();
        ControlGrid updated_control_points = Base::m_control_grid + changes_in_control_points;
        Base::set_control_grid(updated_control_points);
        initialize();
    }
};

} // namespace nanospline
