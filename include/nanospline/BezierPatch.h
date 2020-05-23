#pragma once

#include <cstddef>
#include <nanospline/PatchBase.h>
#include <nanospline/split.h>
#include <nanospline/Bezier.h>
#include <nanospline/Exceptions.h>
#include <vector>
#include <iostream>

        using std::cout;
        using std::endl;
namespace nanospline {

template<typename _Scalar, int _dim=3,
    int _degree_u=3, int _degree_v=3>
class BezierPatch final : public PatchBase<_Scalar, _dim> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        static_assert(_dim > 0, "Dimension must be positive.");

        using Base = PatchBase<_Scalar, _dim>;
        using ThisType = BezierPatch<_Scalar, _dim, _degree_u, _degree_v>;
        using Scalar = typename Base::Scalar;
        using Point = typename Base::Point;
        using ControlGrid = typename Base::ControlGrid;
        using IsoCurveU = Bezier<Scalar, _dim, _degree_u>;
        using IsoCurveV = Bezier<Scalar, _dim, _degree_v>;

    public:
        static BezierPatch<_Scalar, _dim, _degree_u, _degree_v> ZeroPatch() {
            BezierPatch<_Scalar, _dim, _degree_u, _degree_v> patch;
            patch.set_degree_u(_degree_u>0?_degree_u:0);
            patch.set_degree_v(_degree_v>0?_degree_v:0);
            ControlGrid grid((patch.get_degree_u()+1) * (patch.get_degree_v()+1), _dim);
            grid.setZero();
            patch.swap_control_grid(grid);
            patch.initialize();
            return patch;
        }

    public:
        BezierPatch() {
            Base::set_degree_u(_degree_u);
            Base::set_degree_v(_degree_v);
        }

    public:
        Point evaluate(Scalar u, Scalar v) const override {
            auto iso_curve_v = compute_iso_curve_v(u);
            return iso_curve_v.evaluate(v);
        }

        Point evaluate_derivative_u(Scalar u, Scalar v) const override {
            auto iso_curve_u = compute_iso_curve_u(v);
            return iso_curve_u.evaluate_derivative(u);
        }

        Point evaluate_derivative_v(Scalar u, Scalar v) const override {
            auto iso_curve_v = compute_iso_curve_v(u);
            return iso_curve_v.evaluate_derivative(v);
        }

        Point evaluate_2nd_derivative_uu(Scalar u, Scalar v) const override {
            auto iso_curve_u = compute_iso_curve_u(v);
            return iso_curve_u.evaluate_2nd_derivative(u);
        }

        Point evaluate_2nd_derivative_vv(Scalar u, Scalar v) const override {
            auto iso_curve_v = compute_iso_curve_v(u);
            return iso_curve_v.evaluate_2nd_derivative(v);
        }

        Point evaluate_2nd_derivative_uv(Scalar u, Scalar v) const override {
            auto duv_patch = compute_duv_patch();
            return duv_patch.evaluate(u, v);
        }

        void initialize() override {
            const int degree_u = Base::get_degree_u();
            const int degree_v = Base::get_degree_v();
            if (Base::m_control_grid.rows() != (degree_u+1) * (degree_v+1)) {
                throw invalid_setting_error(
                        "Control grid size mismatch uv degrees");
            }
        }

        Scalar get_u_lower_bound() const override {
            return 0.0;
        }

        Scalar get_v_lower_bound() const override {
            return 0.0;
        }

        Scalar get_u_upper_bound() const override {
            return 1.0;
        }

        Scalar get_v_upper_bound() const override {
            return 1.0;
        }
        

    public:
        IsoCurveU compute_iso_curve_u(Scalar v) const {
            typename IsoCurveU::ControlPoints control_points_u(num_control_points_u(), _dim);
            
            std::vector<IsoCurveV> iso_curves_v = get_all_iso_curves_v();
            for (int i=0; i < num_control_points_u(); i++) {
                IsoCurveV iso_curve_v = iso_curves_v[size_t(i)];
                control_points_u.row(i) = iso_curve_v.evaluate(v);
            }

            IsoCurveU iso_curve_u;
            iso_curve_u.set_control_points(std::move(control_points_u));
            return iso_curve_u;
        }

        IsoCurveV compute_iso_curve_v(Scalar u) const {
            typename IsoCurveV::ControlPoints control_points_v(num_control_points_v(), _dim);
            
            std::vector<IsoCurveU> iso_curves_u = get_all_iso_curves_u();
            for (int i=0; i < num_control_points_v(); i++) {
                IsoCurveU iso_curve = iso_curves_u[size_t(i)];
                control_points_v.row(i) = iso_curve.evaluate(u);
            }
            IsoCurveV iso_curve_v;
            iso_curve_v.set_control_points(std::move(control_points_v));
            return iso_curve_v;
        }

        BezierPatch<_Scalar, _dim, -1, -1> compute_du_patch() const {
            const int degree_u = Base::get_degree_u();
            const int degree_v = Base::get_degree_v();

            if (degree_u == 0) {
                return BezierPatch<_Scalar, _dim, -1, -1>::ZeroPatch();
            }

            ControlGrid du_grid(degree_u*(degree_v+1), _dim);
            for (int i=0; i<=degree_u-1; i++) {
                for (int j=0; j<=degree_v; j++) {
                    const int row_id = i*(degree_v+1)+j;
                    du_grid.row(row_id) = degree_u *
                        (get_control_point(i+1,j) - get_control_point(i,j));
                }
            }

            BezierPatch<_Scalar, _dim, -1, -1> du_patch;
            du_patch.set_degree_u(degree_u-1);
            du_patch.set_degree_v(degree_v);
            du_patch.swap_control_grid(du_grid);
            du_patch.initialize();
            return du_patch;
        }

        BezierPatch<_Scalar, _dim, -1, -1> compute_dv_patch() const {
            const int degree_u = Base::get_degree_u();
            const int degree_v = Base::get_degree_v();

            if (degree_v == 0) {
                return BezierPatch<_Scalar, _dim, -1, -1>::ZeroPatch();
            }

            ControlGrid dv_grid((degree_u+1)*degree_v, _dim);
            for (int i=0; i<=degree_u; i++) {
                for (int j=0; j<=degree_v-1; j++) {
                    const int row_id = i*degree_v+j;
                    dv_grid.row(row_id) = degree_v *
                        (get_control_point(i,j+1) - get_control_point(i,j));
                }
            }

            BezierPatch<_Scalar, _dim, -1, -1> dv_patch;
            dv_patch.set_degree_u(degree_u);
            dv_patch.set_degree_v(degree_v-1);
            dv_patch.swap_control_grid(dv_grid);
            dv_patch.initialize();
            return dv_patch;
        }

        BezierPatch<_Scalar, _dim, -1, -1> compute_duv_patch() const {
            const int degree_u = Base::get_degree_u();
            const int degree_v = Base::get_degree_v();

            if (degree_u == 0 || degree_v == 0) {
                return BezierPatch<_Scalar, _dim, -1, -1>::ZeroPatch();
            }

            ControlGrid duv_grid(degree_u*degree_v, _dim);
            for (int i=0; i<=degree_u-1; i++) {
                for (int j=0; j<=degree_v-1; j++) {
                    const int row_id = i*degree_v+j;
                    const Point pu0 = degree_u *
                        (get_control_point(i+1,j) - get_control_point(i,j));
                    const Point pu1 = degree_u *
                        (get_control_point(i+1,j+1) - get_control_point(i,j+1));
                    duv_grid.row(row_id) = degree_v * (pu1 - pu0);
                }
            }

            BezierPatch<_Scalar, _dim, -1, -1> duv_patch;
            duv_patch.set_degree_u(degree_u-1);
            duv_patch.set_degree_v(degree_v-1);
            duv_patch.swap_control_grid(duv_grid);
            duv_patch.initialize();
            return duv_patch;
        }


        Point get_control_point(int ui, int vj) const override {
            return Base::m_control_grid.row(ui*num_control_points_v()+vj);
        }
        int num_control_points_u() const override {
            const int degree_u = Base::get_degree_u();
            return degree_u + 1;
        }
        int num_control_points_v() const override {
            const int degree_v = Base::get_degree_v();
            return degree_v + 1;
        }
    private: 

        // Translate (i,j) control point indexing into a linear index. Note that
        // control points are ordered in v-major order
        int control_point_linear_index(int i, int j) const {
            return i*(num_control_points_v()) + j;
        }
        
        // Construct the implicit isocurves determined by each column of control points
        // i.e. fixed values of u. This copies these columns into
        // individual matrices of control points and returns the resulting
        // curves
        std::vector<IsoCurveV> get_all_iso_curves_v() const {
            
            std::vector<IsoCurveV> iso_curves;
            for (int ui = 0; ui < num_control_points_u(); ui++) {
              typename IsoCurveV::ControlPoints control_points_v(
                  num_control_points_v(), _dim);
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
        std::vector<IsoCurveU> get_all_iso_curves_u() const {

          std::vector<IsoCurveU> iso_curves;
          for (int vj = 0; vj < num_control_points_v(); vj++) {
            typename IsoCurveU::ControlPoints control_points_u( num_control_points_u(), _dim);

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
        std::vector<ThisType> split_u(Scalar u) {
            // TODO domain checking
          const int num_split_patches = 2; // splitting one patch into two
          std::vector<IsoCurveU> iso_curves_u = get_all_iso_curves_u();

          // Initialize control grids of final split patches
          std::vector<ControlGrid> split_control_pts_u(num_split_patches, 
                  ControlGrid(this->num_control_points(), _dim));

          for (int vj = 0; vj < num_control_points_v(); vj++) {
            // split each isocurve
            IsoCurveU iso_curve = iso_curves_u[static_cast<size_t>(vj)];
            std::vector<IsoCurveU> split_iso_curves = nanospline::split(iso_curve, u);

            // Copy control points of each split curve to its proper place in
            // the final control grid
            for (size_t ci = 0; ci < num_split_patches; ci++) {
              const auto &ctrl_pts = split_iso_curves[ci].get_control_points();

              for (int ui = 0; ui < num_control_points_u(); ui++) {
                int index = control_point_linear_index(ui, vj);
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
        std::vector<ThisType> split_v(Scalar v) {
            // TODO domain checking
          const int num_split_patches = 2; // splitting one patch into two

          std::vector<IsoCurveV> iso_curves_v = get_all_iso_curves_v();

          std::vector<ControlGrid> split_control_pts_v(num_split_patches, 
                  ControlGrid(this->num_control_points(), _dim));

          for (int ui = 0; ui < num_control_points_u(); ui++) {
            // split each isocurve
            IsoCurveV iso_curve = iso_curves_v[static_cast<size_t>(ui)];
            std::vector<IsoCurveV> split_iso_curves = nanospline::split(iso_curve, v);

            // Copy control points of each split curve to its proper place in
            // the final control grid
            for (size_t ci = 0; ci < num_split_patches; ci++) {
              auto ctrl_pts = split_iso_curves[ci].get_control_points();

              for (int vj = 0; vj < num_control_points_v(); vj++) {
                int index = control_point_linear_index(ui, vj);
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
        std::vector<ThisType> split(Scalar u, Scalar v) {
          if (this->is_endpoint_u(u) &&
              this->is_endpoint_v(v)) { // no splitting required
            return std::vector<ThisType>{*this};

          } else if (this->is_endpoint_u(u)) { // no need to split along u
            return split_v(v);

          } else if (this->is_endpoint_v(v)) { // no need to split along v
            return split_u(u);

          } else {

            // 1. split in u direction
            std::vector<ThisType> split_patches = split_u(u);
            ThisType less_than_u = split_patches[0];
            ThisType greater_than_u = split_patches[1];

            // 2. Split each subpatch patch at v

            std::vector<ThisType> split_patches_less_than_u =
                less_than_u.split_v(v);

            std::vector<ThisType> split_patches_greater_than_u =
                greater_than_u.split_v(v);

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
            ThisType greater_than_u_less_than_v =
                split_patches_greater_than_u[0];
            ThisType greater_than_u_greater_than_v =
                split_patches_greater_than_u[1];

            return {less_than_u_less_than_v, less_than_u_greater_than_v,
                    greater_than_u_less_than_v, greater_than_u_greater_than_v};
          }
        }

        std::vector<ThisType> subpatch(const ThisType &patch, Scalar u_min,
                                       Scalar u_max, Scalar v_min,
                                       Scalar v_max) {
          if (u_min > u_max) {
            throw invalid_setting_error("u_min must be smaller than u_max");
          }
          if (u_min < 0 || u_min > 1 || u_max < 0 || u_max > 1) {
            throw invalid_setting_error("Invalid range in subcurve: u");
          }
          if (v_min > v_max) {
            throw invalid_setting_error("v_min must be smaller than v_max");
          }
          if (v_min < 0 || v_min > 1 || v_max < 0 || v_max > 1) {
            throw invalid_setting_error("Invalid range in subcurve: u");
          }
        
          // TODO add domain handling to reduce computation
          //
          // 1. Split patch in u direction at u_min; take the patch on >= u_min
          ThisType greater_than_umin = patch.split_u(u_min)[1];

          // 2. Split patch on >= u_min at u_max in u dir; take the patch <=
          // u_max
          ThisType greater_than_umin_less_umax = patch.split_u(u_min)[0];

          // 3. Split patch on u_min <= u <= umax at v_min  in the v dir, take
          // the patch >= v_min
          ThisType greater_than_vmin = greater_than_umin_less_umax[1];

          // 4. Split patch  at v_max in the v dir, take the patch <= v_max
          ThisType greater_than_vmin_less_vmax = greater_than_vmin[0];
          return greater_than_vmin_less_vmax;
        }
};

} // namespace nanospline
