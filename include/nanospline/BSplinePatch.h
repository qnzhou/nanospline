#pragma once

#include <nanospline/PatchBase.h>
#include <nanospline/BSpline.h>
#include <nanospline/split.h>

#include <iostream>
#include <vector>
using std::cout;
using std::endl;

namespace nanospline {

template<typename _Scalar, int _dim=3,
    int _degree_u=3, int _degree_v=3>
class BSplinePatch final : public PatchBase<_Scalar, _dim> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        static_assert(_dim > 0, "Dimension must be positive.");

        using Base = PatchBase<_Scalar, _dim>;
        using ThisType = BSplinePatch<_Scalar, _dim, _degree_u, _degree_v>;
        using Scalar = typename Base::Scalar;
        using Point = typename Base::Point;
        using ControlGrid = typename Base::ControlGrid;
        using KnotVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using IsoCurveU = BSpline<Scalar, _dim, _degree_u>;
        using IsoCurveV = BSpline<Scalar, _dim, _degree_v>;

    public:
        static BSplinePatch<_Scalar, _dim, _degree_u, _degree_v> ZeroPatch() {
            BSplinePatch<_Scalar, _dim, _degree_u, _degree_v> patch;
            const int degree_u = _degree_u>0?_degree_u:0;
            const int degree_v = _degree_v>0?_degree_v:0;
            patch.set_degree_u(degree_u);
            patch.set_degree_v(degree_v);
            ControlGrid grid((degree_u+1) * (degree_v+1), _dim);
            grid.setZero();
            patch.swap_control_grid(grid);
            KnotVector knots_u(2*(degree_u+1));
            knots_u.segment(0, degree_u+1).setZero();
            knots_u.segment(degree_u+1, degree_u+1).setOnes();
            KnotVector knots_v(2*(degree_v+1));
            knots_v.segment(0, degree_v+1).setZero();
            knots_v.segment(degree_v+1, degree_v+1).setOnes();
            patch.set_knots_u(knots_u);
            patch.set_knots_v(knots_v);
            patch.initialize();
            return patch;
        }

    public:
        BSplinePatch() {
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
            const auto num_v_knots = m_knots_v.size();
            const auto num_u_knots = m_knots_u.size();
            const int degree_u = Base::get_degree_u();
            const int degree_v = Base::get_degree_v();
            if (Base::m_control_grid.rows() !=
                    (num_u_knots-degree_u-1) * (num_v_knots-degree_v-1)) {
                throw invalid_setting_error(
                        "Control grid size mismatch uv degrees");
            }
        }

        Scalar get_u_lower_bound() const override {
            const auto degree_u = Base::get_degree_u();
            return m_knots_u[degree_u];
        }

        Scalar get_v_lower_bound() const override {
            const auto degree_v = Base::get_degree_v();
            return m_knots_v[degree_v];
        }

        Scalar get_u_upper_bound() const override {
            const auto num_knots = get_num_knots_u();
            const auto degree_u = Base::get_degree_u();
            return m_knots_u[num_knots-degree_u-1];
        }

        Scalar get_v_upper_bound() const override {
            const auto num_knots = get_num_knots_v();
            const auto degree_v = Base::get_degree_v();
            return m_knots_v[num_knots-degree_v-1];
        }
    private:
        // Hopefully we won't have more than 2 billion knots...
        // If so this will break.
        int get_num_knots_u() const {
            return static_cast<int>(m_knots_u.rows());
        }

        int get_num_knots_v() const {
            return static_cast<int>(m_knots_v.rows());
        }

    public:
        const KnotVector& get_knots_u() const {
            return m_knots_u;
        }

        const KnotVector& get_knots_v() const {
            return m_knots_v;
        }

        template<typename Derived>
        void set_knots_u(const Eigen::PlainObjectBase<Derived>& knots) {
            m_knots_u = knots;
        }

        template<typename Derived>
        void set_knots_v(const Eigen::PlainObjectBase<Derived>& knots) {
            m_knots_v = knots;
        }

        template<typename Derived>
        void set_knots_u(Eigen::PlainObjectBase<Derived>&& knots) {
            m_knots_u.swap(knots);
        }

        template<typename Derived>
        void set_knots_v(Eigen::PlainObjectBase<Derived>&& knots) {
            m_knots_v.swap(knots);
        }

        IsoCurveU compute_iso_curve_u(Scalar v) const {

            typename IsoCurveU::ControlPoints control_points_u(
                    num_control_points_u(),_dim);
            
            std::vector<IsoCurveV> iso_curves_v = get_all_iso_curves_v();
            for (int i=0; i<num_control_points_u(); i++) {
                IsoCurveV iso_curve = iso_curves_v[i];
                control_points_u.row(i) = iso_curve.evaluate(v);
            }

            IsoCurveU iso_curve_u;
            iso_curve_u.set_control_points(std::move(control_points_u));
            iso_curve_u.set_knots(m_knots_u);
            return iso_curve_u;
        }

        IsoCurveV compute_iso_curve_v(Scalar u) const {
            const auto num_v_knots = m_knots_v.size();
            const int degree_v = Base::get_degree_v();

            typename IsoCurveV::ControlPoints control_points_v(
                    num_v_knots-degree_v-1, _dim);
            std::vector<IsoCurveU> iso_curves_u = get_all_iso_curves_u();
            for (int j=0; j<num_v_knots-degree_v-1; j++) {
                IsoCurveU iso_curve = iso_curves_u[j];
                control_points_v.row(j) = iso_curve.evaluate(u);
            }

            IsoCurveV iso_curve_v;
            iso_curve_v.set_control_points(std::move(control_points_v));
            iso_curve_v.set_knots(m_knots_v);
            return iso_curve_v;
        }

        BSplinePatch<_Scalar, _dim, -1, -1> compute_du_patch() const {
            const int num_u_knots = get_num_knots_u();
            const int num_v_knots = get_num_knots_v();
            const int degree_u = Base::get_degree_u();
            const int degree_v = Base::get_degree_v();

            if (degree_u == 0) {
                return BSplinePatch<_Scalar, _dim, -1, -1>::ZeroPatch();
            }

            ControlGrid du_grid((num_u_knots-degree_u-2) * (num_v_knots-degree_v-1), _dim);
            for (int i=0; i<num_u_knots-degree_u-2; i++) {
                for (int j=0; j<num_v_knots-degree_v-1; j++) {
                    const int row_id = i*(num_v_knots-degree_v-1)+j;
                    du_grid.row(row_id) = degree_u *
                        (get_control_point(i+1,j) - get_control_point(i,j)) /
                        (m_knots_u[i+degree_u+1] - m_knots_u[i+1]);
                }
            }

            BSplinePatch<_Scalar, _dim, -1, -1> du_patch;
            du_patch.set_degree_u(degree_u-1);
            du_patch.set_degree_v(degree_v);
            du_patch.swap_control_grid(du_grid);
            du_patch.set_knots_u(m_knots_u.segment(1, num_u_knots-2).eval());
            du_patch.set_knots_v(m_knots_v);
            du_patch.initialize();
            return du_patch;
        }

        BSplinePatch<_Scalar, _dim, -1, -1> compute_dv_patch() const {
            const int num_u_knots = get_num_knots_u();
            const int num_v_knots = get_num_knots_v();
            const int degree_u = Base::get_degree_u();
            const int degree_v = Base::get_degree_v();

            if (degree_v == 0) {
                return BSplinePatch<_Scalar, _dim, -1, -1>::ZeroPatch();
            }

            ControlGrid dv_grid((num_u_knots-degree_u-1) * (num_v_knots-degree_v-2), _dim);
            for (int i=0; i<num_u_knots-degree_u-1; i++) {
                for (int j=0; j<num_v_knots-degree_v-2; j++) {
                    const int row_id = i*(num_v_knots-degree_v-2)+j;
                    dv_grid.row(row_id) = degree_v *
                        (get_control_point(i,j+1) - get_control_point(i,j)) /
                        (m_knots_v[j+degree_v+1] - m_knots_v[j+1]);
                }
            }

            BSplinePatch<_Scalar, _dim, -1, -1> dv_patch;
            dv_patch.set_degree_u(degree_u);
            dv_patch.set_degree_v(degree_v-1);
            dv_patch.swap_control_grid(dv_grid);
            dv_patch.set_knots_u(m_knots_u);
            dv_patch.set_knots_v(m_knots_v.segment(1, num_v_knots-2).eval());
            dv_patch.initialize();
            return dv_patch;
        }

        BSplinePatch<_Scalar, _dim, -1, -1> compute_duv_patch() const {
            return compute_du_patch().compute_dv_patch();
        }



        Point get_control_point(int ui, int vj) const {
            //const auto degree_v = Base::get_degree_v();
            //const auto row_size = m_knots_v.size() - degree_v - 1;
            //return Base::m_control_grid.row(ui*row_size + vj);
            return Base::m_control_grid.row( control_point_linear_index(ui,vj));
        }

        int num_control_points_u() const{
            const auto num_u_knots = get_num_knots_u();
            const int degree_u = Base::get_degree_u();
            return num_u_knots - degree_u - 1;
        }
        
        int num_control_points_v() const{
            const int num_v_knots = get_num_knots_v();
            const int degree_v = Base::get_degree_v();
            return num_v_knots - degree_v - 1;
        }
    protected:
        KnotVector m_knots_u;
        KnotVector m_knots_v;
    private: 
        // Translate (i,j) control point indexing into a linear index. Note that
        // control points are ordered in v-major order
        int control_point_linear_index(int i, int j) const {
            //const auto degree_v = Base::get_degree_v();
            return i*(num_control_points_v()) + j;
        }
        // Translate (i,j) control point indexing into a linear index. Note that
        // control points are ordered in v-major order
        //int control_point_linear_index(int i, int j) const {
        //    return i*(_degree_v+1) + j;
        //}
        
        // Construct the implicit isocurves determined by each rows of control points
        // i.e. fixed values of v. This copies these rows into
        // individual matrices of control points and returns the resulting
        // curves
        std::vector<IsoCurveU> get_all_iso_curves_u() const {
            const int num_control_pts_u = num_control_points_u();
            const int num_control_pts_v = num_control_points_v();

            typename IsoCurveV::ControlPoints control_points_v(
                    num_control_pts_v, _dim);
            
            std::vector<IsoCurveU> iso_curves;
            for (int vj = 0; vj < num_control_pts_v; vj++) {
              typename IsoCurveU::ControlPoints control_points_u(
                  num_control_pts_u, _dim);
              for (int ui = 0; ui < num_control_pts_u; ui++) {
                control_points_u.row(ui) = get_control_point(ui, vj);
              }

              IsoCurveU iso_curve_u;
              iso_curve_u.set_control_points(std::move(control_points_u));
              iso_curve_u.set_knots(m_knots_u);
              iso_curves.push_back(iso_curve_u);
            }
            return iso_curves;
        }
        // Construct the implicit isocurves determined by each column of control points
        // i.e. fixed values of u. This copies these columns into
        // individual matrices of control points and returns the resulting
        // curves
        std::vector<IsoCurveV> get_all_iso_curves_v() const {
            const int num_control_pts_u = num_control_points_u();
            const int num_control_pts_v = num_control_points_v();

            typename IsoCurveU::ControlPoints control_points_u(
                    num_control_pts_u, _dim);
            std::vector<IsoCurveV> iso_curves;
            for (int ui = 0; ui < num_control_pts_u; ui++) {

              typename IsoCurveV::ControlPoints control_points_v(
                  num_control_pts_v, _dim);

              for (int vj = 0; vj < num_control_pts_v; vj++) {
                control_points_v.row(vj) = get_control_point(ui, vj);
              }

              IsoCurveV iso_curve_v;
              iso_curve_v.set_control_points(std::move(control_points_v));
              iso_curve_v.set_knots(m_knots_v);
              iso_curves.push_back(iso_curve_v);
            }
            return iso_curves;
        }
        
    public:
        // Split a patch into two patches along the vertical line at u
        std::vector<ThisType> split_u(Scalar u) {
            if(!this->in_domain_u(u)) {
                throw invalid_setting_error("Parameter not inside of the domain.");
            }
          if (this->is_endpoint_u(u)){ // no split required
              return std::vector<ThisType>{*this};
          }
          
          const int num_split_patches = 2; // splitting one patch into two
          std::vector<IsoCurveU> iso_curves_u = get_all_iso_curves_u();

          // Initialize control grids of final split patches
          std::vector<ControlGrid> split_control_pts_u(num_split_patches, 
                  ControlGrid(this->num_control_points(), _dim));

          KnotVector split_knots_less_than_u;
          KnotVector split_knots_greater_than_u;
          KnotVector split_knots_v = get_knots_v(); // v-knots are the same
          for (int vj = 0; vj < num_control_points_v(); vj++) {
            // split each isocurve
            IsoCurveU iso_curve = iso_curves_u[vj];
            std::vector<IsoCurveU> split_iso_curves = nanospline::split(iso_curve, u);
            if (vj == 0) {
              // all split curves have the same knot sequence, just grab one
              // of them
              split_knots_less_than_u = split_iso_curves[0].get_knots();
              split_knots_greater_than_u = split_iso_curves[1].get_knots();
            }
            // Copy control points of each split curve to its proper place in
            // the final control grid
            for (int ci = 0; ci < num_split_patches; ci++) {
              const auto &ctrl_pts = split_iso_curves[ci].get_control_points();

              for (int ui = 0; ui < num_control_points_u(); ui++) {
                int index = control_point_linear_index(ui, vj);
                split_control_pts_u[ci].row(index) = ctrl_pts.row(ui);
              }
            }
          }

          // Initialize split patches with results
          ThisType patch_less_than_u;
          ThisType patch_greater_than_u;
          patch_less_than_u.set_control_grid(split_control_pts_u[0]);
          patch_less_than_u.set_knots_u(split_knots_less_than_u);
          patch_less_than_u.set_knots_v(split_knots_v);

          patch_greater_than_u.set_control_grid(split_control_pts_u[1]);
          patch_greater_than_u.set_knots_u(split_knots_greater_than_u);
          patch_greater_than_u.set_knots_v(split_knots_v);

          return {patch_less_than_u, patch_greater_than_u};
        }
        std::vector<ThisType> split_v(Scalar v) {
            if(!this->in_domain_v(v)) {
                throw invalid_setting_error("Parameter not inside of the domain.");
            }
          if (this->is_endpoint_v(v)){ // no split required
              return std::vector<ThisType>{*this};
          }

          const int num_split_patches = 2; // splitting one patch into two
          std::vector<IsoCurveV> iso_curves_v = get_all_iso_curves_v();

          std::vector<ControlGrid> split_control_pts_v(num_split_patches, 
                  ControlGrid(this->num_control_points(), _dim));

          KnotVector split_knots_less_than_v;
          KnotVector split_knots_greater_than_v;
          KnotVector split_knots_u = get_knots_u(); // u-knots are the same
          for (int ui = 0; ui < num_control_points_u(); ui++) {
            // split each isocurve
            IsoCurveV iso_curve = iso_curves_v[ui];
            std::vector<IsoCurveV> split_iso_curves = nanospline::split(iso_curve, v);
            if (ui == 0) {
              // all split curves have the same knot sequence, just grab one
              // of them
              split_knots_less_than_v = split_iso_curves[0].get_knots();
              split_knots_greater_than_v = split_iso_curves[1].get_knots();
            }

            // Copy control points of each split curve to its proper place in
            // the final control grid
            for (int ci = 0; ci < num_split_patches; ci++) {
              auto ctrl_pts = split_iso_curves[ci].get_control_points();

              for (int vj = 0; vj < num_control_points_v(); vj++) {
                int index = control_point_linear_index(ui, vj);
                split_control_pts_v[ci].row(index) = ctrl_pts.row(vj);
              }
            }
          }

          // Initialize split patches with results
          ThisType patch_less_than_v;
          ThisType patch_greater_than_v;
          patch_less_than_v.set_control_grid(split_control_pts_v[0]);
          patch_less_than_v.set_knots_u(split_knots_u);
          patch_less_than_v.set_knots_v(split_knots_less_than_v);

          patch_greater_than_v.set_control_grid(split_control_pts_v[1]);
          patch_greater_than_v.set_knots_u(split_knots_u);
          patch_greater_than_v.set_knots_v(split_knots_greater_than_v);

          return {patch_less_than_v, patch_greater_than_v};
        }

        std::vector<ThisType> split(Scalar u, Scalar v) {
          if (!this->in_domain(u, v)) {
            throw invalid_setting_error("Parameter not inside of the domain.");
          }
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
            throw invalid_setting_error("Invalid range in subcurve: v");
          } 

          // TODO add domain handling to reduce computation
        
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

}
