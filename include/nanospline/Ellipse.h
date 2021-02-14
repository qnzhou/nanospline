#pragma once

#include <nanospline/CurveBase.h>
#include <nanospline/Exceptions.h>

namespace nanospline {

template <typename _Scalar, int _dim>
class Ellipse final : public CurveBase<_Scalar, _dim>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static_assert(_dim > 1, "Dimension must be greater than 1.");
    using Base = CurveBase<_Scalar, _dim>;
    using typename Base::Point;
    using typename Base::Scalar;
    using Frame = Eigen::Matrix<_Scalar, 2, _dim>;

public:
    Ellipse()
        : Base()
    {
        m_frame.setZero();
        m_frame(0, 0) = 1;
        m_frame(1, 1) = 1;
        update_periodicity();
    }

    CurveEnum get_curve_type() const override { return CurveEnum::ELLIPSE; }

    std::unique_ptr<Base> clone() const override
    {
        auto ptr = std::make_unique<Ellipse<_Scalar, _dim>>();
        ptr->set_major_radius(m_major_radius);
        ptr->set_minor_radius(m_minor_radius);
        ptr->set_center(m_center);
        ptr->set_frame(m_frame);
        return ptr;
    }

public:
    Scalar get_major_radius() const { return m_major_radius; }
    void set_major_radius(Scalar r) { m_major_radius = r; }

    Scalar get_minor_radius() const { return m_minor_radius; }
    void set_minor_radius(Scalar r) { m_minor_radius = r; }

    const Point& get_center() const { return m_center; }
    void set_center(const Point& c) { m_center = c; }

    // Ellipse is embedded in the plane spanned by the first 2 axes of the frame.
    const Frame get_frame() const { return m_frame; }
    void set_frame(const Frame& f) { m_frame = f; }

    void set_domain_lower_bound(Scalar t)
    {
        m_lower = t;
        update_periodicity();
    }
    void set_domain_upper_bound(Scalar t)
    {
        m_upper = t;
        update_periodicity();
    }

public:
    int get_degree() const override { return -1; }
    bool in_domain(Scalar t) const override { return t >= m_lower && t <= m_upper; }
    Scalar get_domain_lower_bound() const override { return m_lower; }
    Scalar get_domain_upper_bound() const override { return m_upper; }

public:
    Point evaluate(Scalar t) const override
    {
        return m_center + m_major_radius * (m_frame.row(0) * std::cos(t)) +
               m_minor_radius * (m_frame.row(1) * std::sin(t));
    }

    Scalar inverse_evaluate(const Point& p) const override
    {
        return approximate_inverse_evaluate(p, m_lower, m_upper);
    }

    Point evaluate_derivative(Scalar t) const override
    {
        return m_major_radius * (-m_frame.row(0)) * std::sin(t) +
               m_minor_radius * m_frame.row(1) * std::cos(t);
    }

    Point evaluate_2nd_derivative(Scalar t) const override
    {
        return -m_major_radius * m_frame.row(0) * std::cos(t) -
               m_minor_radius * m_frame.row(1) * std::sin(t);
    }

    Scalar approximate_inverse_evaluate(
        const Point& p, const Scalar lower, const Scalar upper, const int level = 3) const override
    {
        (void)level; // Level is not needed here.
        assert(lower <= upper);
        auto x = (p - m_center).dot(m_frame.row(0)) / m_frame.row(0).norm();
        auto y = (p - m_center).dot(m_frame.row(1)) / m_frame.row(1).norm();
        auto t = std::atan2(y * m_major_radius, x * m_minor_radius);

        while (t < lower) {
            t += 2 * M_PI;
        }

        if (t > upper) {
            auto d1 = (evaluate(lower) - p).squaredNorm();
            auto d2 = (evaluate(upper) - p).squaredNorm();
            if (d1 < d2) {
                t = lower;
            } else {
                t = upper;
            }
        }
        return t;
    }

public:
    int get_num_control_points() const override { return 0; }
    Point get_control_point(int) const override
    {
        throw not_implemented_error("Ellipse does not support control points.");
    }
    void set_control_point(int, const Point&) override
    {
        throw not_implemented_error("Ellipse does not support control points.");
    }

    int get_num_weights() const override { return 0; }
    Scalar get_weight(int) const override
    {
        throw not_implemented_error("Ellipse does not support weights.");
    }
    void set_weight(int, Scalar) override
    {
        throw not_implemented_error("Ellipse does not support weights.");
    }

    int get_num_knots() const override { return 0; }
    Scalar get_knot(int) const override
    {
        throw not_implemented_error("Ellipse does not support knots.");
    }
    void set_knot(int, Scalar) override
    {
        throw not_implemented_error("Ellipse does not support knots.");
    }

public:
    std::vector<Scalar> compute_inflections(const Scalar, const Scalar) const override
    {
        return {};
    }

    std::vector<Scalar> reduce_turning_angle(const Scalar, const Scalar) const override
    {
        // TODO.
        throw not_implemented_error("Turning angle of a Ellipse cannot be reduced.");
    }

    std::vector<Scalar> compute_singularities(const Scalar, const Scalar) const override
    {
        return {};
    }

private:
    void update_periodicity()
    {
        if (std::abs(fmod(m_upper - m_lower, 2 * M_PI)) < 1e-6) {
            Base::set_periodic(true);
        } else {
            Base::set_periodic(false);
        }
    }

private:
    Point m_center = Point::Zero();
    Frame m_frame;
    Scalar m_major_radius = 0;
    Scalar m_minor_radius = 0;
    Scalar m_lower = 0;
    Scalar m_upper = 2 * M_PI;
};

} // namespace nanospline
