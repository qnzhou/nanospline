#pragma once

#include <nanospline/CurveBase.h>
#include <nanospline/Exceptions.h>

namespace nanospline {

template <typename _Scalar, int _dim>
class Circle final : public CurveBase<_Scalar, _dim>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static_assert(_dim > 1, "Dimension must be greater than 1.");
    using Base = CurveBase<_Scalar, _dim>;
    using typename Base::Point;
    using typename Base::Scalar;
    using Frame = Eigen::Matrix<_Scalar, 2, _dim>;

public:
    Circle()
        : Base()
    {
        m_frame.setZero();
        m_frame(0, 0) = 1;
        m_frame(1, 1) = 1;
        update_periodicity();
    }

    std::unique_ptr<Base> clone() const override
    {
        auto ptr = std::make_unique<Circle<_Scalar, _dim>>();
        ptr->set_radius(m_radius);
        ptr->set_center(m_center);
        ptr->set_frame(m_frame);
        return ptr;
    }

public:
    Scalar get_radius() const { return m_radius; }
    void set_radius(Scalar r) { m_radius = r; }

    const Point& get_center() const { return m_center; }
    void set_center(const Point& c) { m_center = c; }

    // Circle is embedded in the plane spanned by the first 2 axes of the frame.
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
        return m_center + m_radius * (m_frame.row(0) * std::cos(t) + m_frame.row(1) * std::sin(t));
    }

    Scalar inverse_evaluate(const Point& p) const override
    {
        const Scalar x = m_frame.row(0).dot(p - m_center) / m_frame.row(0).norm();
        const Scalar y = m_frame.row(1).dot(p - m_center) / m_frame.row(1).norm();
        auto t = std::atan2(y, x);
        return Base::unwrap_parameter(t);
    }

    Point evaluate_derivative(Scalar t) const override
    {
        return m_radius * (-m_frame.row(0) * std::sin(t) + m_frame.row(1) * std::cos(t));
    }

    Point evaluate_2nd_derivative(Scalar t) const override
    {
        return m_radius * (-m_frame.row(0) * std::cos(t) - m_frame.row(1) * std::sin(t));
    }

    Scalar approximate_inverse_evaluate(
        const Point& p, const Scalar lower, const Scalar upper, const int level = 3) const override
    {
        (void)level; // Level is not needed here.

        Scalar t = inverse_evaluate(p);
        if (t >= lower && t <= upper) return t;

        Scalar dist_lower = (evaluate(lower) - p).squaredNorm();
        Scalar dist_upper = (evaluate(upper) - p).squaredNorm();
        if (dist_lower < dist_upper) {
            return lower;
        } else {
            return upper;
        }
    }

public:
    int get_num_control_points() const override { return 0; }
    Point get_control_point(int) const override
    {
        throw not_implemented_error("Circle does not support control points.");
    }
    void set_control_point(int, const Point&) override
    {
        throw not_implemented_error("Circle does not support control points.");
    }

    int get_num_weights() const override { return 0; }
    Scalar get_weight(int) const override
    {
        throw not_implemented_error("Circle does not support weights.");
    }
    void set_weight(int, Scalar) override
    {
        throw not_implemented_error("Circle does not support weights.");
    }

    int get_num_knots() const override { return 0; }
    Scalar get_knot(int) const override
    {
        throw not_implemented_error("Circle does not support knots.");
    }
    void set_knot(int, Scalar) override
    {
        throw not_implemented_error("Circle does not support knots.");
    }

public:
    std::vector<Scalar> compute_inflections(const Scalar, const Scalar) const override
    {
        return {};
    }

    std::vector<Scalar> reduce_turning_angle(const Scalar, const Scalar) const override
    {
        throw not_implemented_error("Turning angle of a circle cannot be reduced.");
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
    Scalar m_radius = 0;
    Scalar m_lower = 0;
    Scalar m_upper = 2 * M_PI;
};

} // namespace nanospline
