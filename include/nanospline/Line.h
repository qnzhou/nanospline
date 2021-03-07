#pragma once

#include <nanospline/CurveBase.h>
#include <nanospline/Exceptions.h>

namespace nanospline {

template <typename _Scalar, int _dim>
class Line final : public CurveBase<_Scalar, _dim>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static_assert(_dim > 0, "Dimension must be greater than 0.");
    using Base = CurveBase<_Scalar, _dim>;
    using typename Base::Point;
    using typename Base::Scalar;

public:
    Line()
        : Base()
    {
        m_direction.setZero();
        m_direction[0] = 1;
    }

    CurveEnum get_curve_type() const override { return CurveEnum::LINE; }

    std::unique_ptr<Base> clone() const override
    {
        auto ptr = std::make_unique<Line<_Scalar, _dim>>();
        ptr->set_direction(m_direction);
        ptr->set_location(m_location);
        ptr->set_domain_lower_bound(m_lower);
        ptr->set_domain_upper_bound(m_upper);
        ptr->initialize();
        return ptr;
    }

    void initialize() override {
        assert(m_upper >= m_lower);
        assert(m_direction.squaredNorm() > std::numeric_limits<Scalar>::epsilon());
    }

public:
    const Point& get_location() const { return m_location; }
    void set_location(const Point& c) { m_location = c; }

    const Point& get_direction() const { return m_direction; }
    void set_direction(const Point& d) { m_direction = d; }

    void set_domain_lower_bound(Scalar t) { m_lower = t; }
    void set_domain_upper_bound(Scalar t) { m_upper = t; }

public:
    int get_degree() const override { return -1; }
    bool in_domain(Scalar t) const override { return t >= m_lower && t <= m_upper; }
    Scalar get_domain_lower_bound() const override { return m_lower; }
    Scalar get_domain_upper_bound() const override { return m_upper; }

public:
    Point evaluate(Scalar t) const override { return m_location + t * m_direction; }

    Scalar inverse_evaluate(const Point& p) const override
    {
        return approximate_inverse_evaluate(p, m_lower, m_upper);
    }

    Point evaluate_derivative(Scalar t) const override { return m_direction; }

    Point evaluate_2nd_derivative(Scalar t) const override { return Point::Zero(); }

    Scalar approximate_inverse_evaluate(
        const Point& p, const Scalar lower, const Scalar upper, const int level = 3) const override
    {
        (void)level;
        Scalar t = (p - m_location).dot(m_direction) / m_direction.squaredNorm();
        if (t < lower) return lower;
        if (t > upper) return upper;
        return t;
    }

public:
    int get_num_control_points() const override { return 0; }
    Point get_control_point(int) const override
    {
        throw not_implemented_error("Line does not support control points.");
    }
    void set_control_point(int, const Point&) override
    {
        throw not_implemented_error("Line does not support control points.");
    }

    int get_num_weights() const override { return 0; }
    Scalar get_weight(int) const override
    {
        throw not_implemented_error("Line does not support weights.");
    }
    void set_weight(int, Scalar) override
    {
        throw not_implemented_error("Line does not support weights.");
    }

    int get_num_knots() const override { return 0; }
    Scalar get_knot(int) const override
    {
        throw not_implemented_error("Line does not support knots.");
    }
    void set_knot(int, Scalar) override
    {
        throw not_implemented_error("Line does not support knots.");
    }

public:
    std::vector<Scalar> compute_inflections(const Scalar, const Scalar) const override
    {
        return {};
    }

    std::vector<Scalar> reduce_turning_angle(const Scalar, const Scalar) const override
    {
        throw not_implemented_error("Turning angle of a Line cannot be reduced.");
    }

    std::vector<Scalar> compute_singularities(const Scalar, const Scalar) const override
    {
        return {};
    }

    Scalar get_turning_angle(Scalar t0, Scalar t1) const override
    {
        (void)t0;
        (void)t1;
        if (_dim != 2) {
            throw std::runtime_error("Turning angle reduction is for 2D curves only");
        }
        return 0;
    }

private:
    Point m_location = Point::Zero();
    Point m_direction;
    Scalar m_lower = 0;
    Scalar m_upper = 1;
};

} // namespace nanospline
