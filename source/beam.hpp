#ifndef BEAM_HPP
#define BEAM_HPP

#include <math.h>

#include "mathtools.hpp"

// f(v) = v^3*exp(-v^2 / u^2)
struct BeamVelocityDistribution
{
  double cdf(double v) const
  {
    return exp(-sq(v / u_))*(-sq(v / u_) - 1.0) * sq(sq(u_));
  }

  double cdfProbabilty(double v) const
  {
    return (cdf(v) - cdf0_) / cdf_norm_;
  }

  double velocityFromCDF(double prob) const
  {
    // perform bisection.

    double step = 100.0;
    double tolerance = 0.00001;

    double v = 0.0;
    double y = 0.0;

    while (std::abs(y - prob) > tolerance)
    {
      if (y < prob)
      {
        v += step;
        y = cdfProbabilty(v);
      }
      else
      {
        v -= step;
        y = cdfProbabilty(v);
        step /= 10.0;
      }
    }

    return v;
  }

  template<class RandGen_t>
  double operator()(RandGen_t& rand_gen) const
  {
    return velocityFromCDF(zero_to_one_(rand_gen));
  }

  BeamVelocityDistribution(double temperature_kelvin)
    : zero_to_one_(0.0, 1.0)
  {
    u_ = 2230.0*std::sqrt(temperature_kelvin / 300.0*1.0 / 132.0);

    cdf0_ = cdf(0.0);
    cdf_norm_ = cdf(10000.0 * u_) - cdf0_;
  }


  double u_;
  double cdf_norm_;
  double cdf0_;
  std::uniform_real_distribution<double> zero_to_one_;
};


#endif // BEAM_HPP