#ifndef LASER_HPP
#define LASER_HPP

#include "mathtools.hpp"
#include "physics.hpp"


#include <cmath>
#include <immintrin.h>

constexpr double laser_power_watt = 0.016;
constexpr double focus_point_z = -1.4;

constexpr double beam_spread_angle_rad_ = 0.004;
constexpr double beam_spread_angle_tangent_ = beam_spread_angle_rad_;

constexpr double waist_at_oven_exit = 0.004;
constexpr double oven_exit_pos_z = 0.63 - 0.83 - 0.15;

//
// Beam waist at current position (let's ignore waist of Gaussian beam for now)
// 
__forceinline __m256d __vectorcall beamWaist(__m256d const& pos_on_axis)
{
  __m256d dist = _mm256_sub_pd(pos_on_axis, _mm256_set1_pd(oven_exit_pos_z));

  return _mm256_fmadd_pd(dist, 
    _mm256_set1_pd(beam_spread_angle_tangent_), 
    _mm256_set1_pd(waist_at_oven_exit));
}

__forceinline double beamWaist(double pos_on_axis)
{
  return waist_at_oven_exit + (pos_on_axis - oven_exit_pos_z)* beam_spread_angle_tangent_;
}

// 
// Beam intensity at current position
//
__forceinline __m256d __vectorcall lightIntensity(
  __m256d const (&pos)[3])
{
  __m256d rad_sq =
    _mm256_fmadd_pd(pos[0], pos[0],
      _mm256_mul_pd(pos[1], pos[1]));

  __m256d waist = beamWaist(pos[2]);

  __m256d waist_sq_inv = 
    _mm256_div_pd(
      _mm256_set1_pd(1.0),
      _mm256_mul_pd(waist, waist));

  __m256d peak_intensity =
    _mm256_mul_pd(
      _mm256_set1_pd(2.0*laser_power_watt/pi),
      waist_sq_inv);

  __m256d twice_ratio_squared =
    _mm256_mul_pd(
      _mm256_set1_pd(2.0),
      _mm256_mul_pd(rad_sq, waist_sq_inv));

  alignas(32) double trsq[4];
  _mm256_store_pd(trsq, twice_ratio_squared);
  
  alignas(32) double exped[4];
  for (int i = 0; i < 4; ++i)
  {
    exped[i] = std::exp(-trsq[i]);
  }

  return _mm256_mul_pd(_mm256_load_pd(exped), peak_intensity);
}

double lightIntensity(Vec3D<double> pos)
{
  double rsq = sq(pos.x_) + sq(pos.y_);

  double w = beamWaist(pos.z_);

  double peakIntensity = 2 * laser_power_watt / (pi * sq(w));

  return exp(-2 * rsq / sq(w)) * peakIntensity;
}





#endif // COILS_HPP