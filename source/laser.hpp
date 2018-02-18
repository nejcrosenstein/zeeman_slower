#ifndef LASER_HPP
#define LASER_HPP

#include "mathtools.hpp"
#include "physics.hpp"

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
__forceinline __m256d beamWaist(__m256d const& pos_on_axis)
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
  __m256d const& pos_x,
  __m256d const& pos_y,
  __m256d const& pos_z)
{
  __m256d rad_sq =
    _mm256_fmadd_pd(pos_x, pos_x,
      _mm256_mul_pd(pos_y, pos_y));

  __m256d waist = beamWaist(pos_z);

  alignas(32) double r[4];
  _mm256_store_pd(r, rad_sq);

  alignas(32) double w[4];
  _mm256_store_pd(w, waist);
  
  for (int i = 0; i < 4; ++i)
  {
    double peakIntensity = 2 * laser_power_watt / (pi * sq(w[i]));

    w[i] = exp(-2 * r[i] / sq(w[i])) * peakIntensity;
  }

  return _mm256_load_pd(w);
}

double lightIntensity(Vec3D<double> pos)
{
  double rsq = sq(pos.x_) + sq(pos.y_);

  double w = beamWaist(pos.z_);

  double peakIntensity = 2 * laser_power_watt / (pi * sq(w));

  return exp(-2 * rsq / sq(w)) * peakIntensity;
}





#endif // COILS_HPP