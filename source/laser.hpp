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

namespace laser
{

//
// Beam waist at current position (let's ignore waist of Gaussian beam for now)
// 
__forceinline __m256d __vectorcall beamWaist(__m256d const& pos_on_axis)
{
  __m256d dist = subtract(pos_on_axis, broadcast(oven_exit_pos_z));

  return multiply_add(dist, 
    broadcast(beam_spread_angle_tangent_), 
    broadcast(waist_at_oven_exit));
}

// 
// Beam intensity at current position
//
__forceinline __m256d __vectorcall beamIntensity(
  __m256d const (&pos)[NDir])
{
  __m256d rad_sq =
    multiply_add(pos[X], pos[X],
        multiply(pos[Y], pos[Y]));

  __m256d waist = beamWaist(pos[Z]);

  __m256d waist_sq_inv = reciprocal(multiply(waist, waist));

  __m256d peak_intensity =
    multiply(waist_sq_inv, broadcast(2.0*laser_power_watt / pi));

  __m256d twice_ratio_squared =
    multiply(
      broadcast(2.0),
      multiply(rad_sq, waist_sq_inv));

  Quadruple trsq;
  store(twice_ratio_squared, trsq);
  
  Quadruple exped;
  for (int i = 0; i < 4; ++i)
  {
    exped[i] = std::exp(-trsq[i]);
  }

  return multiply(load(exped), peak_intensity);
}

//
// Computes beam direction. It is assumed that the
// beam converges towards a point on the slower axis. 
// The function returns normalized direction.
//
__forceinline void __vectorcall beamDirection(
  const __m256d (&pos)[NDir],
  __m256d (&dir)[NDir])
{
  // focus point
  __m256d foc[NDir];
  foc[X] = _mm256_setzero_pd();
  foc[Y] = _mm256_setzero_pd();
  foc[Z] = broadcast(focus_point_z);

  __m256d dirs[NDir];
  for (int i = 0; i < NDir; ++i)
    dirs[i] = subtract(foc[i], pos[i]);

  __m256d norm_squared = dotProduct(dirs, dirs);

  __m256d norm = _mm256_sqrt_pd(norm_squared);

  __m256d norm_inv = _mm256_div_pd(_mm256_set1_pd(1.0), norm);

  dir[X] = _mm256_mul_pd(dirs[X], norm_inv);
  dir[Y] = _mm256_mul_pd(dirs[Y], norm_inv);
  dir[Z] = _mm256_mul_pd(dirs[Z], norm_inv);
}

} // namespace laser


#endif // COILS_HPP