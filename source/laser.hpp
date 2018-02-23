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
  __m256d dist = _mm256_sub_pd(pos_on_axis, _mm256_set1_pd(oven_exit_pos_z));

  return _mm256_fmadd_pd(dist, 
    _mm256_set1_pd(beam_spread_angle_tangent_), 
    _mm256_set1_pd(waist_at_oven_exit));
}

// 
// Beam intensity at current position
//
__forceinline __m256d __vectorcall beamIntensity(
  __m256d const (&pos)[NDir])
{
  __m256d rad_sq =
    _mm256_fmadd_pd(pos[X], pos[X],
      _mm256_mul_pd(pos[Y], pos[Y]));

  __m256d waist = beamWaist(pos[Z]);

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
  foc[Z] = _mm256_set1_pd(focus_point_z);

  __m256d dirs[NDir];
  for (int i = 0; i < NDir; ++i)
    dirs[i] = _mm256_sub_pd(foc[i], pos[i]);

  __m256d norm_squared = dotProduct(dirs, dirs);

  __m256d norm = _mm256_sqrt_pd(norm_squared);

  __m256d norm_inv = _mm256_div_pd(_mm256_set1_pd(1.0), norm);

  dir[X] = _mm256_mul_pd(dirs[X], norm_inv);
  dir[Y] = _mm256_mul_pd(dirs[Y], norm_inv);
  dir[Z] = _mm256_mul_pd(dirs[Z], norm_inv);
}

} // namespace laser


#endif // COILS_HPP