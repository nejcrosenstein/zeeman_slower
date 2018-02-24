#ifndef PHYSICS_HPP
#define PHYSICS_HPP

#include "mathtools.hpp"

#include <immintrin.h>

static constexpr double pi = 3.1415926535897932384;
static constexpr double two_pi = 2.0*pi;
static constexpr double degree_to_rad_ = pi / 180.0;

//
// General physical constants
//

static constexpr double planck = 6.626070040e-34;
static constexpr double planck_reduced = planck / two_pi;
static constexpr double bohr_magneton = 9.274009994e-24;
static constexpr double magnetic_constant = pi * 4.0e-7;
static constexpr double speed_of_light = 299792458;


//
// ( F=4 -> F'=5 ) transition properties 
//
static constexpr double excited_state_lifetime = 30.474e-9; // s
static constexpr double natural_linewidth_hz = 32.815e6; // s^-1

static constexpr double transition_lambda = 852.0e-9; //m
static constexpr double transition_wave_vec = two_pi / transition_lambda;
static constexpr double g_factor = 1.0;

static constexpr double saturation_intensity =
  (pi * planck*speed_of_light) /
  (3 * constexpr_functions::pow(transition_lambda, 3) * excited_state_lifetime);

static constexpr double saturation_intensity_inv = 1.0 / saturation_intensity;

static constexpr double cesium_mass_kg = 2.20694650e-25; // kg

static constexpr double recoil_velocity = transition_wave_vec * planck_reduced / cesium_mass_kg;

//
// Doppler effect
//
__forceinline __m256d dopplerEffect(__m256d const& velocities)
{
  return multiply(velocities, broadcast(transition_wave_vec));
}

//
// Zeeman shift
//
static constexpr double zeeman_slope = g_factor * bohr_magneton / planck_reduced;

__forceinline __m256d __vectorcall zeemanEffect(__m256d const& magnetic_fields)
{
  return multiply(broadcast(zeeman_slope), magnetic_fields);
}

//
// The probability that the atom is in excited state
// (Steady-state Bloch equations solution)
//
__forceinline __m256d __vectorcall excitedStateProbabilites(__m256d const& intensities, __m256d const& detunings)
{
  __m256d s = multiply(intensities, broadcast(saturation_intensity_inv));
  __m256d num = multiply(broadcast(0.5), s);

  __m256d twice_lifetime = broadcast(2.0*excited_state_lifetime);
  __m256d prod = multiply(twice_lifetime, detunings);
  
  __m256d prod_sq_plus_s = multiply_add(prod, prod, s);

  __m256d den = add(prod_sq_plus_s, broadcast(1.0));

  return _mm256_div_pd(num, den);
}

//
// Scatter rate 
//
__forceinline __m256d scatteringRate(__m256d const& intensities, __m256d const& detunings)
{
  __m256d probs = excitedStateProbabilites(intensities, detunings);
  return _mm256_mul_pd(probs, _mm256_set1_pd(natural_linewidth_hz));
}


__forceinline __m256d __vectorcall scatteringRate(
  __m256d const& velocity_component_along_light_direction,
  __m256d const& light_intensity,
  __m256d const& light_detuning_hz,
  __m256d const& magnetic_field_tesla)
{
  // shifts (angular frequencies)
  __m256d shift_doppler = dopplerEffect(velocity_component_along_light_direction);
  __m256d shift_zeeman = zeemanEffect(magnetic_field_tesla);
  // Convert detuning to angular frequency
  __m256d shift_detuning = multiply(light_detuning_hz, broadcast(two_pi));

  __m256d total_detuning = subtract(add(shift_zeeman, shift_doppler), shift_detuning);

  return scatteringRate(light_intensity, total_detuning);
}


#endif // PHYSICS_HPP