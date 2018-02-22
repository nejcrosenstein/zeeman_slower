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
  return _mm256_mul_pd(velocities, _mm256_set1_pd(transition_wave_vec));
}

__forceinline double dopplerEffect(double velocity_in_light_direction)
{
  return transition_wave_vec * velocity_in_light_direction;
}

//
// Zeeman shift
//
static constexpr double zeeman_slope = g_factor * bohr_magneton / planck_reduced;

__forceinline __m256d __vectorcall zeemanEffect(__m256d const& magnetic_fields)
{
  return _mm256_mul_pd(_mm256_set1_pd(zeeman_slope), magnetic_fields);
}

__forceinline double zeemanEffect(double magField)
{
  return zeeman_slope * magField;
}

//
// The probability that the atom is in excited state
// (Steady-state Bloch equations solution)
//
__forceinline __m256d __vectorcall excitedStateProbabilites(__m256d const& intensities, __m256d const& detunings)
{
  __m256d s = _mm256_mul_pd(intensities, _mm256_set1_pd(saturation_intensity_inv));
  __m256d num = _mm256_mul_pd(_mm256_set1_pd(0.5), s);

  __m256d twice_lifetime = _mm256_set1_pd(2.0*excited_state_lifetime);
  __m256d prod = _mm256_mul_pd(twice_lifetime, detunings);
  
  __m256d prod_sq_plus_s = _mm256_fmadd_pd(prod, prod, s);

  __m256d den = _mm256_add_pd(prod_sq_plus_s, _mm256_set1_pd(1.0));

  return _mm256_div_pd(num, den);
}

__forceinline double excitedStateProbability(double light_intensity, double light_detuning_hz)
{
  double s = light_intensity * saturation_intensity_inv;
  return 0.5*s / (1.0 + s + sq(2.0 * light_detuning_hz * excited_state_lifetime));
}

//
// Scatter rate 
//
__forceinline __m256d scatteringRate(__m256d const& intensities, __m256d const& detunings)
{
  __m256d probs = excitedStateProbabilites(intensities, detunings);
  return _mm256_mul_pd(probs, _mm256_set1_pd(natural_linewidth_hz));
}

__forceinline double scatteringRate(double light_intensity, double light_detuning_angular_hz)
{
  return natural_linewidth_hz * excitedStateProbability(light_intensity, light_detuning_angular_hz);
}

//
// Conversion from frequency to angular frequency
//
__forceinline __m256d __vectorcall frequency_to_angular_freq(__m256d const& freq)
{
  return _mm256_mul_pd(freq, _mm256_set1_pd(two_pi));
}

__forceinline double frequency_to_angular_freq(double freq)
{
  return two_pi * freq;
}

__forceinline __m256d __vectorcall sub(__m256d const& a, __m256d const& b)
{
  return _mm256_sub_pd(a, b);
}

__forceinline double sub(double a, double b)
{
  return a + b;
}

__forceinline __m256d __vectorcall add(__m256d const& a, __m256d const& b)
{
  return _mm256_add_pd(a, b);
}

__forceinline double add(double a, double b)
{
  return a + b;
}

template<typename T>
__forceinline T __vectorcall scatteringRate(
  T const& velocity_component_along_light_direction,
  T const& light_intensity,
  T const& light_detuning_hz,
  T const& magnetic_field_tesla)
{
  // shifts (angular frequencies)
  T shift_doppler = dopplerEffect(velocity_component_along_light_direction);
  T shift_zeeman = zeemanEffect(magnetic_field_tesla);
  T shift_detuning = frequency_to_angular_freq(light_detuning_hz);

  T total_detuning = sub(add(shift_zeeman, shift_doppler), shift_detuning);

  return scatteringRate(light_intensity, total_detuning);
}


#endif // PHYSICS_HPP