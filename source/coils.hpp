#ifndef COILS_HPP
#define COILS_HPP

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>

#include "physics.hpp"
#include "mathtools.hpp"


struct MagneticField1D
{
  double start_z_;
  double step_z_;
  double step_z_inv_;

  Arr2D magnetic_fields_tesla_;
};

struct ZeemanSlower : public MagneticField1D
{
  static constexpr double tube_radius_ = 0.02;
  static constexpr double wire_width_ = 0.0025;
  static constexpr double wire_thickness_ = 0.001;

  static constexpr double lengths_m_[9] =
  {
    0.63, 0.57, 0.5, 0.42, 0.36, 0.29, 0.22, 0.14, 0.06
  };

  static constexpr int winding_layers_[9] =
  {
    3, 6, 4, 2, 2, 2, 2, 2, 5
  };

  double solenoidField(double zpos, double inner_rad, double length, int num_layers, double current)
  {
    double total_field = 0.0;

    double start = zpos - 0.5*wire_width_;
    double end = zpos - (length + 0.5*wire_width_);

    for (int i = 0; i < num_layers; ++i)
    {
      double layer_radius = inner_rad + wire_thickness_ * (double(i) + 0.5);
      double part0 = start / sqrt(sq(start) + sq(layer_radius));
      double part1 = end / sqrt(sq(end) + sq(layer_radius));
      total_field += 0.5*(part0 - part1) * current * magnetic_constant / wire_width_;
    }

    return total_field;
  }

  ZeemanSlower(double current_bias_ampere, double current_profile_ampere, double start_z, double end_z, int num_points)
  {
    std::vector<double> currents(9);
    currents[0] = current_bias_ampere;
    for (int i = 1; i < 9; ++i)
      currents[i] = current_profile_ampere;

    magnetic_fields_tesla_.resize(1, num_points);

    this->start_z_ = start_z;
    this->step_z_ = (end_z - start_z) / double(num_points);
    this->step_z_inv_ = 1.0 / step_z_;

    for (int j = 0; j < num_points; ++j)
    {
      double z = start_z_ + step_z_ * double(j);

      double total_field = 0.0;
      double inner_rad = tube_radius_;

      for (int i = 0; i < 9; ++i)
      {
        double current = currents[i];
        double length = lengths_m_[i];
        double winds = winding_layers_[i];

        total_field += solenoidField(z, inner_rad, length, winds, current);

        inner_rad += winds * wire_thickness_;
      }

      magnetic_fields_tesla_(0, j) = total_field;
    }
  }
};


struct ImportedField : public MagneticField1D
{
  ImportedField(double distance_from_slower_entry, const char* file_name)
  {
    std::ifstream in(file_name);

    std::vector<double> ax_poss;
    std::vector<double> fields;
    while (!in.eof())
    {
      double z;
      double b;
      in >> z;
      in >> b;

      ax_poss.push_back(z + distance_from_slower_entry);
      fields.push_back(b);
    } 

    magnetic_fields_tesla_.resize(1, fields.size());
    for (int i = 0; i < fields.size(); ++i)
      magnetic_fields_tesla_(0, i) = fields[i];

    std::vector<double> diffs(ax_poss.size());
    std::adjacent_difference(ax_poss.begin(), ax_poss.end(), diffs.begin());

    for (int i = 1; i < diffs.size(); ++i)
    {
      if (diffs[i] < 1.0e-6)
      {
        std::cerr << "Difference between axis points is greater than 1um" << std::endl;
      }
    }

    this->start_z_ = ax_poss[0];
    this->step_z_ = diffs[1];
    this->step_z_inv_ = 1.0 / step_z_;
  }
};



//
// Interpolation (with clamping)
//

namespace debug
{
  void print(__m256d vals)
  {
    double deb[4];
    _mm256_storeu_pd(&deb[0], vals);
    std::cout << deb[0] << "  " << deb[1] << "  " << deb[2] << "  " << deb[3] << std::endl;
  }

  void print(__m256i vals)
  {
    int64_t deb[4];
    _mm256_storeu_si256((__m256i*)&deb[0], vals);
    std::cout << deb[0] << "  " << deb[1] << "  " << deb[2] << "  " << deb[3] << std::endl;
  }
}

__forceinline __m128i __vectorcall clamp32(__m128i const& values, __m128i const& lo, __m128i const& hi)
{
  return _mm_min_epi32(_mm_max_epi32(values, lo), hi);
}

__forceinline __m256d __vectorcall gatherHelper(Arr2D const& src, __m128i const& ix0_32, __m128i const& ix1_32, __m128i const& stride_32)
{
  __m256i ix0 = _mm256_cvtepi32_epi64(ix0_32);
  __m256i ix1 = _mm256_cvtepi32_epi64(ix1_32);
  __m256i stride = _mm256_cvtepi32_epi64(stride_32);

  __m256i indices =
    _mm256_add_epi64(ix1, _mm256_mul_epi32(ix0, stride));

  return _mm256_i64gather_pd(&src.data_[0], indices, sizeof(double));
}

__forceinline __m256d __vectorcall interpolate(
  Arr2D const& arr,
  __m256d const& pos0,
  __m256d const& pos1)
{
  constexpr int rm = _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC;

  __m256d pos0_lo = _mm256_round_pd(pos0, rm);
  __m256d pos1_lo = _mm256_round_pd(pos1, rm);

  // Interpolation weights
  __m256d one = _mm256_set1_pd(1.0);
  __m256d w0_hi = _mm256_sub_pd(pos0, pos0_lo);
  __m256d w0_lo = _mm256_sub_pd(one, w0_hi);

  __m256d w1_hi = _mm256_sub_pd(pos1, pos1_lo);
  __m256d w1_lo = _mm256_sub_pd(one, w1_hi);

  // Interpolation indices
  __m128i ix0_lo = _mm256_cvtpd_epi32(pos0_lo);
  __m128i ix1_lo = _mm256_cvtpd_epi32(pos1_lo);

  __m128i ione = _mm_set1_epi32(1);
  __m128i ix0_hi = _mm_add_epi32(ix0_lo, ione);
  __m128i ix1_hi = _mm_add_epi32(ix1_lo, ione);

  // Clamping
  __m128i izero = _mm_setzero_si128();
  __m128i hi0 = _mm_set1_epi32(arr.size0_ - 1);
  __m128i hi1 = _mm_set1_epi32(arr.size1_ - 1);

  __m128i cl0_lo = clamp32(ix0_lo, izero, hi0);
  __m128i cl0_hi = clamp32(ix0_hi, izero, hi0);
  __m128i cl1_lo = clamp32(ix1_lo, izero, hi1);
  __m128i cl1_hi = clamp32(ix1_hi, izero, hi1);

  __m128i strides = _mm_set1_epi32(arr.size1_);
  __m256d v00 = gatherHelper(arr, cl0_lo, cl1_lo, strides);
  __m256d v01 = gatherHelper(arr, cl0_lo, cl1_hi, strides);
  __m256d v10 = gatherHelper(arr, cl0_hi, cl1_lo, strides);
  __m256d v11 = gatherHelper(arr, cl0_hi, cl1_hi, strides);

  __m256d mul00 = multiply(w0_lo, multiply(w1_lo, v00));
  __m256d mul01 = multiply(w0_lo, multiply(w1_hi, v01));
  __m256d mul10 = multiply(w0_hi, multiply(w1_lo, v10));
  __m256d mul11 = multiply(w0_hi, multiply(w1_hi, v11));

  return add(add(mul00, mul01), add(mul10, mul11));
}


__forceinline __m256d __vectorcall interpolate(
  MagneticField1D const& field,
  __m256d const (&pos)[NDir])
{
  __m256d interp_pos_z =
    _mm256_mul_pd(
      broadcast(field.step_z_inv_),
      subtract(pos[Z], broadcast(field.start_z_)));

  __m256d interp_pos_r = _mm256_setzero_pd();

  return interpolate(field.magnetic_fields_tesla_, interp_pos_r, interp_pos_z);
}

double interpolate(
  MagneticField1D const& field,
  double pos_z)
{
  // this is probably quite ineffective, but no biggie:
  // this function is only called at the beginning of the simulation
  // and measurements show that it is not a bottleneck anyway
  __m256d pos[3] =
  {
    _mm256_setzero_pd(),
    _mm256_setzero_pd(),
    _mm256_set1_pd(pos_z)
  };

  __m256d itp = interpolate(
    field, pos);
  
  alignas(32) double vals[4];
  _mm256_store_pd(vals, itp);

  return vals[0];
}


#endif // COILS_HPP