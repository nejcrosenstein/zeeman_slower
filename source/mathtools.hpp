#ifndef MATHS_HPP
#define MATHS_HPP

#include <cmath>
#include <optional>

#include <immintrin.h>

template<typename T>
T sq(T value)
{
  return value*value;
}

template<typename T>
struct Vec3D
{
  union
  {
    struct { 
      T x_;
      T y_;
      T z_;
    };
    T components[3];
  };

  Vec3D() {};

  Vec3D(T val) : 
    x_(val), y_(val), z_(val){}

  Vec3D(T x, T y, T z) :
    x_(x), y_(y), z_(z) {}

  Vec3D operator*(T val) const
  {
    return Vec3D(val*x_, val*y_, val*z_);
  }

  Vec3D operator/(T val) const
  {
    T rcp = T(1) / val;
    return (*this)*rcp;
  }

  void operator*=(T val) const
  {
    this->x_ *= val;
    this->y_ *= val;
    this->z_ *= val;
  }

  Vec3D operator+(Vec3D const& other) const
  {
    return Vec3D(
      other.x_ + this->x_,
      other.y_ + this->y_,
      other.z_ + this->z_
      );
  }

  void operator+=(Vec3D const& other) 
  {
    this->x_ += other.x_;
    this->y_ += other.y_;
    this->z_ += other.z_;
  }

  double norm_sq() const
  {
    return sq(x_) + sq(y_) + sq(z_);
  }

  double norm() const
  {
    return std::sqrt(norm_sq());
  }

  std::optional<Vec3D> normalized() const
  {
    double norm_value = this->norm();

    return norm_value == 0.0 ?
      std::optional<Vec3D>() : (*this)/norm_value;
  }

  double projectionComponentAlongDirection(Vec3D<T> const& dir_norm) const
  {
    return
      (x_*dir_norm.x_ + y_ * dir_norm.y_ + z_ * dir_norm.z_);
  }
};


template<typename T>
static std::optional<double> dotProduct(Vec3D<T> const& a, Vec3D<T> const& b)
{
  double denom = std::sqrt(a.norm_sq() * b.norm_sq());

  return denom > 0.0 ?
    (a.x_*b.x_ + a.y_*b.y_ + a.z_*b.z_) / denom :
    std::nullopt;
}

namespace constexpr_functions
{
  constexpr double pow(double x, int exp)
  {
    return exp == 0 ? 1.0 : x * pow(x, exp - 1);
  }
}

// works OK for 0 < theta < pi
Vec3D<double> to_cartesian(double phi, double cos_theta)
{
  using namespace std;
  return
  {
    cos(phi)*sqrt(1.0 - sq(cos_theta)),
    sin(phi)*sqrt(1.0 - sq(cos_theta)),
    cos_theta
  };
}

template<typename T>
__forceinline T index_clamp(T index, T low, T high)
{
  return std::min(std::max(index, low), high - 1);
}

//
// Structures and functions for work with 2-dimensional arrays
//

struct Arr2D
{
  ptrdiff_t toIndex(ptrdiff_t pos0, ptrdiff_t pos1) const
  {
    return pos0 * stride_ + pos1;
  }

  void resize(ptrdiff_t size0, ptrdiff_t size1)
  {
    size0_ = size0;
    size1_ = size1;
    stride_ = size1;
    data_.resize(size0_*size1_);
  }

  double operator()(ptrdiff_t pos0, ptrdiff_t pos1) const
  {
    return data_[toIndex(pos0, pos1)];
  }

  double& operator()(ptrdiff_t pos0, ptrdiff_t pos1)
  {
    return data_[toIndex(pos0, pos1)];
  }

  ptrdiff_t size0_;
  ptrdiff_t size1_;
  ptrdiff_t stride_;
  std::vector<double> data_;
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


__forceinline __m256d __vectorcall gatherHelper(Arr2D const& src, __m128i const& ix0_32, __m128i const& ix1_32, __m128i const& stride_32)
{
  __m256i ix0 = _mm256_cvtepi32_epi64(ix0_32);
  __m256i ix1 = _mm256_cvtepi32_epi64(ix1_32);
  __m256i stride = _mm256_cvtepi32_epi64(stride_32);

  __m256i indices =
    _mm256_add_epi64(ix1, _mm256_mul_epi32(ix0, stride));

  return _mm256_i64gather_pd(&src.data_[0], ix1, sizeof(double));
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
  __m256d w0_lo = _mm256_sub_pd(pos0, pos0_lo);
  __m256d w0_hi = _mm256_sub_pd(one, w0_lo);

  __m256d w1_lo = _mm256_sub_pd(pos1, pos1_lo);
  __m256d w1_hi = _mm256_sub_pd(one, w1_lo);

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

  __m128i cl0_lo = _mm_min_epi32(_mm_max_epi32(ix0_lo, izero), hi0);
  __m128i cl0_hi = _mm_min_epi32(_mm_max_epi32(ix0_hi, izero), hi0);
  __m128i cl1_lo = _mm_min_epi32(_mm_max_epi32(ix1_lo, izero), hi1);
  __m128i cl1_hi = _mm_min_epi32(_mm_max_epi32(ix1_hi, izero), hi1);

  __m128i strides = _mm_set1_epi32(arr.size1_);
  __m256d v00 = gatherHelper(arr, cl0_lo, cl1_lo, strides);
  __m256d v01 = gatherHelper(arr, cl0_lo, cl1_hi, strides);
  __m256d v10 = gatherHelper(arr, cl0_hi, cl1_lo, strides);
  __m256d v11 = gatherHelper(arr, cl0_hi, cl1_hi, strides);

  __m256d mul00 = _mm256_mul_pd(w0_lo, _mm256_mul_pd(w1_lo, v00));
  __m256d mul01 = _mm256_mul_pd(w0_lo, _mm256_mul_pd(w1_hi, v01));
  __m256d mul10 = _mm256_mul_pd(w0_hi, _mm256_mul_pd(w1_lo, v10));
  __m256d mul11 = _mm256_mul_pd(w0_hi, _mm256_mul_pd(w1_hi, v11));

  return _mm256_add_pd(_mm256_add_pd(mul00, mul01), _mm256_add_pd(mul10, mul11));
}


#endif // MATHS_HPP