#ifndef MATHS_HPP
#define MATHS_HPP

#include <cmath>

#include <immintrin.h>


enum Directions
{
  X = 0,
  Y = 1,
  Z = 2,
  NDir = 3
};

//
// Alias type for quadruple of double values.
// 
typedef double QuadrupleScalar[4];

// 
// Alias type for static array of three quadruples
//
typedef QuadrupleScalar QuadrupleScalar3D[NDir];

//
// Alias type for static array of three AVX registers (three quadruples)
//
typedef __m256d QuadrupleAVX_3D[NDir];


#define for_every_direction(dir) for(int dir = 0; dir < NDir; ++dir)

#define for_every_quadruple_member(idx) for(int idx = 0; idx < 4; ++idx)

template<typename T>
T sq(T value)
{
  return value*value;
}


namespace constexpr_functions
{
  constexpr double pow(double x, int exp)
  {
    return exp == 0 ? 1.0 : x * pow(x, exp - 1);
  }
}


//
// Structure for work with 2-dimensional arrays
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

  double& operator()(ptrdiff_t pos0, ptrdiff_t pos1)
  {
    return data_[toIndex(pos0, pos1)];
  }

  ptrdiff_t size0_;
  ptrdiff_t size1_;
  ptrdiff_t stride_;
  std::vector<double> data_;
};



__forceinline __m256d __vectorcall reciprocal(__m256d const& value)
{
  return _mm256_div_pd(_mm256_set1_pd(1.0), value);
}

__forceinline __m256d __vectorcall multiply(__m256d const& value, __m256d const& other)
{
  return _mm256_mul_pd(value, other);
}

__forceinline __m256d __vectorcall multiply_add(__m256d const& mul0, __m256d const& mul1, __m256d const& add)
{
  return _mm256_fmadd_pd(mul0, mul1, add);
}

__forceinline __m256d __vectorcall add(__m256d const& a, __m256d const& b)
{
  return _mm256_add_pd(a, b);
}

__forceinline __m256d __vectorcall subtract(__m256d const& value, __m256d const& other)
{
  return _mm256_sub_pd(value, other);
}

__forceinline __m256d __vectorcall broadcast(double value)
{
  return _mm256_set1_pd(value);
}

__forceinline __m256d __vectorcall load(QuadrupleScalar const& value)
{
  return _mm256_load_pd(value);
}

__forceinline void __vectorcall store(__m256d const& value, QuadrupleScalar& dst)
{
  return _mm256_store_pd(dst, value);
}

__forceinline __m256d __vectorcall squareRoot(__m256d const& value)
{
  return _mm256_sqrt_pd(value);
}

//
// Computes dot products of 4 pairs of vectors in packed representation.
//
__forceinline __m256d __vectorcall dotProduct(
  QuadrupleAVX_3D const& a,
  QuadrupleAVX_3D const& b)
{
  __m256d partial = multiply_add(a[Y], b[Y], multiply(a[X], b[X]));

  return multiply_add(a[Z], b[Z], partial);
}

#endif // MATHS_HPP