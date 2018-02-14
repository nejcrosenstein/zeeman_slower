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



#endif // MATHS_HPP