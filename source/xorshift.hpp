#ifndef XORSHIFT_HPP
#define XORSHIFT_HPP

#include <stdint.h>
#include <exception>

#include <immintrin.h>

// xorshift* algorithm (Marsaglia, 2003)
//
// Implementation taken from Wikipedia:
// https://en.wikipedia.org/wiki/Xorshift#xorshift

struct XorshiftStar
{
  XorshiftStar(uint64_t seed) :
    s_(seed),
    min_( uint32_t(0)),
    max_(~uint32_t(0))
  {
    if (s_ == 0)
    {
      std::exception("Xorshift star seed must not equal zero.");
    }

    norm_inv_ = 1.0 / (double(max_) - double(min_));
  }

  typedef uint64_t result_type;

  uint32_t min() const
  {
    return min_;
  }

  uint32_t max() const
  {
    return max_;
  }

  uint32_t operator()() {
    s_ ^= s_ >> 12;
    s_ ^= s_ << 25; 
    s_ ^= s_ >> 27; 
    return uint32_t((s_ * 0x2545F4914F6CDD1Du) >> 32);
  }

  double random(double lo, double hi)
  {
    return lo + double(this->operator()()) * norm_inv_ * (hi - lo);
  }

private :
  // State
  uint64_t s_;
  uint32_t min_;
  uint32_t max_;
  double norm_inv_;
};

#endif // BEAM_HPP