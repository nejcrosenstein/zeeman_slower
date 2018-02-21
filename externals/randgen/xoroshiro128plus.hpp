#ifndef XOROSHIRO_128_PLUS_HPP
#define XOROSHIRO_128_PLUS_HPP

/*  Written in 2016 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

#include <stdint.h>
#include <exception>

/* This is the successor to xorshift128+. It is the fastest full-period
generator passing BigCrush without systematic failures, but due to the
relatively short period it is acceptable only for applications with a
mild amount of parallelism; otherwise, use a xorshift1024* generator.

Beside passing BigCrush, this generator passes the PractRand test suite
up to (and included) 16TB, with the exception of binary rank tests, as
the lowest bit of this generator is an LFSR of degree 128. The next bit
can be described by an LFSR of degree 8256, but in the long run it will
fail linearity tests, too. The other bits needs a much higher degree to
be represented as LFSRs.

We suggest to use a sign test to extract a random Boolean value, and
right shifts to extract subsets of bits.

Note that the generator uses a simulated rotate operation, which most C
compilers will turn into a single instruction. In Java, you can use
Long.rotateLeft(). In languages that do not make low-level rotation
instructions accessible xorshift128+ could be faster.

The state must be seeded so that it is not everywhere zero. If you have
a 64-bit seed, we suggest to seed a splitmix64 generator and use its
output to fill s. */


struct Xoroshiro
{
  Xoroshiro(uint64_t (&seed)[2]) :
    min_(uint64_t(0)),
    max_(~uint64_t(0))
  {
    state_[0] = seed[0];
    state_[1] = seed[1];
    if (state_[0] == 0 || state_[1] == 0)
    {
      std::exception("Xoroshiro seed must not equal zero.");
    }

    norm_inv_ = 1.0 / (double(max_) - double(min_));
  }

  typedef uint64_t result_type;

  uint64_t min() const
  {
    return min_;
  }

  uint64_t max() const
  {
    return max_;
  }

  double random(double lo, double hi)
  {
    return lo + double(this->operator()()) * norm_inv_ * (hi - lo);
  }

  static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
  }

  uint64_t operator()() {
    const uint64_t s0 = state_[0];
    uint64_t s1 = state_[1];
    const uint64_t result = s0 + s1;

    s1 ^= s0;
    state_[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
    state_[1] = rotl(s1, 36); // c

    return result;
  }


  /* This is the jump function for the generator. It is equivalent
  to 2^64 calls to next(); it can be used to generate 2^64
  non-overlapping subsequences for parallel computations. */
  void jump(void) {
    static const uint64_t JUMP[] = { 0xbeac0467eba5facb, 0xd86b048b86aa9922 };

    uint64_t s0 = 0;
    uint64_t s1 = 0;
    for (int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
      for (int b = 0; b < 64; b++) {
        if (JUMP[i] & UINT64_C(1) << b) {
          s0 ^= state_[0];
          s1 ^= state_[1];
        }
        this->operator()();
      }

    state_[0] = s0;
    state_[1] = s1;
  }

private:
  // State
  uint64_t state_[2];
  uint64_t min_;
  uint64_t max_;
  double norm_inv_;
};


struct XoroshiroSIMD
{
  XoroshiroSIMD(uint64_t(&seed)[2]) :
    min_(uint64_t(0)),
    max_(~uint64_t(0))
  {
    uint64_t curr_state0 = seed[0];
    uint64_t curr_state1 = seed[1];

    alignas(32) uint64_t states0[4];
    alignas(32) uint64_t states1[4];
    for (int i = 0; i < 4; ++i)
    {
      states0[i] = curr_state0;
      states1[i] = curr_state1;

      jump(curr_state0, curr_state1);
    }
    
    state_[0] = _mm256_load_si256((__m256i*)&states0[0]);
    state_[1] = _mm256_load_si256((__m256i*)&states1[0]);
    
    if (seed[0] == 0 || seed[1] == 0)
    {
      std::exception("Xoroshiro seed must not equal zero.");
    }

    norm_inv_ = 1.0 / (double(max_) - double(min_));
  }

  __m256d random_simd(double lo, double hi)
  {
    __m256i irnd = generate_simd();

    alignas(32) uint64_t vals[4];
    _mm256_store_si256((__m256i*)&vals[0], irnd);

    alignas(32) double cvt[4];
    for (int i = 0; i < 4; ++i)
      cvt[i] = double(vals[i]);

    __m256d drnd = _mm256_load_pd(cvt);

    return _mm256_add_pd(
      _mm256_set1_pd(lo),
      _mm256_mul_pd(drnd,
        _mm256_set1_pd(norm_inv_*(hi - lo))));
  }

  __m256d random_simd()
  {
    __m256i irnd = generate_simd();

    alignas(32) uint64_t vals[4];
    _mm256_store_si256((__m256i*)&vals[0], irnd);

    alignas(32) double cvt[4];
    for (int i = 0; i < 4; ++i)
      cvt[i] = double(vals[i]);

    __m256d drnd = _mm256_load_pd(cvt);

    return _mm256_mul_pd(drnd, _mm256_set1_pd(norm_inv_));
  }

  double random(double lo, double hi)
  {
    alignas(32) double drnd[4];
    _mm256_store_pd(drnd, random_simd(lo, hi));

    return drnd[0];
  }

  typedef uint64_t result_type;

  uint64_t min() const
  {
    return min_;
  }

  uint64_t max() const
  {
    return max_;
  }

  double operator()()
  {
    alignas(32) uint64_t vals[4];
    _mm256_store_si256((__m256i*)&vals[0], generate_simd());

    return vals[0];
  }

  template<int k>
  static __forceinline __m256i __vectorcall rotl(__m256i const& x)
  {
    return _mm256_or_si256(
             _mm256_slli_epi64(x, k),
             _mm256_srli_epi64(x, 64 - k));
  }

  static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
  }

  __m256i generate_simd() 
  {  
    const __m256i s0 = state_[0];
    __m256i s1 = state_[1];
    const __m256i result = _mm256_add_epi64(s0, s1);

    s1 = _mm256_xor_si256(s0, s1);

    state_[0] =
      _mm256_xor_si256(
        _mm256_xor_si256(
          rotl<55>(s0), s1),
        _mm256_slli_epi64(s1, 14));

    state_[1] = rotl<36>(s1); // c

    return result;
  }

  uint64_t generate_scalar(uint64_t& state0, uint64_t& state1) {
    const uint64_t s0 = state0;
    uint64_t s1 = state1;
    const uint64_t result = s0 + s1;

    s1 ^= s0;
    state0 = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
    state1 = rotl(s1, 36); // c

    return result;
  }


  /* This is the jump function for the generator. It is equivalent
  to 2^64 calls to next(); it can be used to generate 2^64
  non-overlapping subsequences for parallel computations. */
  void jump(uint64_t& state0, uint64_t& state1) {
    static const uint64_t JUMP[] = { 0xbeac0467eba5facb, 0xd86b048b86aa9922 };

    uint64_t s0 = 0;
    uint64_t s1 = 0;
    for (int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
      for (int b = 0; b < 64; b++) {
        if (JUMP[i] & UINT64_C(1) << b) {
          s0 ^= state0;
          s1 ^= state1;
        }
        this->generate_scalar(state0, state1);
      }

    state0 = s0;
    state1 = s1;
  }

  void jump(int offset, int factor)
  {
    alignas(32) uint64_t s0[4];
    alignas(32) uint64_t s1[4];
    _mm256_store_si256((__m256i*)&s0[0], state_[0]);
    _mm256_store_si256((__m256i*)&s1[0], state_[1]);
    
    for (int j = 0; j < 4; ++j)
    {
      for (int i = 0; i < offset + j*factor; ++i)
      {
        jump(s0[j], s1[j]);
      }
    }

    state_[0] = _mm256_load_si256((__m256i*)&s0[0]);
    state_[1] = _mm256_load_si256((__m256i*)&s1[0]);
  }

private:
  // State
  __m256i state_[2];
  uint64_t min_;
  uint64_t max_;
  double norm_inv_;
};

#endif