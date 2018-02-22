#include <iostream>
#include <random>
#include <fstream>
#include <array>
#include <thread>

#include "mathtools.hpp"
#include "physics.hpp"
#include "coils.hpp"
#include "beam.hpp"
#include "histogram.hpp"
#include "laser.hpp"

#include "../externals/randgen/xoroshiro128plus.hpp"

using Uniform = std::uniform_real_distribution<double>;
using Exponential = std::exponential_distribution<double>;

using Vecd = Vec3D<double>;

namespace parameters
{
  static constexpr double oven_temperature_kelvin = 353.0;
}

static constexpr double oven_exit_radius_m = 0.003;

enum Directions
{
  X = 0,
  Y = 1,
  Z = 2,
  NDir = 3
};


struct ParticleQuadruple
{
  alignas(32) double pos_x[4];
  alignas(32) double pos_y[4];
  alignas(32) double pos_z[4];

  alignas(32) double vel_x[4];
  alignas(32) double vel_y[4];
  alignas(32) double vel_z[4];

  void getPositions(__m256d (&pos)[NDir]) const
  {
    pos[X] = _mm256_load_pd(pos_x);
    pos[Y] = _mm256_load_pd(pos_y);
    pos[Z] = _mm256_load_pd(pos_z);
  }

  void setPositions(__m256d const (&pos)[NDir])
  {
    _mm256_store_pd(pos_x, pos[X]);
    _mm256_store_pd(pos_y, pos[Y]);
    _mm256_store_pd(pos_z, pos[Z]);
  }

  void getVelocities(__m256d (&vel)[NDir]) const
  {
    vel[X] = _mm256_load_pd(vel_x);
    vel[Y] = _mm256_load_pd(vel_y);
    vel[Z] = _mm256_load_pd(vel_z);
  }

  void setVelocities(__m256d const (&vel)[NDir])
  {
    _mm256_store_pd(vel_x, vel[X]);
    _mm256_store_pd(vel_y, vel[Y]);
    _mm256_store_pd(vel_z, vel[Z]);
  }
};

__forceinline __m256d __vectorcall dotProduct(
  const __m256d(&a)[NDir],
  const __m256d(&b)[NDir])
{
  // Some micro optimization is possible (fmadd), but
  // the effect will probably not be noticeable
  __m256d prod0 = _mm256_mul_pd(a[X], b[X]);
  __m256d prod1 = _mm256_mul_pd(a[Y], b[Y]);
  __m256d prod2 = _mm256_mul_pd(a[Z], b[Z]);

  return _mm256_add_pd(prod0, _mm256_add_pd(prod1, prod2));
}

__forceinline void __vectorcall computeLightDirection(
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

template<class RandomGen_t>
static void takeOneStep(
  ParticleQuadruple& atoms,
  RandomGen_t& rand_gen,
  ZeemanSlower const& slower,
  ImportedField const& quadrupole, 
  double time_step_s)
{ 
  // Current position and velocity
  __m256d curr_pos[NDir];
  atoms.getPositions(curr_pos);
  
  __m256d curr_vel[NDir];
  atoms.getVelocities(curr_vel);

  __m256d intensity = lightIntensity(curr_pos[X], curr_pos[Y], curr_pos[Z]);
  // TODO: make a parameter
  __m256d detuning = _mm256_set1_pd(10.0e6);

  __m256d r_positions = _mm256_setzero_pd();

  // TODO: combine both fields before simulation starts
  __m256d slower_fields = interpolate(slower, r_positions, curr_pos[Z]);
  __m256d quad_fields = interpolate(quadrupole, r_positions, curr_pos[Z]);
  __m256d field_tesla = _mm256_add_pd(slower_fields, quad_fields);
 
  __m256d light_dir[NDir];
  computeLightDirection(curr_pos, light_dir);

  __m256d vel_along_light_dir = dotProduct(curr_vel, light_dir);
  
  __m256d scatter_rates = scatteringRate(vel_along_light_dir, intensity, detuning, field_tesla);

  __m256d time_step = _mm256_set1_pd(time_step_s);

  __m256d new_pos[NDir];
  new_pos[X] = _mm256_fmadd_pd(curr_vel[X], time_step, curr_pos[X]);
  new_pos[Y] = _mm256_fmadd_pd(curr_vel[Y], time_step, curr_pos[Y]);
  new_pos[Z] = _mm256_fmadd_pd(curr_vel[Z], time_step, curr_pos[Z]);

  atoms.setPositions(new_pos);

  __m256d probs = _mm256_mul_pd(scatter_rates, time_step);

  __m256d uniform = rand_gen.random_simd();
  __m256d compared = _mm256_cmp_pd(uniform, probs, _CMP_LE_OQ);

  int mmask = _mm256_movemask_pd(compared);

  if(mmask)
  { 
    // TODO: could probably do some cleanup below
    double vel_change_mags[4] = { 0, 0, 0, 0 };
    double emm_dir_x[4] = { 0, 0, 0, 0 };
    double emm_dir_y[4] = { 0, 0, 0, 0 };
    double emm_dir_z[4] = { 0, 0, 0, 0 };
    
    double phi[4];
    double ctheta[4];
    __m256d vecphi = rand_gen.random_simd(0.0, two_pi);
    __m256d vecctheta = rand_gen.random_simd(-1.0, 1.0);
    _mm256_store_pd(phi, vecphi);
    _mm256_store_pd(ctheta, vecctheta);

    for (int i = 0; i < 4; ++i)
    {
      bool scatter_occurs = (mmask & (1 << i));
      if (scatter_occurs)
      {
        emm_dir_x[i] = cos(phi[i])*sqrt(1.0 - sq(ctheta[i]));
        emm_dir_y[i] = sin(phi[i])*sqrt(1.0 - sq(ctheta[i]));
        emm_dir_z[i] = ctheta[i];

        vel_change_mags[i] = recoil_velocity; // else remains 0
      }
    }

    __m256d change_dir[NDir];
    change_dir[X] = _mm256_sub_pd(light_dir[X], _mm256_load_pd(emm_dir_x));
    change_dir[Y] = _mm256_sub_pd(light_dir[Y], _mm256_load_pd(emm_dir_y));
    change_dir[Z] = _mm256_sub_pd(light_dir[Z], _mm256_load_pd(emm_dir_z));

    __m256d vel_recoil = _mm256_load_pd(&vel_change_mags[0]);
    
    __m256d new_vel[NDir];
    new_vel[X] = _mm256_fmadd_pd(change_dir[X], vel_recoil, curr_vel[X]);
    new_vel[Y] = _mm256_fmadd_pd(change_dir[Y], vel_recoil, curr_vel[Y]); 
    new_vel[Z] = _mm256_fmadd_pd(change_dir[Z], vel_recoil, curr_vel[Z]);

    atoms.setVelocities(new_vel);
  }
}

template <typename T, typename Compare>
std::vector<std::size_t> sort_permutation(
  const std::vector<T>& vec,
  Compare& compare)
{
  std::vector<std::size_t> p(vec.size());
  std::iota(p.begin(), p.end(), 0);
  std::sort(p.begin(), p.end(),
    [&](std::size_t i, std::size_t j) { return compare(vec[i], vec[j]); });
  return p;
}

struct InitialStates
{
  template<class RandGen_t>
  InitialStates(
    BeamVelocityDistribution const& init_velocity, int number, RandGen_t& rand_gen)
  {
    Uniform phi(0.0, two_pi);
    Uniform cos_theta_initvel(cos(beam_spread_angle_rad_), 1.0);
    Uniform zero_to_one(0.0, 1.0);
    
    for (int i = 0; i < number; ++i)
    {
      double vel_magnitude = init_velocity(rand_gen);

      Vecd vel_init = to_cartesian(phi(rand_gen), cos_theta_initvel(rand_gen))*vel_magnitude;

      double angle = phi(rand_gen);
      double two_rand_sum = zero_to_one(rand_gen) + zero_to_one(rand_gen);
      double r = two_rand_sum > 1 ? 2.0 - two_rand_sum : two_rand_sum;
      double rad = r * oven_exit_radius_m;

      Vecd pos_init = Vecd(rad*cos(angle), rad*sin(angle), -0.2);

      positions_.push_back(pos_init);
      velocities_.push_back(vel_init);
    }

    std::vector<std::size_t> per(number);
    std::iota(per.begin(), per.end(), 0);
    std::sort(per.begin(), per.end(),
      [this](std::size_t i, std::size_t j) { return this->velocities_[i].z_ < this->velocities_[j].z_; });


    std::vector<Vecd> sorted(number);
    for (int i = 0; i < number; ++i) sorted[i] = velocities_[per[i]];
    velocities_ = sorted;


    for (int i = 0; i < number; ++i) sorted[i] = positions_[per[i]];
    positions_ = sorted;

    taken_ = 0;
  }

  int taken_;
  std::vector<Vecd> positions_;
  std::vector<Vecd> velocities_;
};

template<class RanGen_t, class StopCondition>
void simulateQuadruplePath(RanGen_t& rand_gen,
  ZeemanSlower const& slower,
  ImportedField const& quadrupole,
  InitialStates& init_states,
  std::array<Histogram, 4>& hists,
  StopCondition const& stop_condition)
{
  ParticleQuadruple atoms;

  for (int i = 0; i < 4; ++i)
  {
    int& ix = init_states.taken_;

    auto pos = init_states.positions_[ix];
    auto vel = init_states.velocities_[ix];

    atoms.pos_x[i] = pos.x_;
    atoms.pos_y[i] = pos.y_;
    atoms.pos_z[i] = pos.z_;

    atoms.vel_x[i] = vel.x_;
    atoms.vel_y[i] = vel.y_;
    atoms.vel_z[i] = vel.z_;

    init_states.taken_++;
  }

  double time_step = 0.1*excited_state_lifetime;
  
  for(;;)
  {
    int at_end = 0;
    for (int i = 0; i < 4; ++i)
    {
      hists[i].addSample(atoms.pos_z[i], atoms.vel_z[i]);
      at_end += (int)stop_condition(atoms, i);
    }
    if (at_end == 4) break;

    takeOneStep(atoms, rand_gen, slower, quadrupole, time_step);
  }
}

// just pass everything by copy
void simulationSingleThreaded(
  XoroshiroSIMD rand_gen, 
  ZeemanSlower slower, 
  ImportedField quadrupole, 
  int number_of_atoms,
  Histogram* dst)
{
  BeamVelocityDistribution init_velocity_dist(parameters::oven_temperature_kelvin);

  InitialStates init_states(init_velocity_dist, number_of_atoms, rand_gen);

  Histogram hist(-50.5, 1.0, 499.5, -0.2, 0.001, 1000);

  std::array<Histogram, 4> hists = { hist, hist, hist, hist };

  auto stop_condition = [](ParticleQuadruple const& atoms, int idx) 
  {
    return (atoms.pos_z[idx] > 0.8) || (atoms.vel_z[idx] < 0);
  };

  for (int j = 0; j < number_of_atoms/4; ++j)
  {
    simulateQuadruplePath(
      rand_gen, 
      slower, quadrupole, 
      init_states, hists, 
      stop_condition);

    for (auto const& h : hists)
      for (int i = 0; i < h.histogram_.size(); ++i)
        hist.histogram_[i] += h.histogram_[i];

    std::cout << " iteration  " << j << std::endl;
  }

  *dst = hist;
}

void joinAndClearThreads(std::vector<std::thread>& threads)
{
  for (auto& thread : threads)
  {
    thread.join();
  }
  threads.clear();
}

void simulation(ptrdiff_t number_of_threads, size_t total_number_of_atoms)
{
  //
  // Compute seed for first random generator
  //
  std::random_device rd;
  std::uniform_int_distribution<uint64_t> dist;
  uint64_t seed[2];
  seed[0] = dist(rd);
  seed[1] = dist(rd);
  //
  // Create the generator. Each instance of XoroshiroSIMD contains
  // four random generators. In constructor, they are initially seeded
  // with same values. Immediately after that, jump instructions 
  // are used on them: the N-th generator "jumps" N times.
  // 
  // Every time the jump instruction is used, the random generator 
  // internal state is changed. The new change equals the state
  // that would otherwise be reached after 2^64 random number generations.
  //
  // In this simulation we will never get anywhere near 2^64 generated
  // random numbers. That's why these jumps allow us to use four generators 
  // in parallel without any fear that any two sequences generated by two random
  // generators would be the same.
  //
  XoroshiroSIMD random_gen(seed); 

  //
  // Create one random generator for every thread:
  // make N copies of the generator created above and then
  // use jump instructions. 
  // 
  // This time, all four generators inside XoroshiroSIMD 
  // instance jump forward the same ammount of steps.
  //
  // After the end of this step, the generators should
  // have the following internal states (J = 2^64):
  // 
  // SIMD generator 0: 4 generators, advanced by {0J, 1J, 2J, 3J)
  // SIMD generator 1: 4 generators, advanced by {4J, 5J, 6J, 7J)
  // etc.
  //
  std::vector<XoroshiroSIMD> generators;
  for(size_t idx = 0; idx < number_of_threads; ++idx)
  {
    generators.emplace_back(random_gen).jump(4*idx, 1);
  }

  // TODO: create outside this function
  ZeemanSlower slower(0, 0, -0.3, 1.0, 2000);
  ImportedField quadrupole(0.0, "ideal.txt");
  
  int atoms_per_thread = total_number_of_atoms / number_of_threads;

  std::vector<Histogram> hists(number_of_threads);
  std::vector<std::thread> threads;
  for (size_t i = 0; i < number_of_threads; ++i)
  {
    threads.emplace_back(
      std::thread(
        simulationSingleThreaded, 
        generators[i], 
        slower, 
        quadrupole, 
        atoms_per_thread,
        &hists[i]));
  }

  joinAndClearThreads(threads);

  std::ofstream out;
  out.open("out.txt");

  size_t size_vel = hists[0].bins_vel_.number_of_bins_;
  size_t size_pos = hists[0].bins_pos_.number_of_bins_;

  size_t ix = 0;
  for (size_t k = 0; k < size_vel; ++k)
  {
    for (size_t j = 0; j < size_pos; ++j)
    {
      size_t bin_sum = 0;
      for (auto& h : hists)
        bin_sum += h.histogram_[ix];
      
      out << bin_sum << ",";

      ++ix;
    }
    out << std::endl;
  }

  out.close();


  std::ofstream export_fieldshape;
  export_fieldshape.open("fieldshape.txt");

  auto const& bp = hists[0].bins_pos_;
  double z0 = bp.lowest_start_;
  double zstep = bp.bin_width_;
  int nsteps = bp.number_of_bins_;

  for (int i = 0; i < nsteps; ++i)
  {
    double z = z0 + (double(i))*zstep;
    double field = interpolate(slower, z) + interpolate(quadrupole, z);

    double vel = bohr_magneton*field / (planck_reduced * transition_wave_vec);

    std::cout << z << "   " <<field << std::endl;

    export_fieldshape << z << "," << vel << std::endl;
  }

  export_fieldshape.close();

}


int main(int argc, char* argv[])
{

  std::string out_dir = "D:\\Magisterij\\magisterij\\Graphs\\phasespaces\\try";

  uint32_t number_of_particles = 20000;

  simulation(4, number_of_particles);

  std::cout << "ended" << std::endl;
  
  
  return 0;
}

