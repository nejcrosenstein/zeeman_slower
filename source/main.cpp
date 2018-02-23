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

static constexpr double oven_exit_radius_m = 0.003;

struct SimulationParam
{
  // Oven temperatur
  double oven_temperature_kelvin = 353.0;
  
  // Frequency (*not* angular frequency)
  double laser_detuning_hz = 0.0; // TODO: use Mhz

  // Time step in seconds
  double time_step_s = 0.1*excited_state_lifetime;

  // Number of particles
  size_t number_of_particles = 2000;

  // Number of threads
  size_t number_of_threads = 4;
};


enum Directions
{
  X = 0,
  Y = 1,
  Z = 2,
  NDir = 3
};

//
// InitialStates struct provides us with initial positions and velocities of 
// particles. The positions are randomly drawn from uniform distribution of points 
// within a circle. The velocities are drawn from the distribution of 
// velocities inside a collimated beam (TODO: reference from Foot / Metcalf)
//
// After positions and velocities of atoms are computed, the samples are sorted 
// according to the Z component of velocity. This is done purely for performance 
// reasons: in simulation, the paths of four atoms are simulated simultaneously. 
// Simulation stops when *all* four atoms fulfill the so-called "stop condition". 
// Let's consider the following example:
//
// We simultaneously simulate four atoms - one with medium initial velocity and three 
// with very high velocity. The stopping condition is fulfilled when the atoms reach 
// the end of the slower.  One atom is slow enough so that it can be stopped by a slower. 
// Lots of photons (~tens of thousands) will scatter on this atom and its velocity will 
// gradually decrease. This atom will therefore reach the end position after a relatively
// large number of time steps. The other three atoms reach the final position very quickly
// because they are too fast to be stopper by a slower. We don't require a lot of time steps
// to simulate the path of these three atoms, but the simulation nevertheless runs on and on
// until the stopping condition is also fulfilled for the fourth atom. 
// 
// In the above example, results are correct, but the simulation is inefficient. We can
// improve performance if we simulate together the atoms with similar starting velocities. 
// We achieve this by sorting the initial states.
//
struct InitialStates
{
  using Vecd = Vec3D<double>;
  
  template<class RandGen_t>
  
  InitialStates(
    BeamVelocityDistribution const& init_velocity, int number, RandGen_t& rand_gen)
  {
    using Uniform = std::uniform_real_distribution<double>;
    
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

//
// ParticleQuadruple contains positions and velocities of four particles. 
// The component are packed together so that we can efficiently load 
// them into AVX registers.
//
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

//
// Computes dot products of 4 pairs of vectors in packed representation.
//
__forceinline __m256d __vectorcall dotProduct(
  const __m256d(&a)[NDir],
  const __m256d(&b)[NDir])
{
  // Some micro optimization is possible (fmadd), but
  // the effect would probably not be noticeable
  __m256d prod0 = _mm256_mul_pd(a[X], b[X]);
  __m256d prod1 = _mm256_mul_pd(a[Y], b[Y]);
  __m256d prod2 = _mm256_mul_pd(a[Z], b[Z]);

  return _mm256_add_pd(prod0, _mm256_add_pd(prod1, prod2));
}

//
// Computes light direction. It is assumed that the laser
// beam is focused into a point on the slower axis. 
// The result is normalized.
//
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



//
// Move one step forward in time. In this function, the
// atom position is updated: 
//
//    pos += velocity * time_step
//
// It is also determined if light scattering occurs
// or not. We neglect the time that passes between absorbption
// and emission. Therefore, if photon scattering condition is
// fulfilled, we also update the velocity:
//
//    vel += recoil_velocity (in light direction)  // absorption
//    vel -= recoil_velocity (in random direction) // spont. emission
//
template<class RandomGen_t>
static void takeOneStep(
  ParticleQuadruple& atoms,
  RandomGen_t& rand_gen,
  ZeemanSlower const& slower,
  ImportedField const& quadrupole, 
  SimulationParam const& param)
{ 
  // Current position and velocity
  __m256d curr_pos[NDir];
  atoms.getPositions(curr_pos);
  
  __m256d curr_vel[NDir];
  atoms.getVelocities(curr_vel);

  __m256d intensity = lightIntensity(curr_pos);

  // TODO: combine both fields before simulation starts
  __m256d field_slower = interpolate(slower, curr_pos);
  __m256d field_quad = interpolate(quadrupole, curr_pos);
  __m256d field_tesla = _mm256_add_pd(field_slower, field_quad);
 
  __m256d light_dir[NDir];
  computeLightDirection(curr_pos, light_dir);

  __m256d vel_along_light_dir = dotProduct(curr_vel, light_dir);
  
  __m256d laser_detuning = _mm256_set1_pd(param.laser_detuning_hz);
  __m256d scatter_rates = scatteringRate(vel_along_light_dir, intensity, laser_detuning, field_tesla);

  __m256d time_step = _mm256_set1_pd(param.time_step_s);

  __m256d new_pos[NDir];
  new_pos[X] = _mm256_fmadd_pd(curr_vel[X], time_step, curr_pos[X]);
  new_pos[Y] = _mm256_fmadd_pd(curr_vel[Y], time_step, curr_pos[Y]);
  new_pos[Z] = _mm256_fmadd_pd(curr_vel[Z], time_step, curr_pos[Z]);

  atoms.setPositions(new_pos);

  // scatter event probability
  __m256d probs = _mm256_mul_pd(scatter_rates, time_step);

  // number drawn from uniform distribution between 0 and 1
  __m256d uniform = rand_gen.random_simd();

  __m256d comparison = _mm256_cmp_pd(uniform, probs, _CMP_LE_OQ);

  int mmask = _mm256_movemask_pd(comparison);

  if(mmask)
  { 
    double cos_phi[4] = { 0, 0, 0, 0 };
    double sin_phi[4] = { 0, 0, 0, 0 };
    
    double phi[4];
    __m256d planar = rand_gen.random_simd(0.0, two_pi);
    __m256d cos_polar = rand_gen.random_simd(-1.0, 1.0);
    _mm256_store_pd(phi, planar);

    for (int i = 0; i < 4; ++i)
    {
      bool scatter_occurs = (mmask & (1 << i));
      if (scatter_occurs)
      {
        cos_phi[i] = cos(phi[i]);
        sin_phi[i] = sin(phi[i]);
      }
    }

    __m256d cos_planar = _mm256_load_pd(cos_phi);
    __m256d sin_planar = _mm256_load_pd(sin_phi);

    __m256d sin_polar_sq = _mm256_fnmadd_pd(cos_polar, cos_polar, _mm256_set1_pd(1.0));
    __m256d sin_polar = _mm256_sqrt_pd(sin_polar_sq);

    __m256d change_dir[NDir];
    change_dir[X] = _mm256_fnmadd_pd(cos_planar, sin_polar, light_dir[X]);
    change_dir[Y] = _mm256_fnmadd_pd(sin_planar, sin_polar, light_dir[Y]);
    change_dir[Z] = _mm256_sub_pd(light_dir[Z], cos_polar);

    __m256d vel_recoil =
      _mm256_and_pd(
        _mm256_set1_pd(recoil_velocity),
        comparison);

    __m256d new_vel[NDir];
    new_vel[X] = _mm256_fmadd_pd(change_dir[X], vel_recoil, curr_vel[X]);
    new_vel[Y] = _mm256_fmadd_pd(change_dir[Y], vel_recoil, curr_vel[Y]); 
    new_vel[Z] = _mm256_fmadd_pd(change_dir[Z], vel_recoil, curr_vel[Z]);

    atoms.setVelocities(new_vel);
  }
}

template<class RanGen_t, class StopCondition>
void simulateQuadruplePath(RanGen_t& rand_gen,
  ZeemanSlower const& slower,
  ImportedField const& quadrupole,
  SimulationParam const& param,
  InitialStates& init_states,
  std::array<Histogram, 4>& hists,
  StopCondition const& stop_condition)
{
  ParticleQuadruple atoms;

  // Pop initial states and convert them
  // into packed particle quadruple representation.
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
  
  for(;;)
  {
    int at_end = 0;
    for (int i = 0; i < 4; ++i)
    {
      hists[i].addSample(atoms.pos_z[i], atoms.vel_z[i]);
      at_end += (int)stop_condition(atoms, i);
    }
    if (at_end == 4) break;

    takeOneStep(atoms, rand_gen, slower, quadrupole, param);
  }
}

// just pass everything by copy
void simulationSingleThreaded(
  XoroshiroSIMD rand_gen, 
  ZeemanSlower slower, 
  ImportedField quadrupole, 
  SimulationParam param,
  Histogram* dst)
{
  BeamVelocityDistribution init_velocity_dist(param.oven_temperature_kelvin);

  InitialStates init_states(init_velocity_dist, param.number_of_particles, rand_gen);

  Histogram hist(-50.5, 1.0, 499.5, -0.2, 0.001, 1000);

  std::array<Histogram, 4> hists = { hist, hist, hist, hist };

  auto stop_condition = [](ParticleQuadruple const& atoms, int idx) 
  {
    return (atoms.pos_z[idx] > 0.8) || (atoms.vel_z[idx] < 0);
  };

  for (int j = 0; j < param.number_of_particles/4; ++j)
  {
    simulateQuadruplePath(
      rand_gen, 
      slower, quadrupole, 
      param, init_states, 
      hists, stop_condition);

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

void simulation(SimulationParam const& param)
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
  // Generation of random number in different threads 
  //
  // Below we create one random generator for every parralel simulation
  // instance. First we make N copies of the generator we created above 
  // and then we use jump instructions again. 
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
  for(size_t idx = 0; idx < param.number_of_threads; ++idx)
  {
    generators.emplace_back(random_gen).jump(4*idx, 1);
  }

  // TODO: create outside this function
  ZeemanSlower slower(0, 0, -0.3, 1.0, 2000);
  ImportedField quadrupole(0.0, "ideal.txt");
  
  // Parameters for single simulation instance
  // ("st" ----> "single thread")
  SimulationParam param_st = param;
  param_st.number_of_particles /= param.number_of_threads;
  param_st.number_of_threads = 1;

  // TODO: create histograms here
  std::vector<Histogram> hists(param.number_of_threads);
  std::vector<std::thread> threads;
  for (size_t i = 0; i < hists.size(); ++i)
  {
    threads.emplace_back(
      std::thread(
        simulationSingleThreaded, 
        generators[i], 
        slower, 
        quadrupole, 
        param_st,
        &hists[i]));
  }

  // Wait until simulation is finished in all threads
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
      for (auto const& h : hists)
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

  SimulationParam param;

  simulation(param);
  
  return 0;
}

