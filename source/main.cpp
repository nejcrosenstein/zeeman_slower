#include <iostream>
#include <random>
#include <fstream>
#include <array>
#include <thread>
#include <iomanip>
#include <optional>

#include "mathtools.hpp"
#include "physics.hpp"
#include "coils.hpp"
#include "beam.hpp"

#include "../externals/randgen/xoroshiro128plus.hpp"

struct SimulationParam
{
  struct LaserBeamParam
  {
    double beam_spread_angle_rad = 0.004;
    double beam_spread_angle_tangent = beam_spread_angle_rad;

    double waist_at_oven_exit = 0.004;
    double oven_exit_pos_z = 0.63 - 0.83 - 0.15;

    double focus_point_z = -1.4;
    double laser_power_watt = 0.01;

  } laser_beam_param_;
  
  struct HistogramParam
  {
    double vel_lowest = -50.0;
    double vel_binwidth = 1.0;
    int vel_nbins = 500;
      
    double pos_lowest = -0.2; 
    double pos_binwidth = 0.001;
    int pos_nbins = 1000;

  } histogram_param_;

  
  // Oven temperature
  double oven_temperature_kelvin = 353.0;

  // Atomic beam properties
  double beam_spread_angle_rad = 0.004;
  
  // Frequency detuning (*not* angular frequency)
  double laser_detuning_hz = 0.0; // TODO: use Mhz
  double laser_power_watt = 0.005;

  double focus_point_z = -1.4; // in meters

  // Simulation initial conditions
  double initial_atomic_beam_radius_m = 0.003;
  double simulation_start_z_m = -0.2;

  // Time step in seconds
  double time_step_s = 0.1*excited_state_lifetime;

  // Number of particles
  size_t number_of_particles = 10000;

  // Number of threads
  size_t number_of_threads = 4;
};

// **************************************************************
//
//          Histogram 
//
// **************************************************************

struct Histogram
{
  using Param_t = SimulationParam::HistogramParam;

  Histogram() {}

  Histogram(Param_t const& p) : 
    bins_vel_({ p.vel_lowest, p.vel_binwidth, p.vel_nbins }),
    bins_pos_({ p.pos_lowest, p.pos_binwidth, p.pos_nbins }),
    histogram_(std::vector<int>(p.vel_nbins*p.pos_nbins, 0))
  {}

  void addSample(double sample_pos, double sample_vel)
  {
    auto pos_and_vel_idx = std::make_pair(
      bins_pos_.binIndex(sample_pos),
      bins_vel_.binIndex(sample_vel));

    if (!current_pos_and_vel_bin_ ||
      *current_pos_and_vel_bin_ != pos_and_vel_idx)
    {
      current_pos_and_vel_bin_ = pos_and_vel_idx;

      int pos_idx = pos_and_vel_idx.first;
      int vel_idx = pos_and_vel_idx.second;

      if (pos_idx >= 0 && pos_idx < bins_pos_.number_of_bins_ &&
        vel_idx >= 0 && vel_idx < bins_vel_.number_of_bins_)
      {
        histogram_[vel_idx*bins_pos_.number_of_bins_ + pos_idx] += 1;
      }
    }
  }

  struct Bins
  {
    double lowest_start_;
    double bin_width_;
    int number_of_bins_;

    // can be out of bounds
    int binIndex(double value)
    {
      return int((value - lowest_start_) / bin_width_);
    }
  };

  Bins bins_vel_;
  Bins bins_pos_;
  std::optional<std::pair<int, int>> current_pos_and_vel_bin_;
  std::optional<int> current_vel_bin_;
  std::vector<int> histogram_;
};

// **************************************************************
//
//          Laser beam properties
//
// **************************************************************

struct LaserBeam
{
  using Param_t = SimulationParam::LaserBeamParam;
  
  Param_t param_;

  LaserBeam(Param_t const& param) : param_(param){}
  //
  // Beam waist at current position (let's ignore waist of Gaussian beam for now)
  // 
  __forceinline __m256d __vectorcall beamWaist(__m256d const& pos_on_axis) const
  {
    __m256d dist = subtract(pos_on_axis, broadcast(param_.oven_exit_pos_z));

    return multiply_add(dist,
      broadcast(param_.beam_spread_angle_tangent),
      broadcast(param_.waist_at_oven_exit));
  }

  // 
  // Beam intensity at current position
  //
  __forceinline __m256d __vectorcall beamIntensity(
    QuadrupleAVX_3D const& pos) const
  {
    __m256d rad_sq =
      multiply_add(pos[X], pos[X],
          multiply(pos[Y], pos[Y]));

    __m256d waist = beamWaist(pos[Z]);

    __m256d waist_sq_inv = reciprocal(multiply(waist, waist));

    __m256d peak_intensity =
      multiply(waist_sq_inv, broadcast(2.0*param_.laser_power_watt / pi));

    __m256d twice_ratio_squared =
      multiply(
        broadcast(2.0),
        multiply(rad_sq, waist_sq_inv));

    QuadrupleScalar trsq;
    store(twice_ratio_squared, trsq);

    QuadrupleScalar exped;
    for_every_quadruple_member(i)
    {
      exped[i] = std::exp(-trsq[i]);
    }

    return multiply(load(exped), peak_intensity);
  }

  //
  // Computes beam direction. It is assumed that the
  // beam converges towards a point on the slower axis. 
  // The function returns normalized direction.
  //
  __forceinline void __vectorcall beamDirection(
    QuadrupleAVX_3D const& pos,
    QuadrupleAVX_3D& dir_normalized) const
  {
    // focus point
    QuadrupleAVX_3D foc;
    foc[X] = _mm256_setzero_pd();
    foc[Y] = _mm256_setzero_pd();
    foc[Z] = broadcast(param_.focus_point_z);

    QuadrupleAVX_3D dirs;
    for_every_direction(d)
      dirs[d] = subtract(foc[d], pos[d]);

    __m256d norm_squared = dotProduct(dirs, dirs);

    __m256d norm_inv = reciprocal(squareRoot(norm_squared));

    for_every_direction(d)
    {
      dir_normalized[d] = multiply(dirs[d], norm_inv);
    }
  }
};

// **************************************************************
//
//          Initial velocities and positions of particles
//
// **************************************************************
//
//
// InitialStates struct provides us with initial positions and velocities of 
// particles. The positions are randomly drawn from uniform distribution of points 
// within a circle. The velocities are drawn from the distribution of 
// velocities inside a collimated beam (TODO: reference from Foot / Metcalf)
// 
struct InitialStates
{
  typedef double vec3d[NDir];
  struct Particle
  {
    vec3d pos_;
    vec3d vel_;
  };
  
  template<class RandGen_t>
  
  InitialStates(
    BeamVelocityDistribution const& init_velocity, SimulationParam const& param, RandGen_t& rand_gen)
  {
    using Uniform = std::uniform_real_distribution<double>;
    
    Uniform phi(0.0, two_pi);
    Uniform cos_theta_initvel(cos(param.beam_spread_angle_rad), 1.0);
    Uniform zero_to_one(0.0, 1.0);

    size_t num = param.number_of_particles;

    initial_states_.reserve(num);

    for (size_t i = 0; i < num; ++i)
    {
      double vel_magnitude = init_velocity(rand_gen);

      auto& s = initial_states_.emplace_back();

      computeCartesian(phi(rand_gen), cos_theta_initvel(rand_gen), s.vel_);
      
      for_every_direction(dir)
        s.vel_[dir] *= vel_magnitude;

      double angle = phi(rand_gen);
      double two_rand_sum = zero_to_one(rand_gen) + zero_to_one(rand_gen);
      double r = two_rand_sum > 1 ? 2.0 - two_rand_sum : two_rand_sum;
      double rad = r * param.initial_atomic_beam_radius_m;

      s.pos_[X] = rad * cos(angle);
      s.pos_[Y] = rad * sin(angle);
      s.pos_[Z] = param.simulation_start_z_m; 
    }

    // Sort according to Z component of velocity
    // This sorting is not cruiciable and it does not affect the
    // outcome of the simulation or performance. It just makes 
    // some debug / profiling operations easier.
    std::sort(
      initial_states_.begin(), 
      initial_states_.end(),
      [this](auto const& particle, auto const& other) 
      { return particle.vel_[Z] > other.vel_[Z]; });
  }



  std::optional<Particle> pop_state()
  {
    if (initial_states_.size() == 0)
    {
      return std::nullopt;
    }
    
    Particle last = initial_states_.back();
    initial_states_.pop_back();
    return last;
  }

  // works OK for 0 < theta < pi
  void computeCartesian(double phi, double cos_theta, vec3d& dst) const
  {
    using namespace std;
    dst[X] = cos(phi)*sqrt(1.0 - sq(cos_theta));
    dst[Y] = sin(phi)*sqrt(1.0 - sq(cos_theta));
    dst[Z] = cos_theta;
  }

  std::vector<Particle> initial_states_;
};

//
// ParticleQuadruple contains positions and velocities of four particles. 
// The component are packed together so that we can efficiently load 
// them into AVX registers.
//
struct ParticleQuadruple
{
  QuadrupleScalar3D pos;
  QuadrupleScalar3D vel;
};


// **************************************************************
//
//          Monte Carlo simulation
//
// **************************************************************

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
  ImportedField const& quadrupole_coil, 
  LaserBeam const& laser,
  SimulationParam const& param)
{ 
  // Current position and velocity

  QuadrupleAVX_3D curr_pos;
  QuadrupleAVX_3D curr_vel;
  for_every_direction(dir)
  {
    curr_pos[dir] = load(atoms.pos[dir]);
    curr_vel[dir] = load(atoms.vel[dir]);
  }
  __m256d intensity = laser.beamIntensity(curr_pos);

  // TODO: combine both fields before simulation starts
  __m256d field_slower = interpolate(slower, curr_pos);
  __m256d field_quad = interpolate(quadrupole_coil, curr_pos);
  __m256d field_tesla = add(field_slower, field_quad);
 
  __m256d light_dir[NDir];
  laser.beamDirection(curr_pos, light_dir);

  __m256d vel_along_light_dir = dotProduct(curr_vel, light_dir);
  
  __m256d laser_detuning = broadcast(param.laser_detuning_hz);
  __m256d scatter_rates = scatteringRate(vel_along_light_dir, intensity, laser_detuning, field_tesla);

  __m256d time_step = broadcast(param.time_step_s);

  for_every_direction(dir)
  {
    __m256d new_pos = multiply_add(curr_vel[dir], time_step, curr_pos[dir]);
    store(new_pos, atoms.pos[dir]);
  }

  // scatter event probability
  __m256d probs = _mm256_mul_pd(scatter_rates, time_step);

  // number drawn from uniform distribution between 0 and 1
  __m256d uniform = rand_gen.random_simd();

  __m256d comparison = _mm256_cmp_pd(uniform, probs, _CMP_LE_OQ);

  int mmask = _mm256_movemask_pd(comparison);

  if(mmask)
  { 
    QuadrupleScalar cos_phi = { 0, 0, 0, 0 };
    QuadrupleScalar sin_phi = { 0, 0, 0, 0 };
    
    __m256d planar = rand_gen.random_simd(0.0, two_pi);
    __m256d cos_polar = rand_gen.random_simd(-1.0, 1.0);
    QuadrupleScalar phi;
    store(planar, phi);

    for_every_quadruple_member(i)
    {
      bool scatter_occurs = (mmask & (1 << i));
      if (scatter_occurs)
      {
        cos_phi[i] = cos(phi[i]);
        sin_phi[i] = sin(phi[i]);
      }
    }

    __m256d cos_planar = load(cos_phi);
    __m256d sin_planar = load(sin_phi);

    __m256d sin_polar_sq = _mm256_fnmadd_pd(cos_polar, cos_polar, broadcast(1.0));
    __m256d sin_polar = squareRoot(sin_polar_sq);

    QuadrupleAVX_3D change_dir;
    change_dir[X] = _mm256_fnmadd_pd(cos_planar, sin_polar, light_dir[X]);
    change_dir[Y] = _mm256_fnmadd_pd(sin_planar, sin_polar, light_dir[Y]);
    change_dir[Z] = subtract(light_dir[Z], cos_polar);

    __m256d vel_recoil =
      _mm256_and_pd(
        broadcast(recoil_velocity),
        comparison);

    for_every_direction(dir)
    {
      __m256d new_vel = multiply_add(change_dir[dir], vel_recoil, curr_vel[dir]);
      store(new_vel, atoms.vel[dir]);
    }
  }
}

// just pass everything by copy
void simulationSingleThreaded(
  XoroshiroSIMD rand_gen, 
  ZeemanSlower slower, 
  ImportedField quadrupole,
  LaserBeam const& laser,
  SimulationParam param,
  Histogram* dst)
{
  BeamVelocityDistribution init_velocity_dist(param.oven_temperature_kelvin);

  InitialStates init_states(init_velocity_dist, param, rand_gen);

  Histogram hist(param.histogram_param_);

  std::array<Histogram, 4> hists = { hist, hist, hist, hist };

  // TODO: also define it somewhere outside
  auto stop_condition = [](ParticleQuadruple const& atoms, int idx) 
  {
    return (atoms.pos[Z][idx] > 0.8) || 
           (atoms.vel[Z][idx] < 0.0);
  };

  ParticleQuadruple atoms;

  // Pop initial states and convert them
  // into packed particle quadruple representation.
  for_every_quadruple_member(i)
  {
    auto s = init_states.pop_state();
    for_every_direction(dir)
    {
      atoms.pos[dir][i] = s->pos_[dir];
      atoms.vel[dir][i] = s->vel_[dir];
    }
  }

  size_t total = param.number_of_particles;
  size_t num_remaining = total;

  float progress = 0.0f;
  float progress_step = 0.1f;

  while(num_remaining != 0)
  {  
    for_every_quadruple_member(i)
    {
      hists[i].addSample(atoms.pos[Z][i], atoms.vel[Z][i]);
      
      bool finished = stop_condition(atoms, i);

      if (finished)
      {
        --num_remaining; 
        // TODO: number of particles is determined by number of supplied initial states.
        // Use the latter number to determine number of particles.

        auto const& s = init_states.pop_state();

        if (s)
        {
          for_every_direction(dir)
          {
            atoms.pos[dir][i] = s->pos_[dir];
            atoms.vel[dir][i] = s->vel_[dir];
          }
        }

#if 1
        float ratio = 1.0f - float(num_remaining) / float(total);
        if (ratio > progress)
        {
          progress += progress_step;
       
          std::cout
            << std::setprecision(2)
            << "Progress:" << ratio << std::endl;
        }
#endif
      }
    }

    takeOneStep(atoms, rand_gen, slower, quadrupole, laser, param);
  }

  for (auto const& h : hists)
    for (int i = 0; i < h.histogram_.size(); ++i)
      hist.histogram_[i] += h.histogram_[i];

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
  LaserBeam laser(param.laser_beam_param_);

  // Parameters for single simulation instance
  // ("st" as in "single thread")
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
        laser,
        param_st,
        &hists[i]));
  }

  // Wait until simulation is finished in all threads
  joinAndClearThreads(threads);

  std::ofstream out;
  out.open("histogram.txt");

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

  std::ofstream vel_axis;
  vel_axis.open("histogram_axis_vel.txt");

  auto const& hv = hists[0].bins_vel_;
  for (size_t i = 0; i < size_vel; ++i)
  {
    vel_axis << hv.lowest_start_ + double(i)*hv.bin_width_ << " ";
  }
  vel_axis.close();
  
  std::ofstream pos_axis;
  pos_axis.open("histogram_axis_pos.txt");

  auto const& hp = hists[0].bins_pos_;
  for (size_t i = 0; i < size_pos; ++i)
  {
    pos_axis << hp.lowest_start_ + double(i)*hp.bin_width_ << " ";
  }
  pos_axis.close();

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

