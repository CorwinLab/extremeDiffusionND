#include <assert.h>
#include <boost/math/distributions.hpp>
#include <boost/multiprecision/float128.hpp>
#include <boost/random.hpp>
#include <boost/random/beta_distribution.hpp>
#include <boost/random/binomial_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <math.h>
#include <random>
#include <utility>
#include <vector>
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11; 

typedef boost::multiprecision::float128 RealType;


#ifndef DIFFUSIONPDF_HPP_
#define DIFFUSIONPDF_HPP_

class RandDistribution{
  protected:
    std::vector<double> alpha;
    gsl_rng *gen = gsl_rng_alloc(gsl_rng_mt19937);
    size_t K; 
    std::vector<double> theta;

  public: 
    RandDistribution(const std::vector<double> _alpha);
    ~RandDistribution(){};

    std::vector<double> getAlpha() { return alpha; };
    void setAlpha(std::vector<double> _alpha){ alpha = _alpha; };
    std::vector<RealType> getRandomNumbers();
};

// Base Diffusion class
class DiffusionND : public RandDistribution {
protected:
  std::vector<std::vector<RealType> > PDF;
  unsigned long int t;
  unsigned int L;

  std::random_device rd;
  boost::random::mt19937_64 gen;
  std::vector<RealType> biases;

public:
  DiffusionND(const std::vector<double> _alpha, unsigned int _L);
  ~DiffusionND(){};

  // Should consider doing lexical_cast<std::string> to convert to float128 to string to view in Python. 
  // Could also help to save the occupancy into the file
  std::vector<std::vector<RealType> > getPDF() { return PDF; };
  void setPDF(std::vector<std::vector<RealType> > _PDF){ PDF = _PDF; };

  unsigned long int getTime(){ return t; };
  void setTime(unsigned long int _t){ t = _t; }; 
  unsigned int getL() { return L; }
  std::vector<std::vector<RealType> > integratedProbability(std::vector<std::vector<double> >);
  std::vector<std::vector<double> > logIntegratedProbability(std::vector<std::vector<double> >);

  void saveOccupancy(std::string);
  void loadOccupancy(std::string);

  void iterateTimestep();
};

#endif /* DIFFUSIONPDF_HPP_ */
