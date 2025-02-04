#include <math.h>
#include <boost/multiprecision/float128.hpp>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include "diffusionND.hpp"
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"

#define _USE_MATH_DEFINES

using RealType = boost::multiprecision::float128;
static_assert(sizeof(RealType) == 16, "Bad size");

RandDistribution::RandDistribution(const std::vector<double> _alpha)
{
	alpha = _alpha;
	gsl_rng_set(gen, (unsigned)time(NULL));
	K = 4;
	theta.resize(4);
}

// Need to define get alpha method
std::vector<RealType> RandDistribution::getRandomNumbers()
{
	if (isinf(alpha[0]))
	{
		return {(RealType)0.25, (RealType)0.25, (RealType)0.25, (RealType)0.25};
	}
	else
	{
		gsl_ran_dirichlet(gen, K, &alpha[0], &theta[0]);
		/* Cast double precision to quad precision by dividing by sum of double precision rand numbers*/
		std::vector<RealType> biases(theta.begin(), theta.end());

		RealType sum = 0;
		for (unsigned int i = 0; i < biases.size(); i++)
		{
			sum += biases[i];
		}

		for (unsigned int i = 0; i < biases.size(); i++)
		{
			biases[i] /= sum;
		}
		return biases;
	}
}

DiffusionND::DiffusionND(const std::vector<double> _alpha, unsigned int _L) : RandDistribution(_alpha)
{
	L = _L;

	gen.seed(rd());

	std::vector<double> biases;

	PDF.resize(2 * L + 1, std::vector<RealType>(2 * L + 1));
	PDF[L][L] = 1;

	t = 0;
}

void DiffusionND::iterateTimestep()
{

	for (unsigned long int i = 1; i < 2 * L; i++)
	{ // i is the columns
		for (unsigned long int j = 1; j < 2 * L; j++)
		{ // j is the row
			if ((i + j + t) % 2 == 1)
			{
				continue;
			}

			RealType currentPos = PDF.at(i).at(j);
			if (currentPos == 0)
			{
				continue;
			}

			biases = getRandomNumbers();

			PDF.at(i + 1).at(j) += currentPos * (RealType)biases[0];
			PDF.at(i - 1).at(j) += currentPos * (RealType)biases[1];
			PDF.at(i).at(j + 1) += currentPos * (RealType)biases[2];
			PDF.at(i).at(j - 1) += currentPos * (RealType)biases[3];

			PDF.at(i).at(j) = 0;
		}
	}
	/* Ensure we aren't losing/gaining probability */
	//   if ((cdf_new_sum + absorbedProb) < ((RealType)1.-(RealType)pow(10, -25)) || (cdf_new_sum + absorbedProb) > ((RealType)1.+(RealType)pow(10, -25))){
	// 	std::cout << "CDF total: " << cdf_new_sum + absorbedProb << std::endl;
	// 	throw std::runtime_error("Total probability not within tolerance of 10^-25");
	//   }

	t += 1;
}

std::vector<std::vector<RealType>> DiffusionND::integratedProbability(std::vector<std::vector<double> > radii)
{
	std::vector<std::vector<RealType> > probabilities;
	probabilities.resize(radii.size(), std::vector<RealType>(radii.at(0).size()));
	for (unsigned long int i = 0; i < 2 * L + 1; i++)
	{ // i is the columns
		for (unsigned long int j = 0; j < 2 * L + 1; j++)
		{ // j is the row

			if (PDF.at(i).at(j) == 0)
			{
				continue;
			}
			int xval = i - L; 
			int yval = j - L;
			double distanceToOrigin = sqrt(pow(xval, 2) + pow(yval, 2));
			
			for (unsigned long int l = 0; l < radii.size(); l++)
			{
				for (unsigned long int k = 0; k < radii.at(l).size(); k++)
				{
					double currentRadii = radii.at(l).at(k);

					if (distanceToOrigin > currentRadii)
					{
						probabilities.at(l).at(k) += PDF.at(i).at(j);
					}
				}
			}
		}
	}

	return probabilities;
}

std::vector<std::vector<double> > DiffusionND::logIntegratedProbability(std::vector<std::vector<double> > radii){
	std::vector<std::vector<RealType> > probabilities = integratedProbability(radii);
	std::vector<std::vector<double> > logProbabilities(probabilities.size(), std::vector<double>(probabilities.at(0).size()));
	
	for (unsigned long int i = 0; i < radii.size(); i++){
		for (unsigned long int j = 0; j < radii.at(i).size(); j ++){
			logProbabilities.at(i).at(j) = double(log(probabilities.at(i).at(j)));
		}
	}
	return logProbabilities;
}
