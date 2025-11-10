#include <math.h>
#include <boost/multiprecision/float128.hpp>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include "diffusionND.hpp"
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

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

std::vector<RealType> RandDistribution::getRandomNumbers()
{
	if (!isinf(alpha[0]))
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
	else
	{
		return {(RealType)0.25, (RealType)0.25, (RealType)0.25, (RealType)0.25};
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
	// Configure to only iterate over range which is nonzero
	unsigned long int startIdx;
	unsigned long int endIdx;
	if (t < L){
		startIdx = L - t - 1; 
		endIdx = L + t + 1;
	}
	else{
		startIdx = 1;
		endIdx = 2 * L;
	}
	
	for (unsigned long int i = startIdx; i < endIdx; i++)
	{ // i is the columns
		for (unsigned long int j = startIdx; j < endIdx; j++)
		{ // j is the row
			if (((i + j + t) % 2 == 1) && (PDF.at(i).at(j) != 0)){
				biases = getRandomNumbers();
				RealType currentPos = PDF.at(i).at(j);
				PDF.at(i + 1).at(j) += currentPos * (RealType)biases[0];
				PDF.at(i - 1).at(j) += currentPos * (RealType)biases[1];
				PDF.at(i).at(j + 1) += currentPos * (RealType)biases[2];
				PDF.at(i).at(j - 1) += currentPos * (RealType)biases[3];

				PDF.at(i).at(j) = 0;
			}
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
//	for (unsigned long int i = 0; i < 2 * L + 1; i++)
//	{ // i is the columns
//		for (unsigned long int j = 0; j < 2 * L + 1; j++)
//		{ // j is the row
    // shrink wrapping integrated prob
	unsigned long int startIdx;
	unsigned long int endIdx;
	if (t < L){
		startIdx = L - t - 1;
		endIdx = L + t + 1;
	}
	else{
		startIdx = 1;
		endIdx = 2 * L;
	}

	for (unsigned long int i = startIdx; i < endIdx; i++)
	{ // i is the columns
		for (unsigned long int j = startIdx; j < endIdx; j++)
            {
			if (PDF.at(i).at(j) != 0)
			{
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

void DiffusionND::saveOccupancy(std::string fileName){
	std::fstream myFile;
//	RealType runningSum = 0;
	myFile.open(fileName, std::ios::out|std::ios::binary);
	for (unsigned long int i=0; i < 2 * L+1; i++){
		for (unsigned long int j=0; j<2*L+1; j++){
			myFile.write(reinterpret_cast<char*>(&PDF[i][j]), sizeof(RealType));
//			runningSum += PDF[i][j];
		}
	}
//	myFile.write(reinterpret_cast<char*>(&runningSum),sizeof(RealType));
	myFile.close();
}

void DiffusionND::loadOccupancy(std::string fileName){
    // Check whether file exists before we try to load it
	std::ifstream file(fileName, std::ios::binary);
	if (file.good()) {
        std::cout << "File exists" << std::endl;
    } else {
        std::cout << "File doesn't exist" << std::endl;
        throw std::runtime_error("File not found");
    }
	for (unsigned long int i = 0; i < 2*L+1; i++) {
        for (unsigned long int j = 0; j < 2*L+1; j++) {
            file.read(reinterpret_cast<char*>(&PDF[i][j]), sizeof(RealType));
//            occsum += PDF[i][j];
        }
    }
    // Verify that goodbit is true        throw std::exception("File not found");

    if (file.good()) {
        std::cout << "File read correctly and goodbit is True" << std::endl;
    } else {
        std::cout << "Goodbit is FALSE" << std::endl;
        throw std::runtime_error("File too short");
    }

//    file.read(reinterpret_cast<char*>(&loadedSum), sizeof(RealType));
	file.close();
}
