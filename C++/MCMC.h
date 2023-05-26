#ifndef MCMC_H
#define MCMC_H

#include <vector>
#include <random>
#include </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/Dense>
#include "RateStateModel.h"

using namespace std;

class MCMC {
private:
    RateStateModel model;
    double qstart;
    uniform_real_distribution<double> qpriors;
    int nsamples;
    int nburn;
    bool verbose;
    int adapt_interval;
    vector<double> data;
    double n0;
    vector<double> std2;
    Eigen::MatrixXd Vstart;
    Eigen::MatrixXd qstart_limits;

public:
    MCMC(RateStateModel model, vector<double> data, uniform_real_distribution<double> qpriors, double qstart,
         int nsamples = 100, int adapt_interval = 10, bool verbose = true);

    Eigen::MatrixXd sample();

private:
    mt19937 random_engine;

    tuple<bool, double> acceptreject(const Eigen::MatrixXd& q_new, double SSqprev, double std2);

    Eigen::VectorXd SSqcalc(const Eigen::MatrixXd& q_new);
};

#endif /* MCMC_H */
