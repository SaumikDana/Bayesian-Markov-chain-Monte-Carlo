#ifndef MCMC_H
#define MCMC_H

#include <vector>
#include <Eigen/Dense>

class MCMC {
private:
    Model model;
    double qstart;
    std::vector<double> qpriors;
    int nsamples;
    int nburn;
    bool verbose;
    int adapt_interval;
    std::vector<double> data;
    double n0;
    std::vector<double> std2;
    Eigen::MatrixXd Vstart;
    Eigen::MatrixXd qstart_limits;

public:
    MCMC(Model model, std::vector<double> data, std::vector<double> qpriors, double qstart,
         int nsamples = 100, int adapt_interval = 10, bool verbose = true);

    Eigen::MatrixXd sample();

private:
    std::mt19937 random_engine;

    std::tuple<bool, double> acceptreject(const Eigen::MatrixXd& q_new, double SSqprev, double std2);

    Eigen::VectorXd SSqcalc(const Eigen::MatrixXd& q_new);
};

#endif /* MCMC_H */
