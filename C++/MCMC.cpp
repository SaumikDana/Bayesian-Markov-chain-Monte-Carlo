#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/Dense>
#include </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/Cholesky>
#include </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/unsupported/Eigen/MatrixFunctions>

using namespace std;

class MCMC {
private:
    // Define class variables
    RateStateModel model;
    double qstart;
    vector<double> qpriors;
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
    MCMC(RateStateModel model, vector<double> data, vector<double> qpriors, double qstart,
         int nsamples = 100, int adapt_interval = 10, bool verbose = true)
        : model(model), data(data), qpriors(qpriors), qstart(qstart), nsamples(nsamples),
          nburn(nsamples / 2), verbose(verbose), adapt_interval(adapt_interval),
          n0(0.01) {}

    Eigen::MatrixXd sample() {
        // Evaluate the model with the original dc value
        Eigen::MatrixXd acc_ = model.evaluate()[1];

        // Perturb the dc value
        model.Dc *= (1 + 1e-6);

        // Evaluate the model with the perturbed dc value
        Eigen::MatrixXd acc_dq_ = model.evaluate()[1];

        // Extract the values and reshape them to a 1D array
        Eigen::MatrixXd acc = acc_.reshape(1, -1);
        Eigen::MatrixXd acc_dq = acc_dq_.reshape(1, -1);

        // Compute the variance of the noise
        std2.push_back((acc - data).array().square().sum() / (acc.cols() - qpriors.size()));

        // Compute the covariance matrix
        Eigen::MatrixXd X = ((acc_dq - acc) / (model.Dc * 1e-6)).transpose();
        Eigen::MatrixXd XTX = X.transpose() * X;
        Vstart = std2.back() * XTX.inverse();
        Eigen::MatrixXd qstart_vect = Eigen::MatrixXd::Constant(1, 1, qstart);
        qstart_limits = Eigen::MatrixXd::Constant(1, 2, qpriors[0], qpriors[1]);
        Eigen::MatrixXd qparams(qstart_vect.rows(), nsamples);
        qparams.col(0) = qstart_vect;
        Eigen::MatrixXd Vold = Vstart;
        Eigen::MatrixXd Vnew = Vstart;
        double SSqprev = SSqcalc(qparams.col(0))(0);
        int iaccept = 0;

        for (int isample = 1; isample < nsamples; isample++) {
            // Sample new parameters from a normal distribution with mean being the last element of qparams
            Eigen::MatrixXd q_new = qparams.col(isample - 1) +
                                    Eigen::MatrixXd::NullaryExpr(qparams.rows(), 1,
                                    [&]() { return normal_distribution<>(0.0, 1.0)(random_engine); }) *
                                    Vold.llt().matrixL();

            // Accept or reject the new sample based on the Metropolis-Hastings acceptance rule
            double SSqnew;
            bool accept;
            tie(accept, SSqnew) = acceptreject(q_new, SSqprev, std2.back());

            // If the new sample is accepted, add it to the list of sampled parameters
            if (accept) {
                qparams.col(isample) = q_new;
                SSqprev = SSqnew;
                iaccept++;
            } else {
                // If the new sample is rejected, add the previous sample to the list of sampled parameters
                qparams.col(isample) = qparams.col(isample - 1);
            }

            // Update the estimate of the standard deviation
            double aval = 0.5 * (n0 + data.size());
            double bval = 0.5 * (n0 * std2.back() + SSqprev);
            std2.push_back(1.0 / gamma_distribution<>(aval, 1.0 / bval)(random_engine));

            // Update the covariance matrix if it is time to adapt it
            if ((isample + 1) % adapt_interval == 0) {
                try {
                    Vnew = 2.38 * 2 / qpriors.size() * qparams.rightCols(adapt_interval).cov();
                    if (qparams.rows() == 1) {
                        Vnew.resize(1, 1);
                    }
                    Vnew = Vnew.llt().matrixL();
                    Vold = Vnew;
                } catch (...) {
                    // Ignore any errors
                }
            }
        }

        // Print acceptance ratio
        cout << "Acceptance ratio: " << static_cast<double>(iaccept) / nsamples << endl;

        // Trim the estimate of the standard deviation to exclude burn-in samples
        std2.erase(std2.begin(), std2.begin() + nburn);

        // Return accepted samples
        return qparams.rightCols(nsamples - nburn);
    }

private:
    mt19937 random_engine{random_device{}()};

    tuple<bool, double> acceptreject(const Eigen::MatrixXd& q_new, double SSqprev, double std2) {
        // Check if the proposal values are within the limits
        bool accept = (q_new.array() > qstart_limits(0, 0)) && (q_new.array() < qstart_limits(0, 1));

        if (accept) {
            // Compute the sum of squares error of the new proposal
            double SSqnew = SSqcalc(q_new)(0);

            // Compute the acceptance probability
            double accept_prob = min(0.5 * (SSqprev - SSqnew) / std2, 0.0);

            // Check if the proposal is accepted based on the acceptance probability and a random number
            accept = accept_prob > log(uniform_real_distribution<>(0.0, 1.0)(random_engine));
        }

        if (accept) {
            // If accepted, return the boolean true and the sum of squares error of the new proposal
            return make_tuple(true, SSqnew);
        } else {
            // If rejected, return the boolean false and the sum of squares error of the previous proposal
            return make_tuple(false, SSqprev);
        }
    }

    Eigen::VectorXd SSqcalc(const Eigen::MatrixXd& q_new) {
        // Update the Dc parameter of the model with the new proposal
        model.Dc = q_new(0, 0);

        // Evaluate the model's performance on the problem
        Eigen::MatrixXd acc = model.evaluate()[1];

        // Compute the sum of squares error between the model's accuracy and the data
        return (acc - data).array().square().colwise().sum();
    }
};