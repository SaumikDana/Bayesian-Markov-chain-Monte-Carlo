#include </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/Dense>
#include </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/Cholesky>
#include </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/unsupported/Eigen/MatrixFunctions>

#include "MCMC.h"
#include "RateStateModel.h"

#include <iostream>
#include <random>
#include <cmath>
#include <tuple>

using namespace std;

// Constants declaration at the top of the class
const double VAR_MULTIPLIER = 10.0;
const double SCALE_FACTOR = 2.38;
const double DC_PERTURBATION = 1e-1;
const double ADAPT_EPSILON = 1e-6;

MCMC::MCMC(RateStateModel model, vector<double> data, uniform_real_distribution<double> qpriors, double qstart,
           int nsamples, int adapt_interval, bool verbose)
    : model(model),
      data(data),
      qpriors(qpriors),
      qstart(qstart),
      nsamples(nsamples),
      nburn(nsamples / 2),
      verbose(verbose),
      adapt_interval(adapt_interval),
      n0(0.01),
      random_engine(12345) { // fixed seed for debugging
    qstart_limits = Eigen::MatrixXd(1, 2);
    qstart_limits << 0.0, 10000.0;
}

Eigen::MatrixXd MCMC::sample() {
    double var_multiplier = VAR_MULTIPLIER;

    vector<double> t, acc_vector, acc_noise;
    model.evaluate(t, acc_vector, acc_noise);

    Eigen::MatrixXd acc_ = Eigen::Map<Eigen::MatrixXd>(acc_noise.data(), 1, acc_noise.size());

    model.setDc(model.getDc() * (1 + DC_PERTURBATION));

    model.evaluate(t, acc_vector, acc_noise);
    Eigen::MatrixXd acc_dq_ = Eigen::Map<Eigen::MatrixXd>(acc_noise.data(), 1, acc_noise.size());

    Eigen::MatrixXd acc(acc_.rows(), acc_.cols());
    acc = acc_.row(0);

    Eigen::MatrixXd acc_dq(acc_dq_.rows(), acc_dq_.cols());
    acc_dq = acc_dq_.row(0);

    std2.push_back((acc - Eigen::Map<Eigen::MatrixXd>(data.data(), 1, data.size())).array().square().sum() /
                (acc.cols() - qpriors(random_engine)));

    Eigen::MatrixXd X = ((acc_dq - acc) / (model.getDc() * DC_PERTURBATION)).transpose();
    Eigen::MatrixXd XTX = X.transpose() * X;

    Vstart = var_multiplier * std2.back() * XTX.inverse();

    Eigen::MatrixXd qstart_vect = Eigen::MatrixXd::Constant(1, 1, qstart);
    Eigen::MatrixXd qparams(qstart_vect.rows(), nsamples);
    qparams.col(0) = qstart_vect;

    Eigen::MatrixXd Vold = Vstart;
    Eigen::MatrixXd Vnew = Vstart;

    double SSqprev = SSqcalc(qparams.col(0))(0);
    int iaccept = 0;
    double SSqnew;
    bool accept;

    double scale = SCALE_FACTOR / sqrt(qparams.rows());

    for (int isample = 1; isample < nsamples; isample++) {
        Eigen::MatrixXd random_values = Eigen::MatrixXd::Random(qparams.rows(), 1);
        Eigen::MatrixXd q_new = qparams.col(isample - 1) + (Vold.llt().matrixL().toDenseMatrix().cast<double>().array() * random_values.array()).matrix();

        tie(accept, SSqnew) = acceptreject(q_new, SSqprev, std2.back());

        if (accept) {
            qparams.col(isample) = q_new;
            SSqprev = SSqnew;
            iaccept++;
        } else {
            qparams.col(isample) = qparams.col(isample - 1);
        }

        if (verbose) {
            cout << qparams.col(isample) << endl;
        }

        double aval = 0.5 * (n0 + data.size());
        double bval = 0.5 * (n0 * std2.back() + SSqprev);
        std2.push_back(1.0 / gamma_distribution<>(aval, 1.0 / bval)(random_engine));

        if ((isample + 1) % adapt_interval == 0 && isample > 0) {
            Eigen::MatrixXd past_samples = qparams.leftCols(isample);
            Eigen::MatrixXd sample_means = past_samples.rowwise().mean();
            Eigen::MatrixXd centered = past_samples;
            for(int i=0; i<centered.cols(); i++){
                centered.col(i) -= sample_means;
            }
            Eigen::MatrixXd cov = (centered * centered.transpose()) / double(isample);
            Vold = scale * scale * cov + Eigen::MatrixXd::Identity(qparams.rows(), qparams.rows()) * ADAPT_EPSILON;
        }
    }

    std2.erase(std2.begin(), std2.begin() + nburn);

    return qparams.rightCols(nsamples - nburn);
}

tuple<bool, double> MCMC::acceptreject(const Eigen::MatrixXd& q_new, double SSqprev, double std2) {
    bool accept = (q_new.array() > qstart_limits(0, 0)).all() && (q_new.array() < qstart_limits(0, 1)).all();

    double SSqnew;
    double accept_prob;

    if (accept) {
        SSqnew = SSqcalc(q_new)(0);

        accept_prob = exp(0.5 * (SSqprev - SSqnew) / std2);

        accept = uniform_real_distribution<>(0.0, 1.0)(random_engine) < accept_prob;
    }

    if (accept) {
        return make_tuple(true, SSqnew);
    } else {
        return make_tuple(false, SSqprev);
    }
}

Eigen::VectorXd MCMC::SSqcalc(const Eigen::MatrixXd& q_new) {
    model.setDc(q_new(0, 0));

    vector<double> t, acc_vector, acc_noise;
    model.evaluate(t, acc_vector, acc_noise);
    Eigen::MatrixXd acc_ = Eigen::Map<Eigen::MatrixXd>(acc_noise.data(), 1, acc_noise.size());

    Eigen::MatrixXd acc(acc_.rows(), acc_.cols());
    acc = acc_.row(0);

    Eigen::VectorXd data_vector = Eigen::Map<Eigen::VectorXd>(data.data(), data.size());

    if (acc.cols() != data_vector.size()) {
        throw runtime_error("Size mismatch between acc and data_vector");
    }

    return (acc.row(0).transpose() - data_vector).array().square();
}
