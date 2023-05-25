#ifndef RATESTATEMODEL_H
#define RATESTATEMODEL_H

#include <vector>

class RateStateModel {
public:
    RateStateModel(int number_time_steps = 500, double start_time = 0.0, double end_time = 50.0);

    void setA(double value);
    void setB(double value);
    void setMuRef(double value);
    void setVRef(double value);
    void setK1(double value);
    void setTStart(double value);
    void setTFinal(double value);
    void setMuTZero(double value);
    void setDc(double value);
    void setRadiationDamping(bool value);

    int getNumTimesteps() const;

    void evaluate(std::vector<double>& t, std::vector<double>& acc, std::vector<double>& acc_noise);

private:
    double a;
    double b;
    double mu_ref;
    double V_ref;
    double k1;
    double t_start;
    double t_final;
    int num_tsteps;
    double delta_t;
    double mu_t_zero;
    double Dc;
    bool RadiationDamping;

    static int friction(double t, const double y[], double dydt[], void* params);
};

#endif  // RATESTATEMODEL_H
