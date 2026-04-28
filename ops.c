
#include <stdio.h>
#include <math.h>

#define PI (4 * atan(1.0))

void compute_u(const double *ur, const double *ui, size_t n, double ur_PF, double ui_PF, double *u);
void compute_fe(const double *t, const double *ur, const double *ui, const double *omega_ref, size_t n,
        double ur_PF, double ui_PF, double f0, double *fe);

void compute_u(const double *ur, const double *ui, size_t n, double ur_PF, double ui_PF, double *u) {
    double ur2, ui2;
    size_t i;
    for (i = 0; i < n; i++) {
        ur2 = (ur[i] + ur_PF) * (ur[i] + ur_PF);
        ui2 = (ui[i] + ui_PF) * (ui[i] + ui_PF);
        u[i] = sqrt(ur2 + ui2);
    }
}

void compute_fe(const double *t, const double *ur, const double *ui, const double *omega_ref, size_t n,
        double ur_PF, double ui_PF, double f0, double *fe) {
    size_t i;
    double delta, delta_prev, coeff;
    coeff = 1.0 / (2 * PI * f0);
    delta_prev = atan2(ui[0] + ui_PF, ur[0] + ur_PF);
    for (i = 1; i < n; i++) {
        delta = atan2(ui[i] + ui_PF, ur[i] + ur_PF);
        fe[i] = ((delta - delta_prev) / (t[i] - t[i - 1])) * coeff + omega_ref[i];
        delta_prev = delta;
    }
    fe[0] = fe[1];
}

