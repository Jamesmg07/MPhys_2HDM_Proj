#include <iostream>
#include <math.h>
#include <fstream>
#include <iomanip>
#include <sys/time.h>
#include <vector>
#include <mpi.h>
#include <sstream>
#include <complex>

using namespace std;
const double pi = 4.0 * atan(1.0);

// === CONFIGURATION: Set box size here ===
const long long int BOX_SIZE = 256; 

const string inp_path = "./";
const string out_path = "/share/centaurus_nas/jmg_temp/dedr_vs_r/";

// Separation range - need fine resolution for accurate derivatives
const double SEP_START = 0.05;   // Starting separation (fraction of box)
const double SEP_END = 0.35;     // Ending separation (fraction of box)
const double SEP_STEP = 0.02;    // Step size for separation scan

const long long int nx = BOX_SIZE;
const long long int ny = BOX_SIZE;
const long long int nz = BOX_SIZE;
const long long int nPos = nx * ny * nz;

const double dx = 0.5;
const double dy = 0.5;
const double dz = 0.5;

const int seed = 73;

// === GAMMA CONFIGURATION ===
// Set TEST_SINGLE_GAMMA = true to test only cases where gamma1 == gamma2
// Set TEST_SINGLE_GAMMA = false to test all combinations of gamma1 and gamma2
const bool TEST_SINGLE_GAMMA = true;

// Gamma parameters to test
const vector<double> gamma_mult_values = {0, 0.5, 1.0};  // Used when TEST_SINGLE_GAMMA = true



const vector<double> gamma_mult_1_values = {0, 0.5, 1.0};  // Used when TEST_SINGLE_GAMMA = false
const vector<double> gamma_mult_2_values = {0, 0.5, 1.0};  // Used when TEST_SINGLE_GAMMA = false

// Monopole parameters (same as 2gamma_loop.cpp)
const double monopole1_vx = 0.0;
const double monopole1_vy = 0.0;
const double monopole1_vz = 0.0;
const double monopole2_vx = 0.0;
const double monopole2_vy = 0.0;
const double monopole2_vz = -0.0;
const double monopole1_x_offset = 0.0;
const double monopole1_y_offset = 0.0;
const double monopole2_x_offset = 0.0;
const double monopole2_y_offset = 0.0;
const double monopole_grid_spacing = 0.01;
const double monopole_prefactor = pow(2, -1.5);

// 2HDM parameters (same as 2gamma_loop.cpp)
const long double m_h = 125;
const long double V_sm = 246;
const long double m_H = 0;
const long double m_A = 0;
const long double m_H_pm = 125;
const long double M_h = m_h / m_h;
const long double v_sm = V_sm / V_sm;
const long double M_H = m_H / m_h;
const long double M_A = m_A / m_h;
const long double M_H_pm = m_H_pm / m_h;
const long double a = 0.25*pi;
const long double b = 0.25*pi;
const long double s_a = sin(a);
const long double c_a = cos(a);
const long double s_b = sin(b);
const long double c_b = cos(b);
const long double t_b = tan(b);
const long double ct_b = pow(tan(b), -1);
const long double mu_1_sq = (1 / pow(M_h, 2)) * 0.5 * ((pow(M_h, 2) * pow(c_a, 2)) + (pow(M_H, 2) * pow(s_a, 2)) + ((pow(M_h, 2) - pow(M_H, 2)) * c_a * s_a * t_b));
const long double mu_2_sq = (1 / pow(M_h, 2)) * 0.5 * ((pow(M_h, 2) * pow(s_a, 2)) + (pow(M_H, 2) * pow(c_a, 2)) + ((pow(M_h, 2) - pow(M_H, 2)) * c_a * s_a * ct_b));
const double lambda_1 = (pow(v_sm, 2) / pow(M_h, 2)) * (pow(M_h, 2) * pow(c_a, 2) + pow(M_H, 2) * pow(s_a, 2)) / (2 * pow(c_b, 2) * pow(v_sm, 2));
const double lambda_2 = (pow(v_sm, 2) / pow(M_h, 2)) * (pow(M_h, 2) * pow(s_a, 2) + pow(M_H, 2) * pow(c_a, 2)) / (2 * pow(s_b, 2) * pow(v_sm, 2));
const double lambda_3 = (pow(v_sm, 2) / pow(M_h, 2)) * ((pow(M_h, 2) - pow(M_H, 2)) * c_a * s_a + 2 * pow(M_H_pm, 2) * c_b * s_b) / (c_b * s_b * pow(v_sm, 2));
const double l4_m_l5 = (pow(v_sm, 2) / pow(M_h, 2)) * (-2 * pow(M_H_pm, 2)) / (pow(v_sm, 2));
const double l4_p_l5 = (pow(v_sm, 2) / pow(M_h, 2)) * (2 * (pow(M_A, 2) - pow(M_H_pm, 2))) / (pow(v_sm, 2));
const long double v1 = c_b * v_sm;
const long double v2 = s_b * v_sm;


void calculateMonopoleFieldsAtPoint(long long int global_index, double field_values[8],
                                   long long int nx, long long int ny, long long int nz,
                                   double dx, double dy, double dz,
                                   double x1, double y1, double z1, double x2, double y2, double z2,
                                   const vector<double>& k_, const vector<double>& k_p,
                                   double gamma_param_1, double gamma_param_2) {

    long long int i_coord = global_index / (ny * nz);
    long long int j_coord = (global_index / nz) % ny;
    long long int k_coord = global_index % nz;
    
    double v1_mag = sqrt(monopole1_vx*monopole1_vx + monopole1_vy*monopole1_vy + monopole1_vz*monopole1_vz);
    double v2_mag = sqrt(monopole2_vx*monopole2_vx + monopole2_vy*monopole2_vy + monopole2_vz*monopole2_vz);
    
    double gamma1 = (v1_mag > 1e-10) ? 1.0/sqrt(1.0 - v1_mag*v1_mag) : 1.0;
    double gamma2 = (v2_mag > 1e-10) ? 1.0/sqrt(1.0 - v2_mag*v2_mag) : 1.0;

    double v1_hat_x = (v1_mag > 1e-10) ? monopole1_vx / v1_mag : 0.0;
    double v1_hat_y = (v1_mag > 1e-10) ? monopole1_vy / v1_mag : 0.0;
    double v1_hat_z = (v1_mag > 1e-10) ? monopole1_vz / v1_mag : 0.0;
    
    double v2_hat_x = (v2_mag > 1e-10) ? monopole2_vx / v2_mag : 0.0;
    double v2_hat_y = (v2_mag > 1e-10) ? monopole2_vy / v2_mag : 0.0;
    double v2_hat_z = (v2_mag > 1e-10) ? monopole2_vz / v2_mag : 0.0;

    double x_1 = (i_coord - x1) * dx;
    double y_1 = (j_coord - y1) * dy;
    double z_1 = (k_coord - z1) * dz;

    double v_dot_r1 = x_1*v1_hat_x + y_1*v1_hat_y + z_1*v1_hat_z;

    double x_1_prime = x_1 + (gamma1-1)*(v_dot_r1)*v1_hat_x;
    double y_1_prime = y_1 + (gamma1-1)*(v_dot_r1)*v1_hat_y;
    double z_1_prime = z_1 + (gamma1-1)*(v_dot_r1)*v1_hat_z;

    double r_1 = sqrt(x_1_prime*x_1_prime + y_1_prime*y_1_prime + z_1_prime*z_1_prime);
    double r_pos_1 = r_1 / monopole_grid_spacing;
    int r_c_1 = static_cast<int>(round(r_pos_1)); 
    int r_m_1 = r_c_1 - 1;
    int r_p_1 = r_c_1 + 1;

    double k_1 = 0.0;
    double k_1_p = 0.0;
    
    if (r_p_1 >= (k_.size())) {
        k_1 = 1.0;
        k_1_p = 0.0;
    } else if (r_c_1 == 0) {
        k_1 = ((( - (r_c_1 - r_pos_1) * k_[r_p_1] )) + ((r_p_1 - r_pos_1) * k_[r_c_1]));
        k_1_p = ((( - (r_c_1 - r_pos_1) * k_p[r_p_1] )) + ((r_p_1 - r_pos_1) * k_p[r_c_1]));
    } else {
        k_1 = ((((r_m_1 - r_pos_1) * (r_c_1 - r_pos_1) * k_[r_p_1]) / 2) - (((r_m_1 - r_pos_1) * (r_p_1 - r_pos_1) * k_[r_c_1])) + (((r_c_1 - r_pos_1) * (r_p_1 - r_pos_1) * k_[r_m_1]) / 2));
        k_1_p = ((((r_m_1 - r_pos_1) * (r_c_1 - r_pos_1) * k_p[r_p_1]) / 2) - (((r_m_1 - r_pos_1) * (r_p_1 - r_pos_1) * k_p[r_c_1])) + (((r_c_1 - r_pos_1) * (r_p_1 - r_pos_1) * k_p[r_m_1]) / 2));
    }

    double x_2 = (i_coord - x2) * dx;
    double y_2 = (j_coord - y2) * dy;
    double z_2 = (k_coord - z2) * dz;
                
    double v_dot_r2 = x_2*v2_hat_x + y_2*v2_hat_y + z_2*v2_hat_z;

    double x_2_prime = x_2 + (gamma2-1)*(v_dot_r2)*v2_hat_x;
    double y_2_prime = y_2 + (gamma2-1)*(v_dot_r2)*v2_hat_y;
    double z_2_prime = z_2 + (gamma2-1)*(v_dot_r2)*v2_hat_z;

    double r_2 = sqrt(x_2_prime*x_2_prime + y_2_prime*y_2_prime + z_2_prime*z_2_prime);

    double r_pos_2 = r_2 / monopole_grid_spacing;
    int r_c_2 = static_cast<int>(round(r_pos_2)); 
    int r_m_2 = r_c_2 - 1;
    int r_p_2 = r_c_2 + 1;

    double k_2 = 0.0;
    double k_2_p = 0.0;
    
    if (r_p_2 >= (k_.size())) {
        k_2 = 1.0;
        k_2_p = 0.0;
    } else if (r_c_2 == 0) {
        k_2 = ((( - (r_c_2 - r_pos_2) * k_[r_p_2] )) + ((r_p_2 - r_pos_2) * k_[r_c_2]));
        k_2_p = ((( - (r_c_2 - r_pos_2) * k_p[r_p_2] )) + ((r_p_2 - r_pos_2) * k_p[r_c_2]));
    } else {
        k_2 = ((((r_m_2 - r_pos_2) * (r_c_2 - r_pos_2) * k_[r_p_2]) / 2) - (((r_m_2 - r_pos_2) * (r_p_2 - r_pos_2) * k_[r_c_2])) + (((r_c_2 - r_pos_2) * (r_p_2 - r_pos_2) * k_[r_m_2]) / 2));
        k_2_p = ((((r_m_2 - r_pos_2) * (r_c_2 - r_pos_2) * k_p[r_p_2]) / 2) - (((r_m_2 - r_pos_2) * (r_p_2 - r_pos_2) * k_p[r_c_2])) + (((r_c_2 - r_pos_2) * (r_p_2 - r_pos_2) * k_p[r_m_2]) / 2));
    }

    double g_1_p = (k_1 - k_1_p);
    double g_1 = (k_1 + k_1_p);
    double g_2_p = (k_2 - k_2_p);
    double g_2 = (k_2 + k_2_p);

    complex<double> u_1[2][2];
    complex<double> u_2[2][2];

    if ( z_1_prime == r_1 ) {
        u_1[0][0] = complex<double>(1.0, 0.0); u_1[0][1] = complex<double>(0.0, 0.0);
        u_1[1][0] = complex<double>(0.0, 0.0); u_1[1][1] = complex<double>(1.0, 0.0);
        u_2[0][0] = complex<double>(0.0, 0.0); u_2[0][1] = complex<double>(-1.0, 0.0);
        u_2[1][0] = complex<double>(1.0, 0.0); u_2[1][1] = complex<double>(0.0, 0.0);
    } else if ( z_2_prime == -r_2 ) {
        u_1[0][0] = complex<double>(0.0, 0.0); u_1[0][1] = complex<double>(-1.0, 0.0);
        u_1[1][0] = complex<double>(1.0, 0.0); u_1[1][1] = complex<double>(0.0, 0.0);
        u_2[0][0] = complex<double>(-1.0, 0.0); u_2[0][1] = complex<double>(0.0, 0.0);
        u_2[1][0] = complex<double>(0.0, 0.0); u_2[1][1] = complex<double>(-1.0, 0.0);
    } else if ( z_1_prime!= r_1 && z_2_prime != -r_2 && z_2_prime== r_2) {
        u_1[0][0] = complex<double>(0.0, 0.0); u_1[0][1] = complex<double>(-1.0, 0.0);
        u_1[1][0] = complex<double>(1.0, 0.0); u_1[1][1] = complex<double>(0.0, 0.0);
        u_2[0][0] = complex<double>(0.0, 0.0); u_2[0][1] = complex<double>(-1.0, 0.0);
        u_2[1][0] = complex<double>(1.0, 0.0); u_2[1][1] = complex<double>(0.0, 0.0);
    } else {
        double cos_1 = pow(0.5 * (1 + (z_1_prime / r_1)), 0.5);
        u_1[0][0] = complex<double>(cos_1, 0.0);
        u_1[0][1] = complex<double>(- x_1_prime / (2 * r_1 * cos_1), y_1_prime / (2 * r_1 * cos_1));
        u_1[1][0] = complex<double>(x_1_prime / (2 * r_1 * cos_1), y_1_prime / (2 * r_1 * cos_1));
        u_1[1][1] = complex<double>(cos_1, 0.0);

        double sin_2 = pow(0.5 * (1 - (z_2_prime / r_2)), 0.5);
        u_2[0][0] = complex<double>(- sin_2, 0.0);
        u_2[0][1] = complex<double>(- x_2_prime / (2 * r_2 * sin_2), y_2_prime / (2 * r_2 * sin_2));
        u_2[1][0] = complex<double>(x_2_prime / (2 * r_2 * sin_2), y_2_prime / (2 * r_2 * sin_2));
        u_2[1][1] = complex<double>(- sin_2, 0.0);
    }

    double sqrt2_inv = 1.0 / sqrt(2.0);
    complex<double> M[2][2] = {{sqrt2_inv, sqrt2_inv}, {-sqrt2_inv, sqrt2_inv}};
    complex<double> TP[4][4];
    complex<double> T1[2][2] = {{exp(complex<double>(0, 0.5 * gamma_param_1)), complex<double>(0.0, 0.0)}, {complex<double>(0.0, 0.0), exp(complex<double>(0, -0.5 * gamma_param_1))}};
    complex<double> T2[2][2] = {{exp(complex<double>(0, 0.5 * gamma_param_2)), complex<double>(0.0, 0.0)}, {complex<double>(0.0, 0.0), exp(complex<double>(0, -0.5 * gamma_param_2))}};

    complex<double> C1[2][2], C2[2][2];
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 2; ++col) {
            C1[row][col] = complex<double>(0.0, 0.0);
            C2[row][col] = complex<double>(0.0, 0.0);
            for (int index = 0; index < 2; ++index) {
                C1[row][col] += T1[row][index] * u_2[index][col];
                C2[row][col] += T2[row][index] * u_2[index][col];
            }
        }
    }

    complex<double> A1[2][2], A2[2][2];
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 2; ++col) {
            A1[row][col] = complex<double>(0.0, 0.0);
            A2[row][col] = complex<double>(0.0, 0.0);
            for (int index = 0; index < 2; ++index) {
                A1[row][col] += u_1[row][index] * C1[index][col];
                A2[row][col] += u_1[row][index] * C2[index][col];
            }
        }
    }

    complex<double> B1[2][2], B2[2][2];
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 2; ++col) {
            B1[row][col] = complex<double>(0.0, 0.0);
            B2[row][col] = complex<double>(0.0, 0.0);
            for (int index = 0; index < 2; ++index) {
                B1[row][col] += A1[row][index] * M[index][col];
                B2[row][col] += A2[row][index] * M[index][col];
            }
        }
    }

    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            for (int x = 0; x < 2; ++x) {
                for (int y = 0; y < 2; ++y) {
                    TP[2 * r + x][2 * c + y] = B1[r][c] * B2[x][y];
                }
            }
        }
    }

    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            TP[r][c] *= monopole_prefactor;
        }
    }

    double phi_both[4] = { (- g_1_p * g_2_p), (g_1 * g_2), (- g_1 * g_2), (g_1_p * g_2_p) };
    complex<double> phi[4];

    for (int r = 0; r < 4; ++r) {
        phi[r] = complex<double>(0.0, 0.0);
        for (int c = 0; c < 4; ++c) {
            phi[r] += TP[r][c] * phi_both[c];
        }
    }

    field_values[0] = phi[0].real();
    field_values[1] = phi[0].imag();
    field_values[2] = phi[1].real();
    field_values[3] = phi[1].imag();
    field_values[4] = phi[2].real();
    field_values[5] = phi[2].imag();
    field_values[6] = phi[3].real();
    field_values[7] = phi[3].imag();
}

double calculateTotalEnergy(double x1, double y1, double z1, double x2, double y2, double z2,
                           const vector<double>& k_, const vector<double>& k_p,
                           double gamma_param_1, double gamma_param_2,
                           int rank, int size) {

    
    long long int chunk = nPos / size;
    long long int chunkRem = nPos - size * chunk;
    long long int coreSize = (rank >= chunkRem) ? chunk : chunk + 1;
    long long int coreStart = (rank < chunkRem) ? rank * (chunk + 1) : rank * chunk + chunkRem;
    long long int coreEnd = (rank < chunkRem) ? (rank + 1) * (chunk + 1) : (rank + 1) * chunk + chunkRem;

    long long int remFront = coreStart % (ny * nz);
    long long int remBack = coreEnd % (ny * nz);
    long long int frontHaloSize = (remFront == 0) ? 2 * ny * nz : 2 * ny * nz + remFront;
    long long int backHaloSize = (remBack == 0) ? 2 * ny * nz : 4 * ny * nz - remBack;
    long long int dataStart = coreStart - frontHaloSize;

    double local_total_energy = 0.0;

    for (long long int i = frontHaloSize; i < coreSize + frontHaloSize; i++) {
        double center_fields[8], x_plus_fields[8], x_minus_fields[8];
        double y_plus_fields[8], y_minus_fields[8], z_plus_fields[8], z_minus_fields[8];
        
        long long int global_pos = i + dataStart;
        long long int i_coord = global_pos / (ny * nz);
        long long int j_coord = (global_pos / nz) % ny;
        long long int k_coord = global_pos % nz;
        
        calculateMonopoleFieldsAtPoint(global_pos, center_fields, nx, ny, nz, dx, dy, dz,
                                    x1, y1, z1, x2, y2, z2, k_, k_p, gamma_param_1, gamma_param_2);
        
        if (i_coord < nx - 1) {
            calculateMonopoleFieldsAtPoint(global_pos + ny*nz, x_plus_fields, nx, ny, nz, dx, dy, dz,
                                        x1, y1, z1, x2, y2, z2, k_, k_p, gamma_param_1, gamma_param_2);
        } else {
            for (int c = 0; c < 8; c++) x_plus_fields[c] = center_fields[c];
        }
        
        if (i_coord > 0) {
            calculateMonopoleFieldsAtPoint(global_pos - ny*nz, x_minus_fields, nx, ny, nz, dx, dy, dz,
                                        x1, y1, z1, x2, y2, z2, k_, k_p, gamma_param_1, gamma_param_2);
        } else {
            for (int c = 0; c < 8; c++) x_minus_fields[c] = center_fields[c];
        }
        
        if (j_coord < ny - 1) {
            calculateMonopoleFieldsAtPoint(global_pos + nz, y_plus_fields, nx, ny, nz, dx, dy, dz,
                                        x1, y1, z1, x2, y2, z2, k_, k_p, gamma_param_1, gamma_param_2);
        } else {
            for (int c = 0; c < 8; c++) y_plus_fields[c] = center_fields[c];
        }
        
        if (j_coord > 0) {
            calculateMonopoleFieldsAtPoint(global_pos - nz, y_minus_fields, nx, ny, nz, dx, dy, dz,
                                        x1, y1, z1, x2, y2, z2, k_, k_p, gamma_param_1, gamma_param_2);
        } else {
            for (int c = 0; c < 8; c++) y_minus_fields[c] = center_fields[c];
        }
        
        if (k_coord < nz - 1) {
            calculateMonopoleFieldsAtPoint(global_pos + 1, z_plus_fields, nx, ny, nz, dx, dy, dz,
                                        x1, y1, z1, x2, y2, z2, k_, k_p, gamma_param_1, gamma_param_2);
        } else {
            for (int c = 0; c < 8; c++) z_plus_fields[c] = center_fields[c];
        }
        
        if (k_coord > 0) {
            calculateMonopoleFieldsAtPoint(global_pos - 1, z_minus_fields, nx, ny, nz, dx, dy, dz,
                                        x1, y1, z1, x2, y2, z2, k_, k_p, gamma_param_1, gamma_param_2);
        } else {
            for (int c = 0; c < 8; c++) z_minus_fields[c] = center_fields[c];
        }

        double local_energy = 0.0;
        
        for (int comp = 0; comp < 8; comp++) {
            double fieldx_comp, fieldy_comp, fieldz_comp;

            if (i_coord == 0) {
                fieldx_comp = (x_plus_fields[comp] - center_fields[comp]) / dx;
            } else if (i_coord == nx - 1) {
                fieldx_comp = (center_fields[comp] - x_minus_fields[comp]) / dx;
            } else {
                fieldx_comp = (x_plus_fields[comp] - x_minus_fields[comp]) / (2.0 * dx);
            }

            if (j_coord == 0) {
                fieldy_comp = (y_plus_fields[comp] - center_fields[comp]) / dy;
            } else if (j_coord == ny - 1) {
                fieldy_comp = (center_fields[comp] - y_minus_fields[comp]) / dy;
            } else {
                fieldy_comp = (y_plus_fields[comp] - y_minus_fields[comp]) / (2.0 * dy);
            }

            if (k_coord == 0) {
                fieldz_comp = (z_plus_fields[comp] - center_fields[comp]) / dz;
            } else if (k_coord == nz - 1) {
                fieldz_comp = (center_fields[comp] - z_minus_fields[comp]) / dz;
            } else {
                fieldz_comp = (z_plus_fields[comp] - z_minus_fields[comp]) / (2.0 * dz);
            }

            local_energy += (fieldx_comp*fieldx_comp + fieldy_comp*fieldy_comp + fieldz_comp*fieldz_comp);
        }

        double f0 = center_fields[0], f1 = center_fields[1], f2 = center_fields[2], f3 = center_fields[3];
        double f4 = center_fields[4], f5 = center_fields[5], f6 = center_fields[6], f7 = center_fields[7];
        
        double phi1_sq = f0*f0 + f1*f1 + f2*f2 + f3*f3;
        double phi2_sq = f4*f4 + f5*f5 + f6*f6 + f7*f7;
        double phi1_dot_phi2 = f0*f4 + f1*f5 + f2*f6 + f3*f7;
        double phi1_cross_phi2 = f0*f5 - f1*f4 + f2*f7 - f3*f6;
        
        local_energy += (-mu_1_sq * phi1_sq - mu_2_sq * phi2_sq +
                        lambda_1 * phi1_sq * phi1_sq +
                        lambda_2 * phi2_sq * phi2_sq +
                        lambda_3 * phi1_sq * phi2_sq +
                        l4_m_l5 * phi1_dot_phi2 * phi1_dot_phi2 +
                        l4_p_l5 * phi1_cross_phi2 * phi1_cross_phi2);

        local_total_energy += local_energy * dx * dy * dz;
    }
    
    double global_total_energy = 0.0;
    MPI_Reduce(&local_total_energy, &global_total_energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double vev_norm = (1.0/8.0) * pow((nx) * dx, 3.0);
    global_total_energy += vev_norm;

    return global_total_energy;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        cout << "=== BINDING FORCE STUDY (dE/dR vs R) ===" << endl;
        cout << "Box size: " << BOX_SIZE << "³" << endl;
        cout << "Separation range: " << SEP_START << " to " << SEP_END << " (fraction of box)" << endl;
        cout << "Step size: " << SEP_STEP << endl;
        
        if (TEST_SINGLE_GAMMA) {
            cout << "Mode: Testing single gamma (γ₁ = γ₂)" << endl;
            cout << "Gamma values: ";
            for (auto g : gamma_mult_values) cout << g << " ";
            cout << endl;
            cout << "Total combinations: " << gamma_mult_values.size() << endl;
        } else {
            cout << "Mode: Testing all gamma combinations" << endl;
            cout << "Gamma₁ values: ";
            for (auto g : gamma_mult_1_values) cout << g << " ";
            cout << endl;
            cout << "Gamma₂ values: ";
            for (auto g : gamma_mult_2_values) cout << g << " ";
            cout << endl;
            cout << "Total combinations: " << gamma_mult_1_values.size() * gamma_mult_2_values.size() << endl;
        }
    }

    // Load monopole profile
    string fields_ic_data = inp_path + "SOR_Fields.txt";
    ifstream test_file(fields_ic_data);
    if (!test_file.good()) {
        if (rank == 0) {
            cout << "ERROR: Cannot find initial condition file: " << fields_ic_data << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    test_file.close();

    vector<double> k_, k_p;
    ifstream inputFile(fields_ic_data);
    double k_val, k_p_val;
    while (inputFile >> k_val >> k_p_val) {
        k_.push_back(k_val);
        k_p.push_back(k_p_val);
    }
    inputFile.close();

    // Generate separation values
    vector<double> separations;
    for (double sep = SEP_START; sep <= SEP_END + 1e-10; sep += SEP_STEP) {
        separations.push_back(sep);
    }

    if (rank == 0) {
        cout << "Number of separation points: " << separations.size() << endl;
    }


    if (TEST_SINGLE_GAMMA) {
        // Loop over single gamma value (γ₁ = γ₂)
        for (const auto& gamma_val : gamma_mult_values) {
            
            double gamma_param_1 = gamma_val * pi;
            double gamma_param_2 = gamma_val * pi;

            if (rank == 0) {
                cout << "\n=== Processing γ₁ = γ₂ = " << gamma_val << "π ===" << endl;
            }

            // Calculate energies at all separations
            vector<double> R_values, E_values;

            for (const auto& sep : separations) {
                double x1 = 0.5 * (nx - 1);
                double y1 = 0.5 * (ny - 1);
                double z1 = 0.5 * (nz - 1) + sep * nz;

                double x2 = 0.5 * (nx - 1);
                double y2 = 0.5 * (ny - 1);
                double z2 = 0.5 * (nz - 1) - sep * nz;

                double E = calculateTotalEnergy(x1, y1, z1, x2, y2, z2, k_, k_p, 
                                               gamma_param_1, gamma_param_2, rank, size);

                if (rank == 0) {
                    double R_real = 2 * sep * dz * BOX_SIZE;
                    R_values.push_back(R_real);
                    E_values.push_back(E);
                }
            }

            // Calculate dE/dR using finite differences
            if (rank == 0) {
                vector<double> dE_dR_values;
                
                for (size_t i = 0; i < R_values.size(); i++) {
                    double dE_dR;
                    
                    if (i == 0) {
                        dE_dR = (E_values[i+1] - E_values[i]) / (R_values[i+1] - R_values[i]);
                    } else if (i == R_values.size() - 1) {
                        dE_dR = (E_values[i] - E_values[i-1]) / (R_values[i] - R_values[i-1]);
                    } else {
                        dE_dR = (E_values[i+1] - E_values[i-1]) / (R_values[i+1] - R_values[i-1]);
                    }
                    
                    dE_dR_values.push_back(dE_dR);
                }

                // Save to CSV
                string gamma_str = to_string(gamma_val).substr(0,4);
                gamma_str.erase(gamma_str.find_last_not_of('0') + 1);
                gamma_str.erase(gamma_str.find_last_not_of('.') + 1);

                string filename = "binding_force_gamma1_" + gamma_str + "pi_gamma2_" + gamma_str + 
                                 "pi_box" + to_string(BOX_SIZE) + "_seed" + to_string(seed) + ".csv";
                ofstream outfile(out_path + filename);
                
                outfile << "R_real,E_total,dE_dR" << endl;
                outfile << fixed << setprecision(12);
                
                for (size_t i = 0; i < R_values.size(); i++) {
                    outfile << R_values[i] << "," << E_values[i] << "," << dE_dR_values[i] << endl;
                }
                
                outfile.close();
                cout << "Saved: " << filename << endl;
            }
        }
        
    } else {
        // Original full grid loop (γ₁ and γ₂ independent)
        for (const auto& gamma1 : gamma_mult_1_values) {
            for (const auto& gamma2 : gamma_mult_2_values) {
                
                double gamma_param_1 = gamma1 * pi;
                double gamma_param_2 = gamma2 * pi;

                if (rank == 0) {
                    cout << "\n=== Processing γ₁=" << gamma1 << "π, γ₂=" << gamma2 << "π ===" << endl;
                }

                // Calculate energies at all separations
                vector<double> R_values, E_values;

                for (const auto& sep : separations) {
                    double x1 = 0.5 * (nx - 1);
                    double y1 = 0.5 * (ny - 1);
                    double z1 = 0.5 * (nz - 1) + sep * nz;

                    double x2 = 0.5 * (nx - 1);
                    double y2 = 0.5 * (ny - 1);
                    double z2 = 0.5 * (nz - 1) - sep * nz;

                    double E = calculateTotalEnergy(x1, y1, z1, x2, y2, z2, k_, k_p, 
                                                   gamma_param_1, gamma_param_2, rank, size);

                    if (rank == 0) {
                        double R_real = 2 * sep * dz * BOX_SIZE;
                        R_values.push_back(R_real);
                        E_values.push_back(E);
                    }
                }

                if (rank == 0) {
                    vector<double> dE_dR_values;
                    
                    for (size_t i = 0; i < R_values.size(); i++) {
                        double dE_dR;
                        
                        if (i == 0) {
                            dE_dR = (E_values[i+1] - E_values[i]) / (R_values[i+1] - R_values[i]);
                        } else if (i == R_values.size() - 1) {
                            dE_dR = (E_values[i] - E_values[i-1]) / (R_values[i] - R_values[i-1]);
                        } else {
                            dE_dR = (E_values[i+1] - E_values[i-1]) / (R_values[i+1] - R_values[i-1]);
                        }
                        
                        dE_dR_values.push_back(dE_dR);
                    }

                    string gamma1_str = to_string(gamma1).substr(0,4);
                    string gamma2_str = to_string(gamma2).substr(0,4);
                    gamma1_str.erase(gamma1_str.find_last_not_of('0') + 1);
                    gamma1_str.erase(gamma1_str.find_last_not_of('.') + 1);
                    gamma2_str.erase(gamma2_str.find_last_not_of('0') + 1);
                    gamma2_str.erase(gamma2_str.find_last_not_of('.') + 1);

                    string filename = "binding_force_gamma1_" + gamma1_str + "pi_gamma2_" + gamma2_str + 
                                     "pi_box" + to_string(BOX_SIZE) + "_seed" + to_string(seed) + ".csv";
                    ofstream outfile(out_path + filename);
                    
                    outfile << "R_real,E_total,dE_dR" << endl;
                    outfile << fixed << setprecision(12);
                    
                    for (size_t i = 0; i < R_values.size(); i++) {
                        outfile << R_values[i] << "," << E_values[i] << "," << dE_dR_values[i] << endl;
                    }
                    
                    outfile.close();
                    cout << "Saved: " << filename << endl;
                }
            }
        }
    }

    if (rank == 0) {
        cout << "\n=== Binding force study completed! ===" << endl;
    }

    MPI_Finalize();
    return 0;
}
