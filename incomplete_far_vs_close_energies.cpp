#include <iostream>
#include <math.h>
#include <fstream>
#include <iomanip>
#include <sys/time.h>
#include <vector>
#include <mpi.h>
#include <random>
#include <sstream>
#include <complex>

using namespace std;
const double pi = 4.0 * atan(1.0);

// Simulation parameters
const string inp_path = "./";
const string out_path = "/share/centaurus_nas/jmg_temp/binding_energy_study/";

// Simulation settings
const long long int GRID_SIZE = 128;  // Well-converged grid size
const double dx = 0.5;  // Well-converged dx value
const double dy = 0.5;
const double dz = 0.5;

const int seed = 42;

// Separation distances to test (in lattice units)
const vector<double> SEPARATIONS = {16, 24, 32, 40, 48, 64, 80, 96, 112};

// Gamma parameters
const double gamma_mult_1 = 0.0;
const double gamma_mult_2 = 0.0;

// [Copy all the monopole and 2HDM parameters from energy_vs_grid.cpp]
// Monopole Boost Parameters
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

const int nb_fields = 8;

// [Copy calculateMonopoleFieldsAtPoint function from energy_vs_grid.cpp - it's identical]
void calculateMonopoleFieldsAtPoint(long long int global_index, double field_values[8],
                                   long long int nx, long long int ny, long long int nz,
                                   double dx, double dy, double dz,
                                   double x1, double y1, double z1, double x2, double y2, double z2,
                                   const vector<double>& k_, const vector<double>& k_p,
                                   double gamma_param_1, double gamma_param_2,
                                   bool monopole_only, bool antimonopole_only) {
    
    // ...existing code from energy_vs_grid.cpp...
    // [This is the exact same function, just copy it entirely]
    
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
        k_1 = ((( - (r_c_1 - r_pos_1) * k_[r_p_1] )) 
                + ((r_p_1 - r_pos_1) * k_[r_c_1]));
        k_1_p = ((( - (r_c_1 - r_pos_1) * k_p[r_p_1] )) 
                + ((r_p_1 - r_pos_1) * k_p[r_c_1]));
    } else {
        k_1 = ((((r_m_1 - r_pos_1) * (r_c_1 - r_pos_1) * k_[r_p_1]) / 2) 
            - (((r_m_1 - r_pos_1) * (r_p_1 - r_pos_1) * k_[r_c_1])) 
            + (((r_c_1 - r_pos_1) * (r_p_1 - r_pos_1) * k_[r_m_1]) / 2));
        k_1_p = ((((r_m_1 - r_pos_1) * (r_c_1 - r_pos_1) * k_p[r_p_1]) / 2) 
                - (((r_m_1 - r_pos_1) * (r_p_1 - r_pos_1) * k_p[r_c_1])) 
                + (((r_c_1 - r_pos_1) * (r_p_1 - r_pos_1) * k_p[r_m_1]) / 2));
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
        k_2 = ((( - (r_c_2 - r_pos_2) * k_[r_p_2] )) 
                + ((r_p_2 - r_pos_2) * k_[r_c_2]));
        k_2_p = ((( - (r_c_2 - r_pos_2) * k_p[r_p_2] )) 
                + ((r_p_2 - r_pos_2) * k_p[r_c_2]));
    } else {
        k_2 = ((((r_m_2 - r_pos_2) * (r_c_2 - r_pos_2) * k_[r_p_2]) / 2) 
            - (((r_m_2 - r_pos_2) * (r_p_2 - r_pos_2) * k_[r_c_2])) 
            + (((r_c_2 - r_pos_2) * (r_p_2 - r_pos_2) * k_[r_m_2]) / 2));
        k_2_p = ((((r_m_2 - r_pos_2) * (r_c_2 - r_pos_2) * k_p[r_p_2]) / 2) 
                - (((r_m_2 - r_pos_2) * (r_p_2 - r_pos_2) * k_p[r_c_2])) 
                + (((r_c_2 - r_pos_2) * (r_p_2 - r_pos_2) * k_p[r_m_2]) / 2));
    }

    // Modify g values based on mode
    double g_1_p, g_1, g_2_p, g_2;
    
    if (monopole_only) {
        // Only monopole: set antimonopole to vacuum (0, 1, 0, 0)
        g_1_p = (k_1 - k_1_p);
        g_1 = (k_1 + k_1_p);
        g_2_p = 0.0;  // k_2 - k_2_p where both are 0
        g_2 = 1.0;     // k_2 + k_2_p = 0 + 1 = 1 (vacuum)
    } else if (antimonopole_only) {
        // Only antimonopole: set monopole to vacuum (0, 1, 0, 0)
        g_1_p = 0.0;
        g_1 = 1.0;
        g_2_p = (k_2 - k_2_p);
        g_2 = (k_2 + k_2_p);
    } else {
        // Both monopoles
        g_1_p = (k_1 - k_1_p);
        g_1 = (k_1 + k_1_p);
        g_2_p = (k_2 - k_2_p);
        g_2 = (k_2 + k_2_p);
    }

    complex<double> u_1[2][2];
    complex<double> u_2[2][2];

    // [Continue with exact same u_1, u_2 calculation as energy_vs_grid.cpp]
    if ( z_1_prime == r_1 ) {
        u_1[0][0] = complex<double>(1.0, 0.0);  
        u_1[0][1] = complex<double>(0.0, 0.0); 
        u_1[1][0] = complex<double>(0.0, 0.0); 
        u_1[1][1] = complex<double>(1.0, 0.0);

        u_2[0][0] = complex<double>(0.0, 0.0);  
        u_2[0][1] = complex<double>(-1.0, 0.0); 
        u_2[1][0] = complex<double>(1.0, 0.0); 
        u_2[1][1] = complex<double>(0.0, 0.0);

    } else if ( z_2_prime == -r_2 ) {
        u_1[0][0] = complex<double>(0.0, 0.0);  
        u_1[0][1] = complex<double>(-1.0, 0.0); 
        u_1[1][0] = complex<double>(1.0, 0.0); 
        u_1[1][1] = complex<double>(0.0, 0.0);

        u_2[0][0] = complex<double>(-1.0, 0.0);  
        u_2[0][1] = complex<double>(0.0, 0.0); 
        u_2[1][0] = complex<double>(0.0, 0.0); 
        u_2[1][1] = complex<double>(-1.0, 0.0);
                
    } else if ( z_1_prime!= r_1 && z_2_prime != -r_2 && z_2_prime== r_2) {
        u_1[0][0] = complex<double>(0.0, 0.0);  
        u_1[0][1] = complex<double>(-1.0, 0.0); 
        u_1[1][0] = complex<double>(1.0, 0.0); 
        u_1[1][1] = complex<double>(0.0, 0.0);

        u_2[0][0] = complex<double>(0.0, 0.0);  
        u_2[0][1] = complex<double>(-1.0, 0.0); 
        u_2[1][0] = complex<double>(1.0, 0.0); 
        u_2[1][1] = complex<double>(0.0, 0.0);            

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

    // [Continue with T matrix calculations - exact same as energy_vs_grid.cpp]
    double sqrt2_inv = 1.0 / sqrt(2.0);
    complex<double> M[2][2] = {
        {sqrt2_inv, sqrt2_inv},
        {-sqrt2_inv, sqrt2_inv}
    };

    complex<double> TP[4][4];

    complex<double> T1[2][2] = {
        {exp(complex<double>(0, 0.5 * gamma_param_1)), complex<double>(0.0, 0.0)},
        {complex<double>(0.0, 0.0), exp(complex<double>(0, -0.5 * gamma_param_1))}
    };
    complex<double> T2[2][2] = {
        {exp(complex<double>(0, 0.5 * gamma_param_2)), complex<double>(0.0, 0.0)},
        {complex<double>(0.0, 0.0), exp(complex<double>(0, -0.5 * gamma_param_2))}
    };

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

void writeParametersFile(const string& output_path, int seed) {
    ofstream paramFile;
    string paramPath = output_path + "binding_energy_parameters_seed=" + to_string(seed) + ".txt";
    paramFile.open(paramPath.c_str());
    
    if (!paramFile.is_open()) {
        cout << "Warning: Could not create parameters file: " << paramPath << endl;
        return;
    }
    
    paramFile << "# Binding Energy Study Parameters\n";
    paramFile << "seed=" << seed << "\n";
    paramFile << "grid_size=" << GRID_SIZE << "\n";
    paramFile << "dx=" << dx << "\n";
    paramFile << "dy=" << dy << "\n";
    paramFile << "dz=" << dz << "\n";
    paramFile << "gamma_mult_1=" << gamma_mult_1 << "\n";
    paramFile << "gamma_mult_2=" << gamma_mult_2 << "\n";
    
    paramFile << "# Separations tested\n";
    paramFile << "separations=";
    for (size_t i = 0; i < SEPARATIONS.size(); i++) {
        paramFile << SEPARATIONS[i];
        if (i < SEPARATIONS.size() - 1) paramFile << ",";
    }
    paramFile << "\n";
    
    paramFile << "# 2HDM Parameters\n";
    paramFile << "m_h=" << m_h << "\n";
    paramFile << "V_sm=" << V_sm << "\n";
    paramFile << "m_H=" << m_H << "\n";
    paramFile << "m_A=" << m_A << "\n";
    paramFile << "m_H_pm=" << m_H_pm << "\n";
    paramFile << "mixing_angle_a=" << a << "\n";
    paramFile << "mixing_angle_b=" << b << "\n";
    paramFile << "mu_1_sq=" << mu_1_sq << "\n";
    paramFile << "mu_2_sq=" << mu_2_sq << "\n";
    paramFile << "lambda_1=" << lambda_1 << "\n";
    paramFile << "lambda_2=" << lambda_2 << "\n";
    paramFile << "lambda_3=" << lambda_3 << "\n";
    paramFile << "l4_m_l5=" << l4_m_l5 << "\n";
    paramFile << "l4_p_l5=" << l4_p_l5 << "\n";
    paramFile << "v1=" << v1 << "\n";
    paramFile << "v2=" << v2 << "\n";
    
    paramFile.close();
    cout << "Parameters written to: " << paramPath << endl;
}

double calculateTotalEnergy(long long int nx, long long int ny, long long int nz,
                           double x1, double y1, double z1,
                           double x2, double y2, double z2,
                           const vector<double>& k_, const vector<double>& k_p,
                           double gamma_param_1, double gamma_param_2,
                           bool monopole_only, bool antimonopole_only,
                           int rank, int size) {
    
    long long int nPos = nx * ny * nz;
    
    // MPI domain decomposition
    long long int chunk = nPos / size;
    long long int chunkRem = nPos - size * chunk;

    long long int coreSize;
    if (rank >= chunkRem) { coreSize = chunk; }
    else { coreSize = chunk + 1; }

    long long int coreStart, coreEnd;
    if (rank < chunkRem) { 
        coreStart = rank * (chunk + 1); 
        coreEnd = (rank + 1) * (chunk + 1); 
    } else { 
        coreStart = rank * chunk + chunkRem; 
        coreEnd = (rank + 1) * chunk + chunkRem; 
    }

    long long int frontHaloSize, backHaloSize, remFront, remBack;
    remFront = coreStart % (ny * nz);
    remBack = coreEnd % (ny * nz);
    
    if (remFront == 0) {
        frontHaloSize = 2 * ny * nz;
    } else {
        frontHaloSize = 2 * ny * nz + remFront;
    }

    if (remBack == 0) {
        backHaloSize = 2 * ny * nz;
    } else {
        backHaloSize = 4 * ny * nz - remBack;
    }

    long long int dataStart = coreStart - frontHaloSize;

    // Calculate energy density
    double local_total_energy = 0.0;

    for (long long int i = frontHaloSize; i < coreSize + frontHaloSize; i++) {
        double center_fields[8], x_plus_fields[8], x_minus_fields[8];
        double y_plus_fields[8], y_minus_fields[8], z_plus_fields[8], z_minus_fields[8];
        
        long long int global_pos = i + dataStart;
        long long int i_coord = global_pos / (ny * nz);
        long long int j_coord = (global_pos / nz) % ny;
        long long int k_coord = global_pos % nz;
        
        // Generate center point fields
        calculateMonopoleFieldsAtPoint(global_pos, center_fields, nx, ny, nz, dx, dy, dz,
                                    x1, y1, z1, x2, y2, z2, k_, k_p, 
                                    gamma_param_1, gamma_param_2,
                                    monopole_only, antimonopole_only);
        
        // Generate neighbor fields with boundary checks
        if (i_coord < nx - 1) {
            calculateMonopoleFieldsAtPoint(global_pos + ny*nz, x_plus_fields, nx, ny, nz, dx, dy, dz,
                                        x1, y1, z1, x2, y2, z2, k_, k_p, 
                                        gamma_param_1, gamma_param_2,
                                        monopole_only, antimonopole_only);
        } else {
            for (int c = 0; c < 8; c++) x_plus_fields[c] = center_fields[c];
        }
        
        if (i_coord > 0) {
            calculateMonopoleFieldsAtPoint(global_pos - ny*nz, x_minus_fields, nx, ny, nz, dx, dy, dz,
                                        x1, y1, z1, x2, y2, z2, k_, k_p, 
                                        gamma_param_1, gamma_param_2,
                                        monopole_only, antimonopole_only);
        } else {
            for (int c = 0; c < 8; c++) x_minus_fields[c] = center_fields[c];
        }
        
        if (j_coord < ny - 1) {
            calculateMonopoleFieldsAtPoint(global_pos + nz, y_plus_fields, nx, ny, nz, dx, dy, dz,
                                        x1, y1, z1, x2, y2, z2, k_, k_p, 
                                        gamma_param_1, gamma_param_2,
                                        monopole_only, antimonopole_only);
        } else {
            for (int c = 0; c < 8; c++) y_plus_fields[c] = center_fields[c];
        }
        
        if (j_coord > 0) {
            calculateMonopoleFieldsAtPoint(global_pos - nz, y_minus_fields, nx, ny, nz, dx, dy, dz,
                                        x1, y1, z1, x2, y2, z2, k_, k_p, 
                                        gamma_param_1, gamma_param_2,
                                        monopole_only, antimonopole_only);
        } else {
            for (int c = 0; c < 8; c++) y_minus_fields[c] = center_fields[c];
        }
        
        if (k_coord < nz - 1) {
            calculateMonopoleFieldsAtPoint(global_pos + 1, z_plus_fields, nx, ny, nz, dx, dy, dz,
                                        x1, y1, z1, x2, y2, z2, k_, k_p, 
                                        gamma_param_1, gamma_param_2,
                                        monopole_only, antimonopole_only);
        } else {
            for (int c = 0; c < 8; c++) z_plus_fields[c] = center_fields[c];
        }
        
        if (k_coord > 0) {
            calculateMonopoleFieldsAtPoint(global_pos - 1, z_minus_fields, nx, ny, nz, dx, dy, dz,
                                        x1, y1, z1, x2, y2, z2, k_, k_p, 
                                        gamma_param_1, gamma_param_2,
                                        monopole_only, antimonopole_only);
        } else {
            for (int c = 0; c < 8; c++) z_minus_fields[c] = center_fields[c];
        }

        // Calculate kinetic energy
        double local_energy = 0.0;
        
        for (int comp = 0; comp < nb_fields; comp++) {
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

        // Add potential energy
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
    
    // Reduce to get global total energy
    double global_total_energy = 0.0;
    MPI_Reduce(&local_total_energy, &global_total_energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Add vacuum normalization
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
        cout << "=== BINDING ENERGY STUDY ===" << endl;
        cout << "Grid size: " << GRID_SIZE << "^3" << endl;
        cout << "dx: " << dx << endl;
        cout << "Gamma parameters: " << gamma_mult_1 << ", " << gamma_mult_2 << endl;
        cout << "Separations: ";
        for (auto sep : SEPARATIONS) cout << sep << " ";
        cout << endl;
        writeParametersFile(out_path, seed);
    }

    // Load monopole profile data
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

    // Create output file
    ofstream dataFile;
    if (rank == 0) {
        string dataPath = out_path + "binding_energy_data_seed=" + to_string(seed) + ".csv";
        dataFile.open(dataPath.c_str());
        dataFile << "energy_type,separation,total_energy" << endl;
        dataFile << fixed << setprecision(12);
        cout << "Created output file: " << dataPath << endl;
    }

    long long int nx = GRID_SIZE;
    long long int ny = GRID_SIZE;
    long long int nz = GRID_SIZE;

    double gamma_param_1 = gamma_mult_1 * pi;
    double gamma_param_2 = gamma_mult_2 * pi;

    // 1. Calculate monopole-only energy
    if (rank == 0) {
        cout << "\n=== Calculating MONOPOLE ONLY energy ===" << endl;
    }
    
    double x1 = 0.5 * (nx - 1);
    double y1 = 0.5 * (ny - 1);
    double z1 = 0.5 * (nz - 1);
    double x2 = x1;  // Doesn't matter for monopole-only
    double y2 = y1;
    double z2 = z1;
    
    double E_monopole = calculateTotalEnergy(nx, ny, nz, x1, y1, z1, x2, y2, z2,
                                            k_, k_p, gamma_param_1, gamma_param_2,
                                            true, false, rank, size);
    
    if (rank == 0) {
        cout << "Monopole energy: " << E_monopole << endl;
        dataFile << "monopole,0," << E_monopole << endl;
    }

    // 2. Calculate antimonopole-only energy
    if (rank == 0) {
        cout << "\n=== Calculating ANTIMONOPOLE ONLY energy ===" << endl;
    }
    
    double E_antimonopole = calculateTotalEnergy(nx, ny, nz, x1, y1, z1, x2, y2, z2,
                                                k_, k_p, gamma_param_1, gamma_param_2,
                                                false, true, rank, size);
    
    if (rank == 0) {
        cout << "Antimonopole energy: " << E_antimonopole << endl;
        dataFile << "antimonopole,0," << E_antimonopole << endl;
    }

    // 3. Calculate combined energies at different separations
    for (const auto& physical_separation : SEPARATIONS) {
        if (rank == 0) {
            cout << "\n=== Calculating COMBINED energy at separation = " << physical_separation << " ===" << endl;
        }

        double grid_separation = physical_separation / dz;

        x1 = 0.5 * (nx - 1) + monopole1_x_offset;
        y1 = 0.5 * (ny - 1) + monopole1_y_offset;
        z1 = 0.5 * (nz - 1) + 0.5 * grid_separation;

        x2 = 0.5 * (nx - 1) + monopole2_x_offset;
        y2 = 0.5 * (ny - 1) + monopole2_y_offset;
        z2 = 0.5 * (nz - 1) - 0.5 * grid_separation;

        double E_combined = calculateTotalEnergy(nx, ny, nz, x1, y1, z1, x2, y2, z2,
                                                k_, k_p, gamma_param_1, gamma_param_2,
                                                false, false, rank, size);
        
        if (rank == 0) {
            cout << "Combined energy at sep=" << physical_separation << ": " << E_combined << endl;
            dataFile << "combined," << physical_separation << "," << E_combined << endl;
        }
    }

    if (rank == 0) {
        dataFile.close();
        cout << "\n=== Binding energy study completed! ===" << endl;
    }

    MPI_Finalize();
    return 0;
}
