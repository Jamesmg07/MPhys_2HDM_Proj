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

//Simulation paramaters (adjustable):

const vector<double> separations = {0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425}; // Different separation values to test

// **NEW: Global output toggles (moved from main)**
const bool save_energy_density = false;     // Save energy density at each xyz coordinate
const bool save_separation_energy = true;  // Save total energy vs separation (1-to-1 mapping)

const long long int nx = 512; // Grid Dimensions
const long long int ny = 512;
const long long int nz = 512; // Set nz = 1 for 2D.
const long long int nPos = nx * ny * nz;

const double dx = 0.7; //Grid Spacings
const double dy = 0.7;
const double dz = 0.7;


const double gamma_mult = 0.5;

// === NEW: Toggle for two distinct gammas ===
const bool use_two_gammas = false; // Set to true for two gammas, false for original single gamma

// === NEW: Second gamma parameters (only used if use_two_gammas = true) ===
const double gamma_mult_1 = 0.5; // Used for T1
const double gamma_mult_2 = 0; // Used for T2 (only active when use_two_gammas = true)

const int seed = 73;


// Monopole/Antimonopole Configuration Parameters

// Monopole Boost Parameters
const double monopole1_vx = 0.0;  
const double monopole1_vy = 0.0;
const double monopole1_vz = 0.0;  

const double monopole2_vx = 0.0;  
const double monopole2_vy = 0.0;
const double monopole2_vz = -0.0; 

const double gamma_param = (gamma_mult * pi); 

// === NEW: Second gamma_param (only used if use_two_gammas = true) ===
const double gamma_param_1 = (gamma_mult_1 * pi);
const double gamma_param_2 = (gamma_mult_2 * pi);

// Monopole Position Parameters (in grid coordinates)
const double monopole1_x_offset = 0.0;     
const double monopole1_y_offset = 0.0;     

const double monopole2_x_offset = 0.0;     
const double monopole2_y_offset = 0.0;

// Monopole Field Profile Parameters
const double monopole_grid_spacing = 0.01; 
const double monopole_prefactor = pow(2, -1.5); 

// === MODIFIED: Conditional outTag based on use_two_gammas ===
const string outTag = use_two_gammas ? 
    ("gamma1=" + to_string(gamma_mult_1) + "pi_gamma2=" + to_string(gamma_mult_2) + "pi_nx=" + to_string(nx) + "_seed=" + to_string(seed) + "_monopole") :
    ("gamma=" + to_string(gamma_mult) + "pi_nx=" + to_string(nx) + "_seed=" + to_string(seed) + "_monopole");

// 2HDM Z_2 Symmetric Potential Set-Up:
 
// Mass and Energy Paramaters (CAN be chosen)
const long double m_h = 125;
const long double V_sm = 246;
const long double m_H = 0;
const long double m_A = 0;
const long double m_H_pm = 125;

// Scaled Mass and Energy Paramaters (NOT to be edited)
const long double M_h = m_h / m_h;
const long double v_sm = V_sm / V_sm;
const long double M_H = m_H / m_h;
const long double M_A = m_A / m_h;
const long double M_H_pm = m_H_pm / m_h;

// Mixing Angle Paramaters
const long double a = 0.25*pi; 
const long double b = 0.25*pi;

const long double s_a = sin(a);
const long double c_a = cos(a);

const long double s_b = sin(b);
const long double c_b = cos(b);
const long double t_b = tan(b);
const long double ct_b = pow(tan(b), -1);

// Dimensionless Potential Paramaters
const long double mu_1_sq = (1 / pow(M_h, 2)) * 0.5 * ((pow(M_h, 2) * pow(c_a, 2)) + (pow(M_H, 2) * pow(s_a, 2)) + ((pow(M_h, 2) - pow(M_H, 2)) * c_a * s_a * t_b)); // mu_sq paramaters scaled by 1/M_h^2 (sets length scale)
const long double mu_2_sq = (1 / pow(M_h, 2)) * 0.5 * ((pow(M_h, 2) * pow(s_a, 2)) + (pow(M_H, 2) * pow(c_a, 2)) + ((pow(M_h, 2) - pow(M_H, 2)) * c_a * s_a * ct_b));

const double lambda_1 = (pow(v_sm, 2) / pow(M_h, 2)) * (pow(M_h, 2) * pow(c_a, 2) + pow(M_H, 2) * pow(s_a, 2)) / (2 * pow(c_b, 2) * pow(v_sm, 2)); // lambda paramaters scaled by v_SM^2/M_h^2 (sets energy scale)
const double lambda_2 = (pow(v_sm, 2) / pow(M_h, 2)) * (pow(M_h, 2) * pow(s_a, 2) + pow(M_H, 2) * pow(c_a, 2)) / (2 * pow(s_b, 2) * pow(v_sm, 2));
const double lambda_3 = (pow(v_sm, 2) / pow(M_h, 2)) * ((pow(M_h, 2) - pow(M_H, 2)) * c_a * s_a + 2 * pow(M_H_pm, 2) * c_b * s_b) / (c_b * s_b * pow(v_sm, 2));
const double l4_m_l5 = (pow(v_sm, 2) / pow(M_h, 2)) * (-2 * pow(M_H_pm, 2)) / (pow(v_sm, 2));
const double l4_p_l5 = (pow(v_sm, 2) / pow(M_h, 2)) * (2 * (pow(M_A, 2) - pow(M_H_pm, 2))) / (pow(v_sm, 2));

//Doublet VEVs
const long double v1 = c_b * v_sm;
const long double v2 = s_b * v_sm;





// Damping Paramaters:
const int damped_nt = 0; // Number of time steps for which damping is imposed. Useful for random initial conditions
const double dampFac = 0; // Magnitude of damping term, unclear how strong to make this
const int ntHeld = 0; // Hold fields fixed (but effectively continue expansion) for this number of timesteps. Attempting to get the network into the scaling regime. Not sure how useful this is...
const bool expandDamp = false; // If true then the universe expands during the damping regime.

// Expansion Paramaters:
const double alpha = 0; // Factor multiplying hubble damping term for use in PRS algorithm. alpha = #dims has been claimed to give similar dynamics without changing string width. alpha = #dims - 1 is the usual factor.
const double bbeta = 0; // Scale factor^bbeta is the factor that multiplies the potential contribution to the EoMs. Standard is 2, PRS is 0.
const double scaling = 0; // Power law scaling of the scale factor wrt tau. Using conformal time so rad dom is scaling=1 while matter dom is scaling=2. scaling=0 returns a static universe.







// Beggining of Simulation:
int main(int argc, char** argv) {
    // Simulation parameters needed in main
    const string ic_type = "monopole";
    const string bc_type = "fixed";
    const int nb_fields = 8; // Number of fields in simulation
    
    const int saveFreq = 2;
    const string inp_path = "./"; // Input Directory Location - relative path
    const string out_path = "/share/centaurus_nas/jmg_temp/sep_only/"; // Data Directory Location - fixed path
    const int countRate = 20; // Increments for simulation progress status output.


    // Initialize MPI

    // Init MPI
    MPI_Init(&argc, &argv);

    // Get the rank and size
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        cout << "STEP 1: MPI initialization completed" << endl;
    }

    // Add debugging output
    if (rank == 0) {
        cout << "Starting monopole-antimonopole simulation..." << endl;
        cout << "Grid size: " << nx << "x" << ny << "x" << nz << endl;
        cout << "Total grid points: " << nPos << endl;
        cout << "Number of MPI processes: " << size << endl;
        cout << "Initial condition type: " << ic_type << endl;
        cout << "Gamma: " << gamma_mult << "pi" << endl;
    }

    if (rank == 0) {
        cout << "STEP 2: Basic parameters displayed" << endl;
    }

    long long int chunk = nPos / size;
    long long int chunkRem = nPos - size * chunk;

    long long int coreSize;
    if (rank >= chunkRem) { coreSize = chunk; }
    else { coreSize = chunk + 1; }

    if (rank == 0) {
        cout << "Core size per process: " << coreSize << endl;
        cout << "Looking for initial condition file: " << "./Data/SOR_Fields.txt" << endl;
        cout << "STEP 3: Memory allocation calculations completed" << endl;
    }


    // Calculate the position of the start of the chunk in the full array
    long long int coreStart, coreEnd;
    if (rank < chunkRem) { coreStart = rank * (chunk + 1); coreEnd = (rank + 1) * (chunk + 1); }
    else { coreStart = rank * chunk + chunkRem; coreEnd = (rank + 1) * chunk + chunkRem; }


    // Calculate the halo sizes (all data up to the previous x row at start of chunk and all data up to the next x row at the end of the chunk)
    long long int frontHaloSize, backHaloSize, nbrFrontHaloSize, nbrBackHaloSize, remFront, remBack;
    remFront = coreStart % (ny * nz);
    remBack = coreEnd % (ny * nz);
    if (remFront == 0) { // Smallest possible halo size

        frontHaloSize = 2 * ny * nz;
        nbrBackHaloSize = 2 * ny * nz;

    }
    else {

        // The two sum to 3*ny*nz rather than 2*ny*nz. This is inefficient and should be avoided if possible.

        frontHaloSize = 2 * ny * nz + remFront;
        nbrBackHaloSize = 4 * ny * nz - remFront;

    }

    if (remBack == 0) {

        backHaloSize = 2 * ny * nz;
        nbrFrontHaloSize = 2 * ny * nz;

    }
    else {

        backHaloSize = 4 * ny * nz - remBack;
        nbrFrontHaloSize = 2 * ny * nz + remBack;

    }

    // Size the array needs to be to hold the core and the two halos.
    long long int totSize = frontHaloSize + coreSize + backHaloSize;

    // Calculate the position of the start of the local array (including the haloes) in the full array. This quantity wraps around (i.e -ve numbers mean the other side of array)
    long long int dataStart = coreStart - frontHaloSize;
    long long int dataEnd = coreEnd + backHaloSize;

    // Warnings
    if (rank == 0) {

        if (size == 1) { cout << "Warning: Only one processor being used. This code is not designed for only one processor and may not work." << endl; }
        if (chunk < ny * nz) { cout << "Warning: Chunk size is less than the minimum halo size (i.e chunk neighbour data). Code currently assumes this is not the case so it probably won't work." << endl; }
        
        cout << "STEP 4: Halo size calculations and warnings completed" << endl;
    }

    // Define variables for the fields
    vector<vector<double>> fields(nb_fields, vector<double>(totSize, 0.0)); // Only need one timestep
    vector<double> fieldx(nb_fields, 0.0), fieldy(nb_fields, 0.0), fieldz(nb_fields, 0.0), fieldt(nb_fields, 0.0), fieldtt(nb_fields, 0.0), localKinEnergy(nb_fields, 0.0); // Needed for calculation of energy. nb_fields components.
    double fieldxx, fieldyy, fieldzz; // Only need them to calculate second time derivative of each field individually, thus can be reused.
    double x1, y1, z1, x2, y2, z2;
    long long int i, j, TimeStep, tNow, tPast, comp, imx, ipx, imy, ipy, imz, ipz, ipxmy, ipxmz, imxpy, ipymz, imxpz, imypz, imxx, ipxx, imyy, ipyy, imzz, ipzz, ipxpy, ipxpz, ipypz, ipxpypz;

    
    
    if (rank == 0) {
        cout << "STEP 5: Field arrays allocated" << endl;
    }

    struct timeval start, end;
    struct timeval setup_start, setup_end;
    struct timeval ic_start, ic_end;
    struct timeval evolution_start, evolution_end;
    
    if (rank == 0) { 
        gettimeofday(&start, NULL);
        gettimeofday(&setup_start, NULL);
        cout << "Starting monopole-antimonopole simulation timing..." << endl;
    }

    stringstream ss;

    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "STEP 6: MPI barrier passed, starting file operations" << endl;
    }

    //Creates Output Files if required
    string icPath = out_path + "ic.csv";
    ifstream ic(icPath.c_str());



    

    if (rank == 0) {
        cout << "STEP 7: Output files created" << endl;
    }

    // **NEW: Create separation vs energy output file (if enabled)**
    ofstream separationEnergyFile;
    if (save_separation_energy && rank == 0) {
        // === MODIFIED: Conditional filename based on use_two_gammas ===
        string sepEnergyPath;
        if (use_two_gammas) {
            sepEnergyPath = out_path + "separation_vs_energy_gamma1=" + to_string(gamma_mult_1) + 
                           "pi_gamma2=" + to_string(gamma_mult_2) + "pi_nx=" + to_string(nx) + 
                           "_seed=" + to_string(seed) + ".csv";
        } else {
            sepEnergyPath = out_path + "separation_vs_energy_gamma=" + to_string(gamma_mult) + 
                           "pi_nx=" + to_string(nx) + "_seed=" + to_string(seed) + ".csv";
        }
        separationEnergyFile.open(sepEnergyPath.c_str());
        separationEnergyFile << "separation,total_energy" << endl;
        separationEnergyFile << fixed << setprecision(8);
    }

    string fields_ic_data = inp_path + "SOR_Fields.txt";
    
    // Check if file exists
    ifstream test_file(fields_ic_data);
    if (!test_file.good()) {
        if (rank == 0) {
            cout << "ERROR: Cannot find initial condition file: " << fields_ic_data << endl;
            cout << "Make sure the SOR_Fields.txt file exists in the Data directory." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    test_file.close();

    if (rank == 0) {
        cout << "Found initial condition file, loading..." << endl;
        cout << "STEP 9b: Initial condition file validation passed" << endl;
    }
    
    // Vectors to store the values of k and k_p
    vector<double> k_;
    vector<double> k_p;

    ifstream inputFile(fields_ic_data);

    // Variables to hold the data read from each line
    double k_val, k_p_val;

    // Read data depending on the output format in the original file
    while (inputFile >> k_val >> k_p_val) {
        k_.push_back(k_val);
        k_p.push_back(k_p_val);
    }

    inputFile.close();
    //MONOPOLE POSITIONS - calculated from offsets
    // Index values (not necessarily on grid and hence not integers) of the zero coordinate.
    for (int sep_idx = 0; sep_idx < separations.size(); sep_idx++) {
        
    double current_separation = separations[sep_idx];
    
    if (rank == 0) {
        cout << "Processing separation " << current_separation << " (" << sep_idx+1 << "/" << separations.size() << ")" << endl;
    }
    
    // Update monopole positions for current separation
    x1 = 0.5 * (nx - 1) + monopole1_x_offset;
    y1 = 0.5 * (ny - 1) + monopole1_y_offset;
    z1 = 0.5 * (nz - 1) + current_separation * nz;    // Use current_separation

    x2 = 0.5 * (nx - 1) + monopole2_x_offset;
    y2 = 0.5 * (ny - 1) + monopole2_y_offset;
    z2 = 0.5 * (nz - 1) - current_separation * nz;   // Use current_separation
    
    
    
    // === MODIFIED: Conditional outTag_current based on use_two_gammas ===
    string outTag_current;
    if (use_two_gammas) {
        outTag_current = "gamma1=" + to_string(gamma_mult_1) + "pi_gamma2=" + to_string(gamma_mult_2) + 
                        "pi_nx=" + to_string(nx) + "_sep=" + to_string(current_separation) + 
                        "_seed=" + to_string(seed) + "_monopole";
    } else {
        outTag_current = "gamma=" + to_string(gamma_mult) + "pi_nx=" + to_string(nx) + 
                        "_sep=" + to_string(current_separation) + "_seed=" + to_string(seed) + "_monopole";
    }

    if (rank == 0) {
        cout << "STEP 8: Monopole positions calculated" << endl;
        gettimeofday(&setup_end, NULL);
        cout << "Setup and MPI initialization time: " << (setup_end.tv_sec - setup_start.tv_sec) + (setup_end.tv_usec - setup_start.tv_usec)/1000000.0 << "s" << endl;
        gettimeofday(&ic_start, NULL);
    }

    if (ic_type == "random") {

        if (rank == 0) {
            cout << "STEP 9a: Starting random initial conditions" << endl;
        }

        // Creates and assigns RIC for each of the 8 fields independantly.

        // Use the seed to generate the data
        mt19937 generator_1(seed);
        mt19937 generator_2(seed + 1);
        mt19937 generator_3(seed + 2);
        mt19937 generator_4(seed + 3);
        mt19937 generator_5(seed + 4);
        mt19937 generator_6(seed + 5);
        mt19937 generator_7(seed + 6);
        mt19937 generator_8(seed + 7);

        uniform_real_distribution<double> distribution(-1.0, 1.0); // Uniform distribution for the phase of the strings


        double phi1Assign;
        double phi2Assign;
        double phi3Assign;
        double phi4Assign;
        double phi5Assign;
        double phi6Assign;
        double phi7Assign;
        double phi8Assign;



        // Skip the random numbers ahead to the appropriate point.
        for (i = 0; i < coreStart; i++) {
            phi1Assign = distribution(generator_1);
            phi2Assign = distribution(generator_2);
            phi3Assign = distribution(generator_3);
            phi4Assign = distribution(generator_4);
            phi5Assign = distribution(generator_5);
            phi6Assign = distribution(generator_6);
            phi7Assign = distribution(generator_7);
            phi8Assign = distribution(generator_8);
        }



        for (i = frontHaloSize; i < coreSize + frontHaloSize; i++) {

            phi1Assign = distribution(generator_1);
            phi2Assign = distribution(generator_2);
            phi3Assign = distribution(generator_3);
            phi4Assign = distribution(generator_4);
            phi5Assign = distribution(generator_5);
            phi6Assign = distribution(generator_6);
            phi7Assign = distribution(generator_7);
            phi8Assign = distribution(generator_8);

            //Assign values to fields (random case - only one timestep)
            fields[0][i] = phi1Assign;
            fields[1][i] = phi2Assign;
            fields[2][i] = phi3Assign;
            fields[3][i] = phi4Assign;
            fields[4][i] = phi5Assign;
            fields[5][i] = phi6Assign;
            fields[6][i] = phi7Assign;
            fields[7][i] = phi8Assign;

        }


        // Now that the core data has been generated, need to communicate the haloes between processes:

        // Loop over the different fields (the nb_fields components of the vector of fields)
        for (comp = 0; comp < nb_fields; comp++) {

            MPI_Sendrecv(&fields[comp][frontHaloSize], nbrBackHaloSize, MPI_DOUBLE, (rank - 1 + size) % size, comp, // Send this
                &fields[comp][coreSize + frontHaloSize], backHaloSize, MPI_DOUBLE, (rank + 1) % size, comp, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Receive this

            MPI_Sendrecv(&fields[comp][coreSize + frontHaloSize - nbrFrontHaloSize], nbrFrontHaloSize, MPI_DOUBLE, (rank + 1) % size, comp,
                &fields[comp][0], frontHaloSize, MPI_DOUBLE, (rank - 1 + size) % size, comp, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }


    }



    else if (ic_type == "monopole") {
    
        if (rank == 0) {
            cout << "STEP 9a: Starting monopole initial conditions" << endl;
            cout << "Monopole 1 boost: vx=" << monopole1_vx << ", vy=" << monopole1_vy << ", vz=" << monopole1_vz << endl;
            cout << "Monopole 2 boost: vx=" << monopole2_vx << ", vy=" << monopole2_vy << ", vz=" << monopole2_vz << endl;

        }

        

        if (rank == 0) {
            cout << "Debug: Rank 0 entering main loop with coreSize " << coreSize 
                << " and frontHaloSize " << frontHaloSize << endl;
        }

        if (rank == 0) {
            cout << "Debug: k size = " << k_.size() << ", k_p size = " << k_p.size() << endl;
        }

        // Calculate boost parameters for both monopoles
        double v1_mag = sqrt(monopole1_vx*monopole1_vx + monopole1_vy*monopole1_vy + monopole1_vz*monopole1_vz);
        double v2_mag = sqrt(monopole2_vx*monopole2_vx + monopole2_vy*monopole2_vy + monopole2_vz*monopole2_vz);
        
        double gamma1 = (v1_mag > 1e-10) ? 1.0/sqrt(1.0 - v1_mag*v1_mag) : 1.0;
        double gamma2 = (v2_mag > 1e-10) ? 1.0/sqrt(1.0 - v2_mag*v2_mag) : 1.0;

        // Unit vectors (avoid division by zero)
        double v1_hat_x = (v1_mag > 1e-10) ? monopole1_vx / v1_mag : 0.0;
        double v1_hat_y = (v1_mag > 1e-10) ? monopole1_vy / v1_mag : 0.0;
        double v1_hat_z = (v1_mag > 1e-10) ? monopole1_vz / v1_mag : 0.0;
        
        double v2_hat_x = (v2_mag > 1e-10) ? monopole2_vx / v2_mag : 0.0;
        double v2_hat_y = (v2_mag > 1e-10) ? monopole2_vy / v2_mag : 0.0;
        double v2_hat_z = (v2_mag > 1e-10) ? monopole2_vz / v2_mag : 0.0;

        if (rank == 0) {
            cout << "Gamma factors: γ1=" << gamma1 << ", γ2=" << gamma2 << endl;
            cout << "STEP 9c: Boost parameters calculated" << endl;
        }

        // Initialize k and k_p arrays
        vector<vector<double>> k_kp(4, vector<double>(totSize, 0.0));
        vector<vector<double>> g_gp(4, vector<double>(totSize, 0.0));
            
        // Calculate fields for t=0 only (remove unnecessary time loop)
        for (i = frontHaloSize; i < coreSize + frontHaloSize; i++) {

            if (rank == 0 && i == frontHaloSize) {
                cout << "STEP 9d: Starting main monopole calculation loop" << endl;
            }

            //First monopole
            double x_1 = ( (i+dataStart)/(ny*nz) - x1 )*dx;
            double y_1 = ( ((i+dataStart)/nz)%ny - y1 )*dy;
            double z_1 = ( (i+dataStart)%nz - z1 )*dz;

            //Boost points (t=0, so no time displacement)
            double v_dot_r1 = x_1*v1_hat_x + y_1*v1_hat_y + z_1*v1_hat_z;

            double x_1_prime = x_1 + (gamma1-1)*(v_dot_r1)*v1_hat_x;
            double y_1_prime = y_1 + (gamma1-1)*(v_dot_r1)*v1_hat_y;
            double z_1_prime = z_1 + (gamma1-1)*(v_dot_r1)*v1_hat_z;

            double r_1 = sqrt(x_1_prime*x_1_prime + y_1_prime*y_1_prime + z_1_prime*z_1_prime); // Calculate r_pos
            double r_pos_1 = r_1 / monopole_grid_spacing; //Position of r in the smaller grid
            int r_c_1 = static_cast<int>(round(r_pos_1)); 
            int r_m_1 = r_c_1 - 1;
            int r_p_1 = r_c_1 + 1;

            // Debugging output to check bounds and values
            if (r_c_1 < 0) {

                cout << "Error: Index out of bounds at process " << rank 
                    << " with i=" << i << ", r_c_1=" << r_c_1 
                    << ", r_p_1=" << r_p_1 << ", x_1=" << x_1_prime << ", y_1=" << y_1_prime 
                    << ", z_1=" << z_1_prime << ", r_1=" << r_1 << ", r_pos_1=" << r_pos_1 << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Declare k_r and k_p_r here so they are accessible later
            double k_1 = 0.0;
            double k_1_p = 0.0;
            
            // Case where the grid goes out of bounds of the solution fine grid
            if (r_p_1 >= (k_.size())) {

                k_1 = 1.0;
                k_1_p = 0.0;
            
            // Case where the closest grid point is at the origin
            } else if (r_c_1 == 0) {

                // Values of k and k+ at r_value
                k_1 = ((( - (r_c_1 - r_pos_1) * k_[r_p_1] )) 
                        + ((r_p_1 - r_pos_1) * k_[r_c_1]));
                k_1_p = ((( - (r_c_1 - r_pos_1) * k_p[r_p_1] )) 
                        + ((r_p_1 - r_pos_1) * k_p[r_c_1]));
            
            // Middle points
            } else {

                k_1 = ((((r_m_1 - r_pos_1) * (r_c_1 - r_pos_1) * k_[r_p_1]) / 2) 
                    - (((r_m_1 - r_pos_1) * (r_p_1 - r_pos_1) * k_[r_c_1])) 
                    + (((r_c_1 - r_pos_1) * (r_p_1 - r_pos_1) * k_[r_m_1]) / 2));
                k_1_p = ((((r_m_1 - r_pos_1) * (r_c_1 - r_pos_1) * k_p[r_p_1]) / 2) 
                        - (((r_m_1 - r_pos_1) * (r_p_1 - r_pos_1) * k_p[r_c_1])) 
                        + (((r_c_1 - r_pos_1) * (r_p_1 - r_pos_1) * k_p[r_m_1]) / 2));

            }


            //Second monopole
            double x_2 = ( (i+dataStart)/(ny*nz) - x2 )*dx;
            double y_2 = ( ((i+dataStart)/nz)%ny - y2 )*dy;
            double z_2 = ( (i+dataStart)%nz - z2 )*dz;
            
            //Boost points (t=0, so no time displacement)
            double v_dot_r2 = x_2*v2_hat_x + y_2*v2_hat_y + z_2*v2_hat_z;

            double x_2_prime = x_2 + (gamma2-1)*(v_dot_r2)*v2_hat_x;
            double y_2_prime = y_2 + (gamma2-1)*(v_dot_r2)*v2_hat_y;
            double z_2_prime = z_2 + (gamma2-1)*(v_dot_r2)*v2_hat_z;

            double r_2 = sqrt(x_2_prime*x_2_prime + y_2_prime*y_2_prime + z_2_prime*z_2_prime); // Calculate r_pos

            double r_pos_2 = r_2 / monopole_grid_spacing; //Position of r in the smaller grid
            int r_c_2 = static_cast<int>(round(r_pos_2)); 
            int r_m_2 = r_c_2 - 1;
            int r_p_2 = r_c_2 + 1;

            // Debugging output to check bounds and values
            if (r_c_2 < 0) {

                cout << "Error: Index out of bounds at process " << rank 
                    << " with i=" << i << ", r_c_2=" << r_c_2 
                    << ", r_p_2=" << r_p_2 << ", x_2=" << x_2_prime << ", y_2=" << y_2_prime
                    << ", z_2=" << z_2_prime << ", r_2=" << r_2 << ", r_pos_2=" << r_pos_2 << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }


            // Declare k_r and k_p_r here so they are accessible later
            double k_2 = 0.0;
            double k_2_p = 0.0;
            
            // Values of k_ and k_+ at r_value

            if (r_p_2 >= (k_.size())) {

                k_2 = 1.0;
                k_2_p = 0.0;

            } else if (r_c_2 == 0) {

                // Values of k_ and k+ at r_value
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

            double g_1_p = (k_1 - k_1_p);
            double g_1 = (k_1 + k_1_p);
            double g_2_p = (k_2 - k_2_p);
            double g_2 = (k_2 + k_2_p);


            complex<double> u_1[2][2];    // Define a 2x2 matrix of complex<double> numbers named u_1                       
            complex<double> u_2[2][2];   // Define a 2x2 matrix of complex<double> numbers named u_2

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
                
                        
            } else if ( z_1_prime!= r_1 and z_2_prime != -r_2 and z_2_prime== r_2) {

                u_1[0][0] = complex<double>(0.0, 0.0);  
                u_1[0][1] = complex<double>(-1.0, 0.0); 
                u_1[1][0] = complex<double>(1.0, 0.0); 
                u_1[1][1] = complex<double>(0.0, 0.0);

                u_2[0][0] = complex<double>(0.0, 0.0);  
                u_2[0][1] = complex<double>(-1.0, 0.0); 
                u_2[1][0] = complex<double>(1.0, 0.0); 
                u_2[1][1] = complex<double>(0.0, 0.0);            

            } else {


                double cos_1;    // cos(theta_1 / 2), cos(theta_2 / 2)

                cos_1 = pow(0.5 * (1 + (z_1_prime / r_1)), 0.5);  // cos(theta_1 / 2)


                u_1[0][0] = complex<double>(cos_1, 0.0);  
                u_1[0][1] = complex<double>(- x_1_prime / (2 * r_1 * cos_1), y_1_prime / (2 * r_1 * cos_1)); 
                u_1[1][0] = complex<double>(x_1_prime / (2 * r_1 * cos_1), y_1_prime / (2 * r_1 * cos_1)); 
                u_1[1][1] = complex<double>(cos_1, 0.0);

                double sin_2;

                sin_2 = pow(0.5 * (1 - (z_2_prime / r_2)), 0.5);

                u_2[0][0] = complex<double>(- sin_2, 0.0);  
                u_2[0][1] = complex<double>(- x_2_prime / (2 * r_2 * sin_2), y_2_prime / (2 * r_2 * sin_2)); 
                u_2[1][0] = complex<double>(x_2_prime / (2 * r_2 * sin_2), y_2_prime / (2 * r_2 * sin_2)); 
                u_2[1][1] = complex<double>(- sin_2, 0.0);

            }


            // === MODIFIED: T matrix and tensor product calculation ===
            // Define matrix M
            double sqrt2_inv = 1.0 / sqrt(2.0);
            complex<double> M[2][2] = {
                {sqrt2_inv, sqrt2_inv},
                {-sqrt2_inv, sqrt2_inv}
            };

            complex<double> TP[4][4];

            if (use_two_gammas) {
                // --- Two distinct gamma case: B1 ⊗ B2 ---
                
                // Define T1 and T2 matrices with different gamma parameters
                complex<double> T1[2][2] = {
                    {exp(complex<double>(0, 0.5 * gamma_param_1)), complex<double>(0.0, 0.0)},
                    {complex<double>(0.0, 0.0), exp(complex<double>(0, -0.5 * gamma_param_1))}
                };
                complex<double> T2[2][2] = {
                    {exp(complex<double>(0, 0.5 * gamma_param_2)), complex<double>(0.0, 0.0)},
                    {complex<double>(0.0, 0.0), exp(complex<double>(0, -0.5 * gamma_param_2))}
                };

                // Compute C1 = T1 * u_2 and C2 = T2 * u_2
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

                // Compute A1 = u_1 * C1 and A2 = u_1 * C2
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

                // Compute B1 = A1 * M and B2 = A2 * M
                complex<double> B1[2][2], B2[2][2];
                for (int row = 0; row < 2; ++row) {
                    for (int col = 0; col < 2; ++row) {
                        B1[row][col] = complex<double>(0.0, 0.0);
                        B2[row][col] = complex<double>(0.0, 0.0);
                        for (int index = 0; index < 2; ++index) {
                            B1[row][col] += A1[row][index] * M[index][col];
                            B2[row][col] += A2[row][index] * M[index][col];
                        }
                    }
                }

                // Compute tensor product B1 ⊗ B2
                for (int r = 0; r < 2; ++r) {
                    for (int c = 0; c < 2; ++c) {
                        for (int x = 0; x < 2; ++x) {
                            for (int y = 0; y < 2; ++y) {
                                TP[2 * r + x][2 * c + y] = B1[r][c] * B2[x][y];
                            }
                        }
                    }
                }

            } else {
                // --- Original single gamma case: B ⊗ B ---
                
                // Define the T matrix (original)
                complex<double> T[2][2] = {
                    {exp(complex<double>(0, 0.5 * gamma_param)), complex<double>(0.0, 0.0)},
                    {complex<double>(0.0, 0.0), exp(complex<double>(0, -0.5 * gamma_param))}
                };

                // Step 1: Compute C = T * u_2
                complex<double> C[2][2];
                for (int row = 0; row < 2; ++row) {
                    for (int col = 0; col < 2; ++col) {
                        C[row][col] = complex<double>(0.0, 0.0);
                        for (int index = 0; index < 2; ++index) {
                            C[row][col] += T[row][index] * u_2[index][col];
                        }
                    }
                }

                // Step 2: Compute A = u_1 * C
                complex<double> A[2][2];
                for (int row = 0; row < 2; ++row) {
                    for (int col = 0; col < 2; ++col) {
                        A[row][col] = complex<double>(0.0, 0.0);
                        for (int index = 0; index < 2; ++index) {
                            A[row][col] += u_1[row][index] * C[index][col];
                        }
                    }
                }

                // Compute the matrix product B = A * M
                complex<double> B[2][2];
                for (int row = 0; row < 2; ++row) {
                    for (int col = 0; col < 2; ++col) {
                        B[row][col] = complex<double>(0.0, 0.0);
                        for (int index = 0; index < 2; ++index) {
                            B[row][col] += A[row][index] * M[index][col];
                        }
                    }
                }

                // Compute the tensor product B ⊗ B
                for (int r = 0; r < 2; ++r) {
                    for (int c = 0; c < 2; ++c) {
                        for (int x = 0; x < 2; ++x) {
                            for (int y = 0; y < 2; ++y) {
                                TP[2 * r + x][2 * c + y] = B[r][c] * B[x][y];
                            }
                        }
                    }
                }
            }

            // Multiply the tensor product by the prefactor
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    TP[r][c] *= monopole_prefactor;
                }
            }

            // Define the real 4-component vector
            double phi_both[4] = { (- g_1_p * g_2_p), (g_1 * g_2), (- g_1 * g_2), (g_1_p * g_2_p) };

            // Populate k_kp and g_gp
            k_kp[0][i] = k_1;
            k_kp[1][i] = k_1_p;
            k_kp[2][i] = k_2;
            k_kp[3][i] = k_2_p;

            g_gp[0][i] = phi_both[0];
            g_gp[1][i] = phi_both[1];
            g_gp[2][i] = phi_both[2];
            g_gp[3][i] = phi_both[3];

            // Define the complex vector phi
            complex<double> phi[4];

            // Initialize the phi vector components
            for (int r = 0; r < 4; ++r) {
                phi[r] = complex<double>(0.0, 0.0);
                for (int c = 0; c < 4; ++c) {
                    phi[r] += TP[r][c] * phi_both[c];
                }
            }

            // Assign the real and imaginary parts of the phi vector to fields (only one timestep)
            fields[0][i] = phi[0].real();  
            fields[1][i] = phi[0].imag();  
            fields[2][i] = phi[1].real();  
            fields[3][i] = phi[1].imag();  
            fields[4][i] = phi[2].real();  
            fields[5][i] = phi[2].imag();  
            fields[6][i] = phi[3].real();  
            fields[7][i] = phi[3].imag();
        }

        // Now that the core data has been generated, need to communicate the haloes between processes:

        // Loop over the different fields (the nb_fields components of the vector of fields)
        for (comp = 0; comp < nb_fields; comp++) {

            MPI_Sendrecv(&fields[comp][frontHaloSize], nbrBackHaloSize, MPI_DOUBLE, (rank - 1 + size) % size, comp, // Send this
                &fields[comp][coreSize + frontHaloSize], backHaloSize, MPI_DOUBLE, (rank + 1) % size, comp, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Receive this

            MPI_Sendrecv(&fields[comp][coreSize + frontHaloSize - nbrFrontHaloSize], nbrFrontHaloSize, MPI_DOUBLE, (rank + 1) % size, comp,
                &fields[comp][0], frontHaloSize, MPI_DOUBLE, (rank - 1 + size) % size, comp, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
       
    }


    gettimeofday(&end, NULL);

    if (rank == 0) { 
        cout << "Total initial data loaded/generated in: " << (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1000000.0 << "s" << endl;
        gettimeofday(&ic_end, NULL); 
        gettimeofday(&evolution_start, NULL);
        cout << "Starting field evolution..." << endl;
    }

    double fric, tau;
    double tau_scaling_bbeta;
    double totalLocalEnergy, localNDW, localADW_simple, localADW_full, localNM;
    vector<bool> isBoundaryPoint(totSize, false);
    vector<bool> isXBoundary(totSize, false);
    vector<bool> isYBoundary(totSize, false);
    vector<bool> isZBoundary(totSize, false);

    // Also move these R-value calculation variables (from inside the inner i-loop around line ~1200)
  

    // Index calculation variables (from inner loop around line ~1180)
    long long int global_pos, slice_pos, slice_base, z_pos, z_base;
    long long int base_now, base_past;

    // Field caching variables (from inner loop around line ~1220)
    double f0, f1, f2, f3, f4, f5, f6, f7;
    double phi1_sq, phi2_sq, phi1_dot_phi2, phi1_cross_phi2;
    double mu1_term, mu2_term, lambda1_term, lambda2_term, lambda3_term, l4m5_term, l4p5_term;
    double spatial_laplacians[nb_fields];
    double temporal_derivs[nb_fields];

 

    if (rank == 0) {
        cout << "Calculating energy density for separation " << current_separation << endl;
    }

    // Calculate energy density at each grid point
    vector<double> energy_density(coreSize, 0.0);
    double local_total_energy = 0.0;  // Track total energy for this separation

    for (i = frontHaloSize; i < coreSize + frontHaloSize; i++) {
        
        // **FIXED: Use the same neighbor calculations as optimized file**
        // No need to worry about periodicity with the x neighbours because halo is designed to contain them
        imx = i - ny * nz;
        ipx = i + ny * nz;

        // Cache periodic boundary calculations (same as optimized file)
        global_pos = i + dataStart;
        slice_pos = global_pos % (ny * nz);
        slice_base = (global_pos / (ny * nz)) * ny * nz - dataStart;
        
        imy = (slice_pos - nz + ny * nz) % (ny * nz) + slice_base;
        ipy = (slice_pos + nz) % (ny * nz) + slice_base;
        
        z_pos = global_pos % nz;
        z_base = (global_pos / nz) * nz - dataStart;
        
        imz = (z_pos - 1 + nz) % nz + z_base;
        ipz = (z_pos + 1) % nz + z_base;

            
            
        
        // Calculate spatial derivatives for energy density (kinetic energy)
        double local_energy = 0.0;
        
        
        // Calculate kinetic energy: |∇φ|² using central/one-sided differences
        for (comp = 0; comp < nb_fields; comp++) {
            // Get 3D coordinates for this point
            long long int i_coord = (i + dataStart) / (ny * nz);
            long long int j_coord = ((i + dataStart) / nz) % ny;
            long long int k_coord = (i + dataStart) % nz;

            double fieldx_comp, fieldy_comp, fieldz_comp;

            // X direction
            if (i_coord == 0) {
                fieldx_comp = (fields[comp][ipx] - fields[comp][i]) / dx;
            } else if (i_coord == nx - 1) {
                fieldx_comp = (fields[comp][i] - fields[comp][imx]) / dx;
            } else {
                fieldx_comp = (fields[comp][ipx] - fields[comp][imx]) / (2.0 * dx);
            }

            // Y direction
            if (j_coord == 0) {
                fieldy_comp = (fields[comp][ipy] - fields[comp][i]) / dy;
            } else if (j_coord == ny - 1) {
                fieldy_comp = (fields[comp][i] - fields[comp][imy]) / dy;
            } else {
                fieldy_comp = (fields[comp][ipy] - fields[comp][imy]) / (2.0 * dy);
            }

            // Z direction
            if (k_coord == 0) {
                fieldz_comp = (fields[comp][ipz] - fields[comp][i]) / dz;
            } else if (k_coord == nz - 1) {
                fieldz_comp = (fields[comp][i] - fields[comp][imz]) / dz;
            } else {
                fieldz_comp = (fields[comp][ipz] - fields[comp][imz]) / (2.0 * dz);
            }

            local_energy += (fieldx_comp*fieldx_comp + fieldy_comp*fieldy_comp + fieldz_comp*fieldz_comp);
        }

        
        // Add potential energy terms (matching optimized file exactly)
        f0 = fields[0][i]; f1 = fields[1][i]; f2 = fields[2][i]; f3 = fields[3][i];
        f4 = fields[4][i]; f5 = fields[5][i]; f6 = fields[6][i]; f7 = fields[7][i];
        
        phi1_sq = f0*f0 + f1*f1 + f2*f2 + f3*f3;
        phi2_sq = f4*f4 + f5*f5 + f6*f6 + f7*f7;
        phi1_dot_phi2 = f0*f4 + f1*f5 + f2*f6 + f3*f7;
        phi1_cross_phi2 = f0*f5 - f1*f4 + f2*f7 - f3*f6;
        
        local_energy += (-mu_1_sq * phi1_sq - mu_2_sq * phi2_sq +
                        lambda_1 * phi1_sq * phi1_sq +
                        lambda_2 * phi2_sq * phi2_sq +
                        lambda_3 * phi1_sq * phi2_sq +
                        l4_m_l5 * phi1_dot_phi2 * phi1_dot_phi2 +
                        l4_p_l5 * phi1_cross_phi2 * phi1_cross_phi2);

        energy_density[i - frontHaloSize] = local_energy;
        local_total_energy += local_energy * dx * dy * dz;  // Accumulate total energy
    }

    double global_total_energy = 0.0;
    MPI_Reduce(&local_total_energy, &global_total_energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // **FIXED: Output energy density with proper separation handling**
    if (rank == 0) {
        
        // **NEW: Save separation vs energy (lightweight)**
        if (save_separation_energy) 
        {
        separationEnergyFile << current_separation << "," << global_total_energy << endl;
        cout << "Separation " << current_separation << " -> Total Energy: " << global_total_energy << endl;
        }
        if (save_energy_density) {
            
            
            for (j = 1; j < size; j++) {
                int localCoreStart, localCoreSize;
                if (j < chunkRem) { 
                    localCoreStart = j * (chunk + 1); 
                    localCoreSize = chunk + 1; 
                } else { 
                    localCoreStart = j * chunk + chunkRem; 
                    localCoreSize = chunk; 
                }
                
                MPI_Recv(&energy_density[localCoreStart], localCoreSize, MPI_DOUBLE, j, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            
            
        
        
            // **NEW: Save detailed energy density (heavy, optional)**
        
            string energyDensityPath = out_path + "energy_density_" + outTag_current + ".csv";
            ofstream energyFile(energyDensityPath.c_str());
            energyFile << "x,y,z,energy_density" << endl;
            energyFile << fixed << setprecision(6);
            
            // Output energy density for each grid point
            for (j = 0; j < nPos; j++) {
                int k_coord = j % nz;
                int j_coord = (j / nz) % ny;
                int i_coord = j / (ny * nz);
                
                double x_pos = i_coord * dx;
                double y_pos = j_coord * dy;
                double z_pos = k_coord * dz;
                
                energyFile << x_pos << "," << y_pos << "," << z_pos << "," << energy_density[j] << endl;
            }
            
            energyFile.close();
            cout << "Energy density data saved for separation " << current_separation << endl;
        }
        
    } else {
        // Send energy density to rank 0 (same MPI as before)
        MPI_Send(&energy_density[0], coreSize, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD);
    }

    } // End of separation loop

    // **NEW: Close separation vs energy file**
    if (save_separation_energy && rank == 0) {
        separationEnergyFile.close();
        cout << "Separation vs energy data saved to separation_vs_energy file" << endl;
    }

    MPI_Finalize();

    return 0;
}
