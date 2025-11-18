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

const int nts = 2; // Number of time steps saved in data arrays

const long long int nx = 128; // Grid Dimensions
const long long int ny = 128;
const long long int nz = 128; // Set nz = 1 for 2D.
const long long int nPos = nx * ny * nz;

const double dx = 0.5; //Grid Spacings
const double dy = 0.5;
const double dz = 0.5;
const double dt = 0.1; //..KEEP 1 TO 5 RATIO, KEEP BELOW 0.5

// const int nt = (nx * dx / (2 * dt)); // nt required for sim to end at light crossing time is nx*dx/(2*dt)
const int nt = (nx * dx / (2 * dt));
const int R_saveTot = 10;

const int seed = 73;

const double gamma_mult = 0;
// Monopole/Antimonopole Configuration Parameters

const double offset_from_centre = 0.25; // Offset of monopole/antimonopole from centre as a fraction of box size
// * nz;  in z direction 

// Monopole Boost Parameters (add after monopole position parameters)
const double monopole1_vx = 0.5;  // Velocity components for monopole 1 (in units of c)
const double monopole1_vy = 0.0;
const double monopole1_vz = 0.0;  // Example: 0.1c boost in z direction

const double monopole2_vx = -0.5;  // Velocity components for monopole 2
const double monopole2_vy = 0.0;
const double monopole2_vz = -0.0; // Example: -0.1c boost in z direction (opposite)


const double gamma_param = (gamma_mult * pi); // Phase difference parameter




// Monopole Position Parameters (in grid coordinates)
const double monopole1_x_offset = 0.0;     // Offset from center in x
const double monopole1_y_offset = 0.0;     // Offset from center in y  
const double monopole1_z_offset = offset_from_centre * nz;    // Offset from center in z (z1 = center + 23)

const double monopole2_x_offset = 0.0;     // Offset from center in x
const double monopole2_y_offset = 0.0;     // Offset from center in y
const double monopole2_z_offset = -1 * (offset_from_centre * nz);   // Offset from center in z (z2 = center - 25)

// Monopole Field Profile Parameters
const double monopole_grid_spacing = 0.01; // Radial grid spacing for SOR_Fields.txt interpolation
const double monopole_prefactor = pow(2, -1.5); // Field normalization factor (v_sm / sqrt(2))

const int sep_saveFreq = 2;
const int R_saveFreq = int(nt / R_saveTot);


const string outTag = "gamma=" + to_string(gamma_mult) + "pi_nx=" + to_string(nx) + "_sep=" + to_string(2*offset_from_centre*nz) + "_nt=" + to_string(nt) + "_seed=" + to_string(seed) + "_monopole" ;

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





// Begginning of Simulation:
int main(int argc, char** argv) {
    // Simulation parameters needed in main
    const string ic_type = "monopole";
    const string bc_type = "fixed";
    const int nb_fields = 8; // Number of fields in simulation
    const bool calcEnergy = true; // Output Choices
    const bool wallDetect = false;
    const bool finalOut = true;
    const bool monopoleDetect = false;
    const bool makeGif = true;

    const string inp_path = "./"; // Input Directory Location - relative path
    const string out_path = "/share/centaurus_nas/jmg_temp/boost/"; // Data Directory Location - fixed path
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
        cout << "Number of timesteps: " << nt << endl;
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
    vector<vector<double>> fields(nb_fields, vector<double>(2 * totSize, 0.0)); // Define and initialize vector of field vectors.
    vector<double> fieldx(nb_fields, 0.0), fieldy(nb_fields, 0.0), fieldz(nb_fields, 0.0), fieldt(nb_fields, 0.0), fieldtt(nb_fields, 0.0), localKinEnergy(nb_fields, 0.0); // Needed for calculation of energy. nb_fields components.
    double fieldxx, fieldyy, fieldzz; // Only need them to calculate second time derivative of each field individually, thus can be reused.
    double x1, y1, z1, x2, y2, z2;
    long long int i, j, k, TimeStep, tNow, tPast, comp, imx, ipx, imy, ipy, imz, ipz, ipxmy, ipxmz, imxpy, ipymz, imxpz, imypz, imxx, ipxx, imyy, ipyy, imzz, ipzz, ipxpy, ipxpz, ipypz, ipxpypz;

    
    
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

    string finalFieldPath = out_path + "vortices_gif_finalFields" +  outTag + ".csv";
    ofstream finalFields(finalFieldPath.c_str());
    finalFields << fixed << setprecision(6); // Add this line


    string valsPerLoopPath = out_path + "energy_" +  outTag + ".csv";
    ofstream valsPerLoop(valsPerLoopPath.c_str());
    valsPerLoop << fixed << setprecision(6); // Add this line

    string monopoleNumberPath = out_path + "2m_monopoleNumber" +  outTag + ".csv";
    ofstream monopoleNumber(monopoleNumberPath.c_str());
    monopoleNumber << fixed << setprecision(6); // Add this line

    if (rank == 0) {
        cout << "STEP 7: Output files created" << endl;
    }
    //MONOPOLE POSITIONS - calculated from offsets
    // Index values (not necessarily on grid and hence not integers) of the zero coordinate.
    x1 = 0.5 * (nx - 1) + monopole1_x_offset;
    y1 = 0.5 * (ny - 1) + monopole1_y_offset;
    z1 = 0.5 * (nz - 1) + monopole1_z_offset;

    x2 = 0.5 * (nx - 1) + monopole2_x_offset;
    y2 = 0.5 * (ny - 1) + monopole2_y_offset;
    z2 = 0.5 * (nz - 1) + monopole2_z_offset;

    // Create simulation parameters file for Python analysis
    if (rank == 0) {
        string paramPath = out_path + "simulation_parameters_" + outTag + ".txt";
        ofstream paramFile(paramPath.c_str());
        
        paramFile << "# Simulation Parameters for Python Analysis" << endl;
        paramFile << "# Generated automatically by C++ simulation" << endl;
        paramFile << "nx=" << nx << endl;
        paramFile << "ny=" << ny << endl;
        paramFile << "nz=" << nz << endl;
        paramFile << "dx=" << dx << endl;
        paramFile << "dy=" << dy << endl;
        paramFile << "dz=" << dz << endl;
        paramFile << "dt=" << dt << endl;
        paramFile << "nt=" << nt << endl;
        paramFile << "gamma_mult=" << gamma_mult << endl;
        paramFile << "offset_from_centre=" << offset_from_centre << endl;
        paramFile << "monopole1_vx=" << monopole1_vx << endl;
        paramFile << "monopole1_vy=" << monopole1_vy << endl;
        paramFile << "monopole1_vz=" << monopole1_vz << endl;
        paramFile << "monopole2_vx=" << monopole2_vx << endl;
        paramFile << "monopole2_vy=" << monopole2_vy << endl;
        paramFile << "monopole2_vz=" << monopole2_vz << endl;
        paramFile << "seed=" << seed << endl;
        paramFile << "ic_type=" << ic_type << endl;
        paramFile << "sep_saveFreq=" << sep_saveFreq << endl;
        paramFile << "R_saveFreq=" << R_saveFreq << endl;
        paramFile << "outTag=" << outTag << endl;
        
        // Calculate and store monopole positions for Python
        paramFile << "# Calculated monopole positions (grid indices)" << endl;
        paramFile << "monopole1_x=" << x1 << endl;
        paramFile << "monopole1_y=" << y1 << endl;
        paramFile << "monopole1_z=" << z1 << endl;
        paramFile << "monopole2_x=" << x2 << endl;
        paramFile << "monopole2_y=" << y2 << endl;
        paramFile << "monopole2_z=" << z2 << endl;
        
        paramFile.close();
        cout << "STEP 7a: Simulation parameters file created: " << paramPath << endl;
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

            //Assign values to fields
            fields[0][i] = phi1Assign;
            fields[1][i] = phi2Assign;
            fields[2][i] = phi3Assign;
            fields[3][i] = phi4Assign;
            fields[4][i] = phi5Assign;
            fields[5][i] = phi6Assign;
            fields[6][i] = phi7Assign;
            fields[7][i] = phi8Assign;

            // Set next timestep as equal to the first
            fields[0][totSize + i] = fields[0][i];
            fields[1][totSize + i] = fields[1][i];
            fields[2][totSize + i] = fields[2][i];
            fields[3][totSize + i] = fields[3][i];
            fields[4][totSize + i] = fields[4][i];
            fields[5][totSize + i] = fields[5][i];
            fields[6][totSize + i] = fields[6][i];
            fields[7][totSize + i] = fields[7][i];

        }


        // Now that the core data has been generated, need to communicate the haloes between processes:

        // Loop over the different fields (the nb_fields components of the vector of fields)
        for (comp = 0; comp < nb_fields; comp++) {

            MPI_Sendrecv(&fields[comp][frontHaloSize], nbrBackHaloSize, MPI_DOUBLE, (rank - 1 + size) % size, comp, // Send this
                &fields[comp][coreSize + frontHaloSize], backHaloSize, MPI_DOUBLE, (rank + 1) % size, comp, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Receive this

            MPI_Sendrecv(&fields[comp][coreSize + frontHaloSize - nbrFrontHaloSize], nbrFrontHaloSize, MPI_DOUBLE, (rank + 1) % size, comp,
                &fields[comp][0], frontHaloSize, MPI_DOUBLE, (rank - 1 + size) % size, comp, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Sendrecv(&fields[comp][totSize + frontHaloSize], nbrBackHaloSize, MPI_DOUBLE, (rank - 1 + size) % size, comp + nb_fields,
                &fields[comp][totSize + coreSize + frontHaloSize], backHaloSize, MPI_DOUBLE, (rank + 1) % size, comp + nb_fields, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Sendrecv(&fields[comp][totSize + coreSize + frontHaloSize - nbrFrontHaloSize], nbrFrontHaloSize, MPI_DOUBLE, (rank + 1) % size, comp + nb_fields,
                &fields[comp][totSize], frontHaloSize, MPI_DOUBLE, (rank - 1 + size) % size, comp + nb_fields, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        }


    }


    else if (ic_type == "monopole") {

        if (rank == 0) {
            cout << "STEP 9a: Starting monopole initial conditions" << endl;
            cout << "Monopole 1 boost: vx=" << monopole1_vx << ", vy=" << monopole1_vy << ", vz=" << monopole1_vz << endl;
            cout << "Monopole 2 boost: vx=" << monopole2_vx << ", vy=" << monopole2_vy << ", vz=" << monopole2_vz << endl;
   
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
        vector<double> k;
        vector<double> k_p;

        ifstream inputFile(fields_ic_data);

        // Variables to hold the data read from each line
        double k_val, k_p_val;

        // Read data depending on the output format in the original file
        while (inputFile >> k_val >> k_p_val) {
            k.push_back(k_val);
            k_p.push_back(k_p_val);
        }

        inputFile.close();

        if (rank == 0) {
            cout << "Debug: Rank 0 entering main loop with coreSize " << coreSize 
                << " and frontHaloSize " << frontHaloSize << endl;
        }

        if (rank == 0) {
            cout << "Debug: k size = " << k.size() << ", k_p size = " << k_p.size() << endl;
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
        vector<vector<double>> k_kp(4, vector<double>(2 * totSize, 0.0));
        vector<vector<double>> g_gp(4, vector<double>(2 * totSize, 0.0));
            
        
        // Calculate fields for t=0 and t=dt
        for (int time_step = 0; time_step < 2; time_step++) {
            double t_lab = time_step * dt;  // t=0 for first step, t=dt for second step
        
            for (i = frontHaloSize; i < coreSize + frontHaloSize; i++) {

                if (rank == 0 && i == frontHaloSize) {
                    cout << "STEP 9d: Starting main monopole calculation loop" << endl;
                }

                //First monopole
                double x_1 = ( (i+dataStart)/(ny*nz) - x1 )*dx;
                double y_1 = ( ((i+dataStart)/nz)%ny - y1 )*dy;
                double z_1 = ( (i+dataStart)%nz - z1 )*dz;

                //Boost points
                double v_dot_r1 = x_1*v1_hat_x + y_1*v1_hat_y + z_1*v1_hat_z;

                double x_1_prime = x_1 + (gamma1-1)*(v_dot_r1)*v1_hat_x - gamma1*t_lab*v1_mag*v1_hat_x;
                double y_1_prime = y_1 + (gamma1-1)*(v_dot_r1)*v1_hat_y - gamma1*t_lab*v1_mag*v1_hat_y;
                double z_1_prime = z_1 + (gamma1-1)*(v_dot_r1)*v1_hat_z - gamma1*t_lab*v1_mag*v1_hat_z;

                

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
                if (r_p_1 >= (k.size())) {

                    k_1 = 1.0;
                    k_1_p = 0.0;
                
                // Case where the closest grid point is at the origin
                } else if (r_c_1 == 0) {

                    // Values of k and k+ at r_value
                    k_1 = ((( - (r_c_1 - r_pos_1) * k[r_p_1] )) 
                            + ((r_p_1 - r_pos_1) * k[r_c_1]));
                    k_1_p = ((( - (r_c_1 - r_pos_1) * k_p[r_p_1] )) 
                            + ((r_p_1 - r_pos_1) * k_p[r_c_1]));
                
                // Middle points
                } else {

                    k_1 = ((((r_m_1 - r_pos_1) * (r_c_1 - r_pos_1) * k[r_p_1]) / 2) 
                        - (((r_m_1 - r_pos_1) * (r_p_1 - r_pos_1) * k[r_c_1])) 
                        + (((r_c_1 - r_pos_1) * (r_p_1 - r_pos_1) * k[r_m_1]) / 2));
                    k_1_p = ((((r_m_1 - r_pos_1) * (r_c_1 - r_pos_1) * k_p[r_p_1]) / 2) 
                            - (((r_m_1 - r_pos_1) * (r_p_1 - r_pos_1) * k_p[r_c_1])) 
                            + (((r_c_1 - r_pos_1) * (r_p_1 - r_pos_1) * k_p[r_m_1]) / 2));

                }


                //Second monopole
                double x_2 = ( (i+dataStart)/(ny*nz) - x2 )*dx;
                double y_2 = ( ((i+dataStart)/nz)%ny - y2 )*dy;
                double z_2 = ( (i+dataStart)%nz - z2 )*dz;
                
                
                //Boost points
                double v_dot_r2 = x_2*v2_hat_x + y_2*v2_hat_y + z_2*v2_hat_z;

                double x_2_prime = x_2 + (gamma2-1)*(v_dot_r2)*v2_hat_x - gamma2*t_lab*v2_mag*v2_hat_x;
                double y_2_prime = y_2 + (gamma2-1)*(v_dot_r2)*v2_hat_y - gamma2*t_lab*v2_mag*v2_hat_y;
                double z_2_prime = z_2 + (gamma2-1)*(v_dot_r2)*v2_hat_z - gamma2*t_lab*v2_mag*v2_hat_z;

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
                
                // Values of k and k+ at r_value

                if (r_p_2 >= (k.size())) {

                    k_2 = 1.0;
                    k_2_p = 0.0;

                } else if (r_c_2 == 0) {

                    // Values of k and k+ at r_value
                    k_2 = ((( - (r_c_2 - r_pos_2) * k[r_p_2] )) 
                            + ((r_p_2 - r_pos_2) * k[r_c_2]));
                    k_2_p = ((( - (r_c_2 - r_pos_2) * k_p[r_p_2] )) 
                            + ((r_p_2 - r_pos_2) * k_p[r_c_2]));

                } else {
                    k_2 = ((((r_m_2 - r_pos_2) * (r_c_2 - r_pos_2) * k[r_p_2]) / 2) 
                        - (((r_m_2 - r_pos_2) * (r_p_2 - r_pos_2) * k[r_c_2])) 
                        + (((r_c_2 - r_pos_2) * (r_p_2 - r_pos_2) * k[r_m_2]) / 2));
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

               

                u_1[0][0] = 1;
                u_1[0][1] = 0;
                u_1[1][0] = 0;
                u_1[1][1] = 1;

                // Define the T matrix
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
                            
                            
                // Define matrix M
                double sqrt2_inv = 1.0 / sqrt(2.0);
                complex<double> M[2][2] = {
                    {sqrt2_inv, sqrt2_inv},
                    {-sqrt2_inv, sqrt2_inv}
                };

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

                complex<double> TP[4][4];

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
                
                

                // Assign the real and imaginary parts of the phi vector to fields
                fields[0][i+totSize*time_step] = phi[0].real();  
                fields[1][i+totSize*time_step] = phi[0].imag();  
                fields[2][i+totSize*time_step] = phi[1].real();  
                fields[3][i+totSize*time_step] = phi[1].imag();  
                fields[4][i+totSize*time_step] = phi[2].real();  
                fields[5][i+totSize*time_step] = phi[2].imag();  
                fields[6][i+totSize*time_step] = phi[3].real();  
                fields[7][i+totSize*time_step] = phi[3].imag();

            }
        }
        if (rank == 0) {
            cout << "STEP 9e: Main monopole calculation loop completed" << endl;
        }

        // Now that the core data has been generated, need to communicate the haloes between processes:

        // Loop over the different fields (the nb_fields components of the vector of fields)
        for (comp = 0; comp < nb_fields; comp++) {

            MPI_Sendrecv(&fields[comp][frontHaloSize], nbrBackHaloSize, MPI_DOUBLE, (rank - 1 + size) % size, comp, // Send this
                &fields[comp][coreSize + frontHaloSize], backHaloSize, MPI_DOUBLE, (rank + 1) % size, comp, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Receive this

            MPI_Sendrecv(&fields[comp][coreSize + frontHaloSize - nbrFrontHaloSize], nbrFrontHaloSize, MPI_DOUBLE, (rank + 1) % size, comp,
                &fields[comp][0], frontHaloSize, MPI_DOUBLE, (rank - 1 + size) % size, comp, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Sendrecv(&fields[comp][totSize + frontHaloSize], nbrBackHaloSize, MPI_DOUBLE, (rank - 1 + size) % size, comp + nb_fields,
                &fields[comp][totSize + coreSize + frontHaloSize], backHaloSize, MPI_DOUBLE, (rank + 1) % size, comp + nb_fields, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Sendrecv(&fields[comp][totSize + coreSize + frontHaloSize - nbrFrontHaloSize], nbrFrontHaloSize, MPI_DOUBLE, (rank + 1) % size, comp + nb_fields,
                &fields[comp][totSize], frontHaloSize, MPI_DOUBLE, (rank - 1 + size) % size, comp + nb_fields, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        }

        if (rank == 0) {
            
            string test_kValuesPath = out_path + "test_m_am_kValues" +  outTag + ".csv";
            string test_gValuesPath = out_path + "test_m_am_gValues" +  outTag + ".csv";
            
            ofstream test_kValuesFile(test_kValuesPath.c_str());
            ofstream test_gValuesFile(test_gValuesPath.c_str());

            test_kValuesFile << "k1,k1p,k2,k2p\n";
            test_gValuesFile << "g1p_g2p_neg,g1_g2,g1_g2_neg,g1p_g2p\n";
            test_kValuesFile << fixed << setprecision(6);
            test_gValuesFile << fixed << setprecision(6);
            

            vector<vector<double>> kOut(4, vector<double>(nPos, 0.0));
            vector<vector<double>> gOut(4, vector<double>(nPos, 0.0));
            int localCoreStartnt;
            int localCoreSizent;

            // Gather field data from all processes
            for (comp = 0; comp < 4; comp++) {

                for (j = 0; j < coreSize; j++) { 
                    
                    kOut[comp][j] = k_kp[comp][frontHaloSize + j]; 
                    gOut[comp][j] = g_gp[comp][frontHaloSize + j]; 
                    
                }

                for (j = 1; j < size; j++) {

                    
                    if (j < chunkRem) { localCoreStartnt = j * (chunk + 1); localCoreSizent = chunk + 1; }
                    else { localCoreStartnt = j * chunk + chunkRem; localCoreSizent = chunk; }

                    MPI_Recv(&kOut[comp][localCoreStartnt], localCoreSizent, MPI_DOUBLE, j, comp, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&gOut[comp][localCoreStartnt], localCoreSizent, MPI_DOUBLE, j, comp, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }

            // Output fields and R values to separate files
            for (j = 0; j < nPos; j++) {


                // Write R values to R values file, ensuring explicit output of 0.0
                test_kValuesFile << kOut[0][j] << "," << kOut[1][j] << "," << kOut[2][j] << "," << kOut[3][j] << "\n";
                test_gValuesFile << gOut[0][j] << "," << gOut[1][j] << "," << gOut[2][j] << "," << gOut[3][j] << "\n";
                
            }

            test_kValuesFile.close();
            test_gValuesFile.close();
        }

        else {
            // Send field data to rank 0
            for (comp = 0; comp < 4; comp++) {
                MPI_Send(&k_kp[comp][frontHaloSize], coreSize, MPI_DOUBLE, 0, comp, MPI_COMM_WORLD);
                MPI_Send(&g_gp[comp][frontHaloSize], coreSize, MPI_DOUBLE, 0, comp, MPI_COMM_WORLD);
            }
        }

        if (rank == 0) {
            // Create files for fields and R values
            string test_fieldsPath = out_path + "test_m_am_fieldValues" +  outTag + ".csv";
            string test_rValuesPath = out_path + "test_m_am_RValues" +  outTag + ".csv";
            
            ofstream test_fieldsFile(test_fieldsPath.c_str());
            ofstream test_rValuesFile(test_rValuesPath.c_str());

            // Headers for fields and R values
            test_fieldsFile << "field0,field1,field2,field3,field4,field5,field6,field7\n";
            test_rValuesFile << "R0nt,R1nt,R2nt,R3nt\n";
            // Set precision once
            test_fieldsFile << fixed << setprecision(6);
            test_rValuesFile << fixed << setprecision(6);

            vector<vector<double>> fieldsOutnt(nb_fields, vector<double>(nPos, 0.0));
            double R0nt, R1nt, R2nt, R3nt;
            int localCoreStartnt;
            int localCoreSizent;

            // Gather field data from all processes
            for (comp = 0; comp < nb_fields; comp++) {

                for (j = 0; j < coreSize; j++) { fieldsOutnt[comp][j] = fields[comp][frontHaloSize + j]; }

                for (j = 1; j < size; j++) {

                    
                    if (j < chunkRem) { localCoreStartnt = j * (chunk + 1); localCoreSizent = chunk + 1; }
                    else { localCoreStartnt = j * chunk + chunkRem; localCoreSizent = chunk; }

                    MPI_Recv(&fieldsOutnt[comp][localCoreStartnt], localCoreSizent, MPI_DOUBLE, j, comp, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }

            // Output fields and R values to separate files
            for (j = 0; j < nPos; j++) {

                // Compute R values
                R1nt = 2 * (fieldsOutnt[0][j] * fieldsOutnt[4][j] + fieldsOutnt[1][j] * fieldsOutnt[5][j] + fieldsOutnt[2][j] * fieldsOutnt[6][j] + fieldsOutnt[3][j] * fieldsOutnt[7][j]);
                R2nt = 2 * (fieldsOutnt[0][j] * fieldsOutnt[5][j] + fieldsOutnt[2][j] * fieldsOutnt[7][j] - fieldsOutnt[1][j] * fieldsOutnt[4][j] - fieldsOutnt[3][j] * fieldsOutnt[6][j]);
                R3nt = pow(fieldsOutnt[0][j], 2) + pow(fieldsOutnt[1][j], 2) + pow(fieldsOutnt[2][j], 2) + pow(fieldsOutnt[3][j], 2) - pow(fieldsOutnt[4][j], 2) - pow(fieldsOutnt[5][j], 2) - pow(fieldsOutnt[6][j], 2) - pow(fieldsOutnt[7][j], 2);
                R0nt = pow(fieldsOutnt[0][j], 2) + pow(fieldsOutnt[1][j], 2) + pow(fieldsOutnt[2][j], 2) + pow(fieldsOutnt[3][j], 2) + pow(fieldsOutnt[4][j], 2) + pow(fieldsOutnt[5][j], 2) + pow(fieldsOutnt[6][j], 2) + pow(fieldsOutnt[7][j], 2);
                

                // Write field values to fields file, ensuring explicit output of 0.0
                test_fieldsFile << fieldsOutnt[0][j] << "," << fieldsOutnt[1][j] << "," << fieldsOutnt[2][j] << "," 
                                << fieldsOutnt[3][j] << "," << fieldsOutnt[4][j] << "," << fieldsOutnt[5][j] << "," 
                                << fieldsOutnt[6][j] << "," << fieldsOutnt[7][j] << "\n";
                test_rValuesFile << R0nt << "," << R1nt << "," << R2nt << "," << R3nt << "\n";
            }


            test_fieldsFile.close();
            test_rValuesFile.close();
        }

        else {
            // Send field data to rank 0
            for (comp = 0; comp < nb_fields; comp++) {
                MPI_Send(&fields[comp][frontHaloSize], coreSize, MPI_DOUBLE, 0, comp, MPI_COMM_WORLD);
            }
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
    double R1_i, R1_ipx, R1_ipy, R1_ipz, R1_imx, R1_imy, R1_imz;
    double R2_i, R2_ipx, R2_ipy, R2_ipz, R2_ipxpy, R2_ipxpz, R2_ipypz, R2_ipxpypz;
    double R3_i, R3_ipx, R3_ipy, R3_ipz, R3_ipxpy, R3_ipxpz, R3_ipypz, R3_ipxpypz;
    double R1x, R1y, R1z;

    // Index calculation variables (from inner loop around line ~1180)
    long long int global_pos, slice_pos, slice_base, z_pos, z_base;
    long long int base_now, base_past;

    // Field caching variables (from inner loop around line ~1220)
    double f0, f1, f2, f3, f4, f5, f6, f7;
    double phi1_sq, phi2_sq, phi1_dot_phi2, phi1_cross_phi2;
    double mu1_term, mu2_term, lambda1_term, lambda2_term, lambda3_term, l4m5_term, l4p5_term;
    double spatial_laplacians[nb_fields];
    double temporal_derivs[nb_fields];
    double l4m5_coeffs[8], l4p5_coeffs[8], mu_terms[8], lambda_terms[8], lambda3_phi_sq[8];

    double R1nt, R2nt, R3nt;
    int localCoreStartnt, localCoreSizent;
    int k1, j1, i1;
    int k_j, j_j, i_j;
    double distance_squared;
    double dx_diff, dy_diff, dz_diff;

    
    double min1_value, min2_value;             
    int min1_idx, min2_idx;                  
    double max_near_min1; 



    // Main for loop that evolves the fields:
    for (TimeStep = 0; TimeStep < nt; TimeStep++) {

        if (rank == 0 && TimeStep == 0) {
            cout << "STEP 11: Starting main evolution loop" << endl;
        }

        

        // Expansion during damping check:
        if (expandDamp) { tau = 1 + (ntHeld + TimeStep) * dt; }
        else { tau = 1 + (ntHeld + TimeStep - damped_nt) * dt; }

        // Is damping switched on or not?
        if (TimeStep < damped_nt) {

            if (expandDamp) { fric = dampFac + alpha * scaling / tau; } // denominator is conformal time
            else { fric = dampFac; }
        }
        else {

            if (expandDamp) { fric = alpha * scaling / tau; } // Time needs to have moved along during the damped phase
            else { fric = alpha * scaling / tau; } // Time was not progressing during the damped phase

        }

        tNow = (TimeStep + 1) % 2;
        tPast = TimeStep % 2;

        // Pre-calculate scaling factor to avoid repeated computation
        tau_scaling_bbeta = pow(pow(tau, scaling), bbeta);

        if (rank == 0 && TimeStep == 0) {
            cout << "STEP 12: Damping and expansion parameters calculated" << endl;
        }

        // Main calculations and evolutions section, using the EoM:
        totalLocalEnergy = 0;
        for (comp = 0; comp < nb_fields; comp++) { localKinEnergy[comp] = 0; }
        localNDW = 0;
        localADW_simple = 0;
        localADW_full = 0;
        localNM = 0;

   
        
        // Precompute boundary status for all points in this process's domain
        if (bc_type == "fixed") {
            for (i = frontHaloSize; i < coreSize + frontHaloSize; i++) {
                double i_coord = (i + dataStart) / (ny * nz);
                double j_coord = (((i + dataStart) / nz) % ny);
                double k_coord = ((i + dataStart) % nz);
                
                bool isOnBoundary = (i_coord == 0 || i_coord == nx-1 || 
                                   j_coord == 0 || j_coord == ny-1 || 
                                   k_coord == 0 || k_coord == nz-1);
                
                isBoundaryPoint[i] = isOnBoundary;
                isXBoundary[i] = (i_coord == 1 || i_coord == nx-2);
                isYBoundary[i] = (j_coord == 1 || j_coord == ny-2);
                isZBoundary[i] = (k_coord == 1 || k_coord == nz-2);
            }
        }

        //Loops over and evolves all assigned data points
        for (i = frontHaloSize; i < coreSize + frontHaloSize; i++) { 
            
            // Skip boundary points if using fixed BCs
            if (bc_type == "fixed" && isBoundaryPoint[i]) {
                continue;
            }
            
            //Encoding the periodic boundary conditions for the evolving data point:
            
            // No need to worry about periodicity with the x neighbours because halo is designed to contain them
            imx = i - ny * nz;
            ipx = i + ny * nz;
            imxx = i - 2 * ny * nz;
            ipxx = i + 2 * ny * nz;

            // Cache periodic boundary calculations
            global_pos = i + dataStart;
            slice_pos = global_pos % (ny * nz);
            slice_base = (global_pos / (ny * nz)) * ny * nz - dataStart;
            
            imy = (slice_pos - nz + ny * nz) % (ny * nz) + slice_base;
            ipy = (slice_pos + nz) % (ny * nz) + slice_base;
            imyy = (slice_pos - 2 * nz + ny * nz) % (ny * nz) + slice_base;
            ipyy = (slice_pos + 2 * nz) % (ny * nz) + slice_base;
            
            z_pos = global_pos % nz;
            z_base = (global_pos / nz) * nz - dataStart;
            
            imz = (z_pos - 1 + nz) % nz + z_base;
            ipz = (z_pos + 1) % nz + z_base;
            imzz = (z_pos - 2 + nz) % nz + z_base;
            ipzz = (z_pos + 2) % nz + z_base;

            // Additionally needed for wilson loop calculations. Avoid using x shifted points first as this makes the calculations more complicated and some of these points aren't in the correct positions
            ipxmy = imy + ny * nz;
            ipxmz = imz + ny * nz;
            imxpy = ipy - ny * nz;
            ipymz = (ipy + dataStart - 1 + nz) % nz + ((ipy + dataStart) / nz) * nz - dataStart;
            imxpz = ipz - ny * nz;
            imypz = (imy + dataStart + 1) % nz + ((imy + dataStart) / nz) * nz - dataStart;
            ipxpy = (ipy + ny * nz);
            ipxpz = (ipz + ny * nz);
            ipypz = (ipy + dataStart + 1) % nz + ((ipy + dataStart) / nz) * nz - dataStart;
            ipxpypz = ipypz + ny * nz;

            // More efficient field caching - access memory sequentially
            base_now = totSize * tNow;
            base_past = totSize * tPast;
            
            f0 = fields[0][base_now + i];
            f1 = fields[1][base_now + i];
            f2 = fields[2][base_now + i];
            f3 = fields[3][base_now + i];
            f4 = fields[4][base_now + i];
            f5 = fields[5][base_now + i];
            f6 = fields[6][base_now + i];
            f7 = fields[7][base_now + i];

            // Pre-calculate all field combinations once
            phi1_sq = f0*f0 + f1*f1 + f2*f2 + f3*f3;
            phi2_sq = f4*f4 + f5*f5 + f6*f6 + f7*f7;
            phi1_dot_phi2 = f0*f4 + f1*f5 + f2*f6 + f3*f7;
            phi1_cross_phi2 = f0*f5 - f1*f4 + f2*f7 - f3*f6;

            // Pre-calculate potential terms once
            mu1_term = tau_scaling_bbeta * mu_1_sq;
            mu2_term = tau_scaling_bbeta * mu_2_sq;
            lambda1_term = tau_scaling_bbeta * 2 * lambda_1 * phi1_sq;
            lambda2_term = tau_scaling_bbeta * 2 * lambda_2 * phi2_sq;
            lambda3_term = tau_scaling_bbeta * lambda_3;
            l4m5_term = tau_scaling_bbeta * l4_m_l5;
            l4p5_term = tau_scaling_bbeta * l4_p_l5;

            // Calculate spatial derivatives for all fields with boundary handling
            
            for (comp = 0; comp < nb_fields; comp++) {
                double fieldxx, fieldyy, fieldzz;
                
                // Optimized spatial derivatives with boundary handling
                if (bc_type == "fixed" && isXBoundary[i]) {
                    fieldxx = (fields[comp][base_now + ipx] - 2 * fields[comp][base_now + i] + fields[comp][base_now + imx]) / (dx * dx);
                } else {
                    fieldxx = (16 * (fields[comp][base_now + ipx] + fields[comp][base_now + imx]) - 
                              30 * fields[comp][base_now + i] - 
                              fields[comp][base_now + ipxx] - fields[comp][base_now + imxx]) / (12 * dx * dx);
                }
                
                // Y second derivatives with boundary conditions
                if (bc_type == "fixed" && isYBoundary[i]) {
                    fieldyy = (fields[comp][base_now + ipy] - 2 * fields[comp][base_now + i] + fields[comp][base_now + imy]) / (dy * dy);
                } else {
                    fieldyy = (16 * (fields[comp][base_now + ipy] + fields[comp][base_now + imy]) - 
                              30 * fields[comp][base_now + i] - 
                              fields[comp][base_now + ipyy] - fields[comp][base_now + imyy]) / (12 * dy * dy);
                }
                
                // Z second derivatives with boundary conditions
                if (bc_type == "fixed" && isZBoundary[i]) {
                    fieldzz = (fields[comp][base_now + ipz] - 2 * fields[comp][base_now + i] + fields[comp][base_now + imz]) / (dz * dz);
                } else {
                    fieldzz = (16 * (fields[comp][base_now + ipz] + fields[comp][base_now + imz]) - 
                              30 * fields[comp][base_now + i] - 
                              fields[comp][base_now + ipzz] - fields[comp][base_now + imzz]) / (12 * dz * dz);
                }
                
                // Laplacian is the sum of second derivatives
                spatial_laplacians[comp] = fieldxx + fieldyy + fieldzz;
                temporal_derivs[comp] = (fields[comp][base_now + i] - fields[comp][base_past + i]) / dt;
            }

            // Optimized evolution for all fields using lookup tables
            l4m5_coeffs[0] = f4; l4m5_coeffs[1] = f5; l4m5_coeffs[2] = f6; l4m5_coeffs[3] = f7;
            l4m5_coeffs[4] = f0; l4m5_coeffs[5] = f1; l4m5_coeffs[6] = f2; l4m5_coeffs[7] = f3;
            
            l4p5_coeffs[0] = f5; l4p5_coeffs[1] = -f4; l4p5_coeffs[2] = f7; l4p5_coeffs[3] = -f6;
            l4p5_coeffs[4] = -f1; l4p5_coeffs[5] = f0; l4p5_coeffs[6] = -f3; l4p5_coeffs[7] = f2;
            
            mu_terms[0] = mu1_term; mu_terms[1] = mu1_term; mu_terms[2] = mu1_term; mu_terms[3] = mu1_term;
            mu_terms[4] = mu2_term; mu_terms[5] = mu2_term; mu_terms[6] = mu2_term; mu_terms[7] = mu2_term;
            
            lambda_terms[0] = lambda1_term; lambda_terms[1] = lambda1_term; lambda_terms[2] = lambda1_term; lambda_terms[3] = lambda1_term;
            lambda_terms[4] = lambda2_term; lambda_terms[5] = lambda2_term; lambda_terms[6] = lambda2_term; lambda_terms[7] = lambda2_term;
            
            lambda3_phi_sq[0] = phi2_sq; lambda3_phi_sq[1] = phi2_sq; lambda3_phi_sq[2] = phi2_sq; lambda3_phi_sq[3] = phi2_sq;
            lambda3_phi_sq[4] = phi1_sq; lambda3_phi_sq[5] = phi1_sq; lambda3_phi_sq[6] = phi1_sq; lambda3_phi_sq[7] = phi1_sq;

            
            // Vectorized field evolution
            for (comp = 0; comp < nb_fields; comp++) {
                double fieldtt_comp = spatial_laplacians[comp] - fric * temporal_derivs[comp] + 
                                     mu_terms[comp] * fields[comp][base_now + i] -
                                     lambda_terms[comp] * fields[comp][base_now + i] -
                                     lambda3_term * fields[comp][base_now + i] * lambda3_phi_sq[comp] -
                                     l4m5_term * l4m5_coeffs[comp] * phi1_dot_phi2 -
                                     l4p5_term * l4p5_coeffs[comp] * phi1_cross_phi2;

                fields[comp][base_past + i] = 2 * fields[comp][base_now + i] - 
                                             fields[comp][base_past + i] + 
                                             dt * dt * fieldtt_comp;
            }

            // Optimized energy calculation
            if (calcEnergy) {
                // Calculate spatial derivatives for energy once
                for (comp = 0; comp < nb_fields; comp++) {
                    double fieldx_comp = (fields[comp][base_now + i] - fields[comp][base_now + imx]) / dx;
                    double fieldy_comp = (fields[comp][base_now + i] - fields[comp][base_now + imy]) / dy;
                    double fieldz_comp = (fields[comp][base_now + i] - fields[comp][base_now + imz]) / dz;

                    totalLocalEnergy += (fieldx_comp*fieldx_comp + fieldy_comp*fieldy_comp + fieldz_comp*fieldz_comp) * dx * dy * dz;
                }

                // Potential energy terms using pre-computed values
                totalLocalEnergy += (-mu_1_sq * phi1_sq - mu_2_sq * phi2_sq +
                                    lambda_1 * phi1_sq * phi1_sq +
                                    lambda_2 * phi2_sq * phi2_sq +
                                    lambda_3 * phi1_sq * phi2_sq +
                                    l4_m_l5 * phi1_dot_phi2 * phi1_dot_phi2 +
                                    l4_p_l5 * phi1_cross_phi2 * phi1_cross_phi2) * dx * dy * dz;
            }

            // Wall detection using pre-computed R1 value (optimized like monopole code)
            if (wallDetect) {
                R1_i = 2 * phi1_dot_phi2;
                R1_ipx = 2 * (fields[0][totSize * tNow + ipx] * fields[4][totSize * tNow + ipx] + fields[1][totSize * tNow + ipx] * fields[5][totSize * tNow + ipx] + fields[2][totSize * tNow + ipx] * fields[6][totSize * tNow + ipx] + fields[3][totSize * tNow + ipx] * fields[7][totSize * tNow + ipx]);
                R1_ipy = 2 * (fields[0][totSize * tNow + ipy] * fields[4][totSize * tNow + ipy] + fields[1][totSize * tNow + ipy] * fields[5][totSize * tNow + ipy] + fields[2][totSize * tNow + ipy] * fields[6][totSize * tNow + ipy] + fields[3][totSize * tNow + ipy] * fields[7][totSize * tNow + ipy]);
                double R1_ipz = 2 * (fields[0][totSize * tNow + ipz] * fields[4][totSize * tNow + ipz] + fields[1][totSize * tNow + ipz] * fields[5][totSize * tNow + ipz] + fields[2][totSize * tNow + ipz] * fields[6][totSize * tNow + ipz] + fields[3][totSize * tNow + ipz] * fields[7][totSize * tNow + ipz]);

                R1_imx = 2 * (fields[0][totSize * tNow + imx] * fields[4][totSize * tNow + imx] + fields[1][totSize * tNow + imx] * fields[5][totSize * tNow + imx] + fields[2][totSize * tNow + imx] * fields[6][totSize * tNow + imx] + fields[3][totSize * tNow + imx] * fields[7][totSize * tNow + imx]);
                R1_imy = 2 * (fields[0][totSize * tNow + imy] * fields[4][totSize * tNow + imy] + fields[1][totSize * tNow + imy] * fields[5][totSize * tNow + imy] + fields[2][totSize * tNow + imy] * fields[6][totSize * tNow + imy] + fields[3][totSize * tNow + imy] * fields[7][totSize * tNow + imy]);
                R1_imz = 2 * (fields[0][totSize * tNow + imz] * fields[4][totSize * tNow + imz] + fields[1][totSize * tNow + imz] * fields[5][totSize * tNow + imz] + fields[2][totSize * tNow + imz] * fields[6][totSize * tNow + imz] + fields[3][totSize * tNow + imz] * fields[7][totSize * tNow + imz]);


                // x neighbour
                if (R1_i * R1_ipx < 0) {

                    localNDW += 1;
                    localADW_simple += 2.0 * dy * dz / 3.0;

                    R1x = (R1_i - R1_imx) / dx;
                    R1y = (R1_i - R1_imy) / dy;
                    R1z = (R1_i - R1_imz) / dz;
                    localADW_full += dy * dz * sqrt(pow(R1x, 2) + pow(R1y, 2) + pow(R1z, 2)) / (abs(R1x) + abs(R1y) + abs(R1z));
                }

                // y neighbour
                if (R1_i * R1_ipy < 0) {

                    localNDW += 1;
                    localADW_simple += 2.0 * dy * dz / 3.0;

                    R1x = (R1_i - R1_imx) / dx;
                    R1y = (R1_i - R1_imy) / dy;
                    R1z = (R1_i - R1_imz) / dz;
                    localADW_full += dx * dz * sqrt(pow(R1x, 2) + pow(R1y, 2) + pow(R1z, 2)) / (abs(R1x) + abs(R1y) + abs(R1z));
                }

                // z neighbour
                if (R1_i * R1_ipz < 0) {

                    localNDW += 1;
                    localADW_simple += 2.0 * dy * dz / 3.0;

                    R1x = (R1_i - R1_imx) / dx;
                    R1y = (R1_i - R1_imy) / dy;
                    R1z = (R1_i - R1_imz) / dz;
                    localADW_full += dx * dy * sqrt(pow(R1x, 2) + pow(R1y, 2) + pow(R1z, 2)) / (abs(R1x) + abs(R1y) + abs(R1z));
                }
            }

            if (monopoleDetect) {

                R1_i = 2 * (fields[0][totSize * tNow + i] * fields[4][totSize * tNow + i] + fields[1][totSize * tNow + i] * fields[5][totSize * tNow + i] + fields[2][totSize * tNow + i] * fields[6][totSize * tNow + i] + fields[3][totSize * tNow + i] * fields[7][totSize * tNow + i]);
                R1_ipx = 2 * (fields[0][totSize * tNow + ipx] * fields[4][totSize * tNow + ipx] + fields[1][totSize * tNow + ipx] * fields[5][totSize * tNow + ipx] + fields[2][totSize * tNow + ipx] * fields[6][totSize * tNow + ipx] + fields[3][totSize * tNow + ipx] * fields[7][totSize * tNow + ipx]);
                R1_ipy = 2 * (fields[0][totSize * tNow + ipy] * fields[4][totSize * tNow + ipy] + fields[1][totSize * tNow + ipy] * fields[5][totSize * tNow + ipy] + fields[2][totSize * tNow + ipy] * fields[6][totSize * tNow + ipy] + fields[3][totSize * tNow + ipy] * fields[7][totSize * tNow + ipy]);
                R1_ipz = 2 * (fields[0][totSize * tNow + ipz] * fields[4][totSize * tNow + ipz] + fields[1][totSize * tNow + ipz] * fields[5][totSize * tNow + ipz] + fields[2][totSize * tNow + ipz] * fields[6][totSize * tNow + ipz] + fields[3][totSize * tNow + ipz] * fields[7][totSize * tNow + ipz]);
                double R1_ipxpy = 2 * (fields[0][totSize * tNow + ipxpy] * fields[4][totSize * tNow + ipxpy] + fields[1][totSize * tNow + ipxpy] * fields[5][totSize * tNow + ipxpy] + fields[2][totSize * tNow + ipxpy] * fields[6][totSize * tNow + ipxpy] + fields[3][totSize * tNow + ipxpy] * fields[7][totSize * tNow + ipxpy]);
                double R1_ipxpz = 2 * (fields[0][totSize * tNow + ipxpz] * fields[4][totSize * tNow + ipxpz] + fields[1][totSize * tNow + ipxpz] * fields[5][totSize * tNow + ipxpz] + fields[2][totSize * tNow + ipxpz] * fields[6][totSize * tNow + ipxpz] + fields[3][totSize * tNow + ipxpz] * fields[7][totSize * tNow + ipxpz]);
                double R1_ipypz = 2 * (fields[0][totSize * tNow + ipypz] * fields[4][totSize * tNow + ipypz] + fields[1][totSize * tNow + ipypz] * fields[5][totSize * tNow + ipypz] + fields[2][totSize * tNow + ipypz] * fields[6][totSize * tNow + ipypz] + fields[3][totSize * tNow + ipypz] * fields[7][totSize * tNow + ipypz]);
                double R1_ipxpypz = 2 * (fields[0][totSize * tNow + ipxpypz] * fields[4][totSize * tNow + ipxpypz] + fields[1][totSize * tNow + ipxpypz] * fields[5][totSize * tNow + ipxpypz] + fields[2][totSize * tNow + ipxpypz] * fields[6][totSize * tNow + ipxpypz] + fields[3][totSize * tNow + ipxpypz] * fields[7][totSize * tNow + ipxpypz]);

                R2_i = 2 * (fields[0][totSize * tNow + i] * fields[5][totSize * tNow + i] + fields[2][totSize * tNow + i] * fields[7][totSize * tNow + i] - fields[1][totSize * tNow + i] * fields[4][totSize * tNow + i] - fields[3][totSize * tNow + i] * fields[6][totSize * tNow + i]);
                R2_ipx = 2 * (fields[0][totSize * tNow + ipx] * fields[5][totSize * tNow + ipx] + fields[2][totSize * tNow + ipx] * fields[7][totSize * tNow + ipx] - fields[1][totSize * tNow + ipx] * fields[4][totSize * tNow + ipx] - fields[3][totSize * tNow + ipx] * fields[6][totSize * tNow + ipx]);
                R2_ipy = 2 * (fields[0][totSize * tNow + ipy] * fields[5][totSize * tNow + ipy] + fields[2][totSize * tNow + ipy] * fields[7][totSize * tNow + ipy] - fields[1][totSize * tNow + ipy] * fields[4][totSize * tNow + ipy] - fields[3][totSize * tNow + ipy] * fields[6][totSize * tNow + ipy]);
                R2_ipz = 2 * (fields[0][totSize * tNow + ipz] * fields[5][totSize * tNow + ipz] + fields[2][totSize * tNow + ipz] * fields[7][totSize * tNow + ipz] - fields[1][totSize * tNow + ipz] * fields[4][totSize * tNow + ipz] - fields[3][totSize * tNow + ipz] * fields[6][totSize * tNow + ipz]);
                R2_ipxpy = 2 * (fields[0][totSize * tNow + ipxpy] * fields[5][totSize * tNow + ipxpy] + fields[2][totSize * tNow + ipxpy] * fields[7][totSize * tNow + ipxpy] - fields[1][totSize * tNow + ipxpy] * fields[4][totSize * tNow + ipxpy] - fields[3][totSize * tNow + ipxpy] * fields[6][totSize * tNow + ipxpy]);
                R2_ipxpz = 2 * (fields[0][totSize * tNow + ipxpz] * fields[5][totSize * tNow + ipxpz] + fields[2][totSize * tNow + ipxpz] * fields[7][totSize * tNow + ipxpz] - fields[1][totSize * tNow + ipxpz] * fields[4][totSize * tNow + ipxpz] - fields[3][totSize * tNow + ipxpz] * fields[6][totSize * tNow + ipxpz]);
                R2_ipypz = 2 * (fields[0][totSize * tNow + ipypz] * fields[5][totSize * tNow + ipypz] + fields[2][totSize * tNow + ipypz] * fields[7][totSize * tNow + ipypz] - fields[1][totSize * tNow + ipypz] * fields[4][totSize * tNow + ipypz] - fields[3][totSize * tNow + ipypz] * fields[6][totSize * tNow + ipypz]);
                R2_ipxpypz = 2 * (fields[0][totSize * tNow + ipxpypz] * fields[5][totSize * tNow + ipxpypz] + fields[2][totSize * tNow + ipxpypz] * fields[7][totSize * tNow + ipxpypz] - fields[1][totSize * tNow + ipxpypz] * fields[4][totSize * tNow + ipxpypz] - fields[3][totSize * tNow + ipxpypz] * fields[6][totSize * tNow + ipxpypz]);

                R3_i = pow(fields[0][totSize * tNow + i], 2) + pow(fields[1][totSize * tNow + i], 2) + pow(fields[2][totSize * tNow + i], 2) + pow(fields[3][totSize * tNow + i], 2) - pow(fields[4][totSize * tNow + i], 2) - pow(fields[5][totSize * tNow + i], 2) - pow(fields[6][totSize * tNow + i], 2) - pow(fields[7][totSize * tNow + i], 2);
                R3_ipx = pow(fields[0][totSize * tNow + ipx], 2) + pow(fields[1][totSize * tNow + ipx], 2) + pow(fields[2][totSize * tNow + ipx], 2) + pow(fields[3][totSize * tNow + ipx], 2) - pow(fields[4][totSize * tNow + ipx], 2) - pow(fields[5][totSize * tNow + ipx], 2) - pow(fields[6][totSize * tNow + ipx], 2) - pow(fields[7][totSize * tNow + ipx], 2);
                R3_ipy = pow(fields[0][totSize * tNow + ipy], 2) + pow(fields[1][totSize * tNow + ipy], 2) + pow(fields[2][totSize * tNow + ipy], 2) + pow(fields[3][totSize * tNow + ipy], 2) - pow(fields[4][totSize * tNow + ipy], 2) - pow(fields[5][totSize * tNow + ipy], 2) - pow(fields[6][totSize * tNow + ipy], 2) - pow(fields[7][totSize * tNow + ipy], 2);
                R3_ipz = pow(fields[0][totSize * tNow + ipz], 2) + pow(fields[1][totSize * tNow + ipz], 2) + pow(fields[2][totSize * tNow + ipz], 2) + pow(fields[3][totSize * tNow + ipz], 2) - pow(fields[4][totSize * tNow + ipz], 2) - pow(fields[5][totSize * tNow + ipz], 2) - pow(fields[6][totSize * tNow + ipz], 2) - pow(fields[7][totSize * tNow + ipz], 2);
                R3_ipxpy = pow(fields[0][totSize * tNow + ipxpy], 2) + pow(fields[1][totSize * tNow + ipxpy], 2) + pow(fields[2][totSize * tNow + ipxpy], 2) + pow(fields[3][totSize * tNow + ipxpy], 2) - pow(fields[4][totSize * tNow + ipxpy], 2) - pow(fields[5][totSize * tNow + ipxpy], 2) - pow(fields[6][totSize * tNow + ipxpy], 2) - pow(fields[7][totSize * tNow + ipxpy], 2);
                R3_ipxpz = pow(fields[0][totSize * tNow + ipxpz], 2) + pow(fields[1][totSize * tNow + ipxpz], 2) + pow(fields[2][totSize * tNow + ipxpz], 2) + pow(fields[3][totSize * tNow + ipxpz], 2) - pow(fields[4][totSize * tNow + ipxpz], 2) - pow(fields[5][totSize * tNow + ipxpz], 2) - pow(fields[6][totSize * tNow + ipxpz], 2) - pow(fields[7][totSize * tNow + ipxpz], 2);
                R3_ipypz = pow(fields[0][totSize * tNow + ipypz], 2) + pow(fields[1][totSize * tNow + ipypz], 2) + pow(fields[2][totSize * tNow + ipypz], 2) + pow(fields[3][totSize * tNow + ipypz], 2) - pow(fields[4][totSize * tNow + ipypz], 2) - pow(fields[5][totSize * tNow + ipypz], 2) - pow(fields[6][totSize * tNow + ipypz], 2) - pow(fields[7][totSize * tNow + ipypz], 2);
                R3_ipxpypz = pow(fields[0][totSize * tNow + ipxpypz], 2) + pow(fields[1][totSize * tNow + ipxpypz], 2) + pow(fields[2][totSize * tNow + ipxpypz], 2) + pow(fields[3][totSize * tNow + ipxpypz], 2) - pow(fields[4][totSize * tNow + ipxpypz], 2) - pow(fields[5][totSize * tNow + ipxpypz], 2) - pow(fields[6][totSize * tNow + ipxpypz], 2) - pow(fields[7][totSize * tNow + ipxpypz], 2);


                if (((R1_i * R1_ipx < 0) or (R1_i * R1_ipy < 0) or (R1_i * R1_ipz < 0) or
                    (R1_ipx * R1_ipxpy < 0) or (R1_ipx * R1_ipxpz < 0) or 
                    (R1_ipy * R1_ipxpy < 0) or (R1_ipy * R1_ipypz < 0) or 
                    (R1_ipz * R1_ipxpz < 0) or (R1_ipz * R1_ipypz < 0) or 
                    (R1_ipxpy * R1_ipxpypz < 0) or (R1_ipxpz * R1_ipxpypz < 0) or 
                    (R1_ipypz * R1_ipxpypz < 0)) and

                    ((R2_i * R2_ipx < 0) or (R2_i * R2_ipy < 0) or (R2_i * R2_ipz < 0) or
                    (R2_ipx * R2_ipxpy < 0) or (R2_ipx * R2_ipxpz < 0) or 
                    (R2_ipy * R2_ipxpy < 0) or (R2_ipy * R2_ipypz < 0) or 
                    (R2_ipz * R2_ipxpz < 0) or (R2_ipz * R2_ipypz < 0) or 
                    (R2_ipxpy * R2_ipxpypz < 0) or (R2_ipxpz * R2_ipxpypz < 0) or 
                    (R2_ipypz * R2_ipxpypz < 0)) and

                    ((R3_i * R3_ipx < 0) or (R3_i * R3_ipy < 0) or (R3_i * R3_ipz < 0) or
                    (R3_ipx * R3_ipxpy < 0) or (R3_ipx * R3_ipxpz < 0) or 
                    (R3_ipy * R3_ipxpy < 0) or (R3_ipy * R3_ipypz < 0) or 
                    (R3_ipz * R3_ipxpz < 0) or (R3_ipz * R3_ipypz < 0) or 
                    (R3_ipxpy * R3_ipxpypz < 0) or (R3_ipxpz * R3_ipxpypz < 0) or 
                    (R3_ipypz * R3_ipxpypz < 0))) {
                        localNM += 1;
                }
            }
        }

        // Puts required headers on valsPerLoop output file:

        if (TimeStep == 0 and rank == 0) {
            if (calcEnergy and wallDetect) { valsPerLoop << "Energy,NDW,ADW_Simple,ADW_Full\n"; }
            else {
                if (calcEnergy) { valsPerLoop << "Energy\n"; }
                if (wallDetect) { valsPerLoop << "NDW,ADW_Simple,ADW_Full\n"; }
            }
        }
        


        if (TimeStep == 0 and rank == 0) {
            if(monopoleDetect) { monopoleNumber << "NM\n"; } 
            }

        
        // If calculating the energy, add it all up and output to text
        if (calcEnergy) {

            if (rank == 0) {

                double energy = totalLocalEnergy; // Initialise the energy as the energy in the domain of this process. Then add the energy in the regions of the other processes.

                for (i = 1; i < size; i++) { MPI_Recv(&totalLocalEnergy, 1, MPI_DOUBLE, i, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  energy += totalLocalEnergy; }

                valsPerLoop << energy;
                if (wallDetect) valsPerLoop << ",";

            }
            else { MPI_Send(&totalLocalEnergy, 1, MPI_DOUBLE, 0, 20, MPI_COMM_WORLD); }

        }


        // Sum up the locally detected walls and output to text
        if (wallDetect) {

            if (rank == 0) {

                double NDW = localNDW;
                double ADW_simple = localADW_simple;
                double ADW_full = localADW_full;

                for (i = 1; i < size; i++) {

                    MPI_Recv(&localNDW, 1, MPI_DOUBLE, i, 21, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    NDW += localNDW;

                    MPI_Recv(&localADW_simple, 1, MPI_DOUBLE, i, 21, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    ADW_simple += localADW_simple;

                    MPI_Recv(&localADW_full, 1, MPI_DOUBLE, i, 21, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    ADW_full += localADW_full;

                }

                
                if (calcEnergy) valsPerLoop << ",";
                valsPerLoop << NDW << "," << ADW_simple << "," << ADW_full;

            }
            else {

                MPI_Send(&localNDW, 1, MPI_DOUBLE, 0, 21, MPI_COMM_WORLD);
                MPI_Send(&localADW_simple, 1, MPI_DOUBLE, 0, 21, MPI_COMM_WORLD);
                MPI_Send(&localADW_full, 1, MPI_DOUBLE, 0, 21, MPI_COMM_WORLD);

            }

        }

        if (monopoleDetect) {

            if (rank == 0) {

                double NM = localNM;

                for (i = 1; i < size; i++) {

                    MPI_Recv(&localNM, 1, MPI_DOUBLE, i, 21, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    NM += localNM;

                }

                monopoleNumber << NM;

            }
            else {

                MPI_Send(&localNM, 1, MPI_DOUBLE, 0, 21, MPI_COMM_WORLD);
            }

        }

        
        if (rank == 0 and monopoleDetect) { monopoleNumber << "\n"; }


        if (rank == 0 and (calcEnergy or wallDetect)) { valsPerLoop << "\n"; }


        // Update the core
        // Send sections of the core that are haloes for the other processes across to the relevant process. Then receive data for the halo of this process.
        for (comp = 0; comp < nb_fields; comp++) {

            MPI_Sendrecv(&fields[comp][totSize * tPast + frontHaloSize], nbrBackHaloSize, MPI_DOUBLE, (rank - 1 + size) % size, comp, // Send this
                &fields[comp][totSize * tPast + coreSize + frontHaloSize], backHaloSize, MPI_DOUBLE, (rank + 1) % size, comp, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Receive this

            MPI_Sendrecv(&fields[comp][totSize * tPast + coreSize + frontHaloSize - nbrFrontHaloSize], nbrFrontHaloSize, MPI_DOUBLE, (rank + 1) % size, comp,
                &fields[comp][totSize * tPast], frontHaloSize, MPI_DOUBLE, (rank - 1 + size) % size, comp, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        }

        if (rank == 0 && TimeStep == 0) {
            cout << "STEP 16: Halo update completed for timestep 0" << endl;
        }
        
    
        //Output the final fields.
        if (finalOut and TimeStep == nt - 1) {
            
            
            if (rank == 0) {
                finalFields << "R0,R1,R2,R3,R4,R5,n1,n2,n3\n";

                double R0, R1, R2, R3, R4, R5;
                double n1, n2, n3;
                int localCoreStart, localCoreSize;

                vector<vector<double>> fieldsOut(nb_fields, vector<double>(nPos, 0.0));

                for (comp = 0; comp < nb_fields; comp++) {

                    for (i = 0; i < coreSize; i++) { fieldsOut[comp][i] = fields[comp][frontHaloSize + i]; }

                    for (i = 1; i < size; i++) {

                        
                        if (i < chunkRem) { localCoreStart = i * (chunk + 1); localCoreSize = chunk + 1; }
                        else { localCoreStart = i * chunk + chunkRem; localCoreSize = chunk; }

                        MPI_Recv(&fieldsOut[comp][localCoreStart], localCoreSize, MPI_DOUBLE, i, comp, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    }

                }


                for (i = 0; i < nPos; i++) {

                    R0 = pow(fieldsOut[0][i], 2) + pow(fieldsOut[1][i], 2) + pow(fieldsOut[2][i], 2) + pow(fieldsOut[3][i], 2) + pow(fieldsOut[4][i], 2) + pow(fieldsOut[5][i], 2) + pow(fieldsOut[6][i], 2) + pow(fieldsOut[7][i], 2);
                    R1 = 2 * (fieldsOut[0][i] * fieldsOut[4][i] + fieldsOut[1][i] * fieldsOut[5][i] + fieldsOut[2][i] * fieldsOut[6][i] + fieldsOut[3][i] * fieldsOut[7][i]);
                    R2 = 2 * (fieldsOut[0][i] * fieldsOut[5][i] - fieldsOut[1][i] * fieldsOut[4][i] + fieldsOut[2][i] * fieldsOut[7][i] - fieldsOut[3][i] * fieldsOut[6][i]);
                    R3 = pow(fieldsOut[0][i], 2) + pow(fieldsOut[1][i], 2) + pow(fieldsOut[2][i], 2) + pow(fieldsOut[3][i], 2) - pow(fieldsOut[4][i], 2) - pow(fieldsOut[5][i], 2) - pow(fieldsOut[6][i], 2) - pow(fieldsOut[7][i], 2);
                    R4 = 2 * (fieldsOut[0][i] * fieldsOut[6][i] - fieldsOut[1][i] * fieldsOut[7][i] - fieldsOut[2][i] * fieldsOut[4][i] + fieldsOut[3][i] * fieldsOut[5][i]);
                    R5 = 2 * (fieldsOut[0][i] * fieldsOut[7][i] + fieldsOut[1][i] * fieldsOut[6][i] - fieldsOut[2][i] * fieldsOut[5][i] - fieldsOut[3][i] * fieldsOut[4][i]);

                    n1 = -2 * (fieldsOut[0][i] * fieldsOut[2][i] + fieldsOut[1][i] * fieldsOut[3][i] + fieldsOut[4][i] * fieldsOut[6][i] + fieldsOut[5][i] * fieldsOut[7][i]);
                    n2 = -2 * (fieldsOut[0][i] * fieldsOut[3][i] - fieldsOut[1][i] * fieldsOut[2][i] + fieldsOut[4][i] * fieldsOut[7][i] - fieldsOut[5][i] * fieldsOut[6][i]);
                    n3 = -1 * (pow(fieldsOut[0][i], 2) + pow(fieldsOut[1][i], 2) - pow(fieldsOut[2][i], 2) - pow(fieldsOut[3][i], 2) + pow(fieldsOut[4][i], 2) + pow(fieldsOut[5][i], 2) - pow(fieldsOut[6][i], 2) - pow(fieldsOut[7][i], 2));


                    finalFields << R0 << "," << R1 << "," << R2 << "," << R3 << "," << R4 << "," << R5 << "," << n1 << "," << n2 << "," << n3 << "\n";


                }



            }

            else {

                for (comp = 0; comp < nb_fields; comp++) {

                    MPI_Send(&fields[comp][frontHaloSize], coreSize, MPI_DOUBLE, 0, comp, MPI_COMM_WORLD);

                }
            }





        }

        // Gif Output

        /*

        if (makeGif and TimeStep % saveFreq == 0 and TimeStep != 0) {


            if (rank == 0) {
                // Create files
                string TimeStepPath = out_path + "fields_timestep=" + to_string(TimeStep) + outTag + ".csv";
                ofstream Gif(TimeStepPath.c_str());
                Gif << "R1" << " " << "R2" << " " << "R3" << endl;


                double R1nt;
                double R2nt;
                double R3nt;
                int localCoreStartnt;
                int localCoreSizent;

                for (comp = 0; comp < nb_fields; comp++) {

                    for (j = 0; j < coreSize; j++) { fieldsOutnt[comp][j] = fields[comp][frontHaloSize + j]; }

                    for (j = 1; j < size; j++) {

                        
                        if (j < chunkRem) { localCoreStartnt = j * (chunk + 1); localCoreSizent = chunk + 1; }
                        else { localCoreStartnt = j * chunk + chunkRem; localCoreSizent = chunk; }

                        MPI_Recv(&fieldsOutnt[comp][localCoreStartnt], localCoreSizent, MPI_DOUBLE, j, comp, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    }
                }

                for (j = 0; j < nPos; j++) {

                    R1nt = 2 * (fieldsOutnt[0][j] * fieldsOutnt[4][j] + fieldsOutnt[1][j] * fieldsOutnt[5][j] + fieldsOutnt[2][j] * fieldsOutnt[6][j] + fieldsOutnt[3][j] * fieldsOutnt[7][j]);
                    R2nt = 2 * (fieldsOutnt[0][j] * fieldsOutnt[5][j] + fieldsOutnt[2][j] * fieldsOutnt[7][j] - fieldsOutnt[1][j] * fieldsOutnt[4][j] - fieldsOutnt[3][j] * fieldsOutnt[6][j]);
                    R3nt = pow(fieldsOutnt[0][j], 2) + pow(fieldsOutnt[1][j], 2) + pow(fieldsOutnt[2][j], 2) + pow(fieldsOutnt[3][j], 2) - pow(fieldsOutnt[4][j], 2) - pow(fieldsOutnt[5][j], 2) - pow(fieldsOutnt[6][j], 2) - pow(fieldsOutnt[7][j], 2);

                    Gif << R1nt << " " << R2nt << " " << R3nt << endl;

                }

            }

            else {

                for (comp = 0; comp < nb_fields; comp++) {

                    MPI_Send(&fields[comp][frontHaloSize], coreSize, MPI_DOUBLE, 0, comp, MPI_COMM_WORLD);

                }
            }

        }
        */


        if (makeGif and TimeStep % sep_saveFreq == 0) {

            if (rank == 0) {

                
                ofstream rValuesFile;
                ostringstream dummyStream;
                ostream* rValuesStreamPtr = nullptr;

                if (TimeStep % R_saveFreq == 0) {
                    string rValuesPath = out_path + "R_values_" +  "_timestep=" + to_string(TimeStep) + outTag + ".csv";
                    rValuesFile.open(rValuesPath.c_str());
                    rValuesFile << "R1nt,R2nt,R3nt\n";
                    rValuesFile << fixed << setprecision(6);
                    rValuesStreamPtr = &rValuesFile;
                } else {
                    dummyStream.str(""); // Clear any previous content
                    rValuesStreamPtr = &dummyStream;
                }
                // Create files for fields and R values
                
                
                vector<vector<double>> fieldsOutnt(nb_fields, vector<double>(nPos, 0.0));


                // Gather field data from all processes
                for (comp = 0; comp < nb_fields; comp++) {

                    for (j = 0; j < coreSize; j++) { 
                        fieldsOutnt[comp][j] = fields[comp][frontHaloSize + j]; 
                    }

                    for (j = 1; j < size; j++) 
                    {

                        if (j < chunkRem) { 
                            localCoreStartnt = j * (chunk + 1); localCoreSizent = chunk + 1; 
                        }
                        else {
                             localCoreStartnt = j * chunk + chunkRem; localCoreSizent = chunk; 
                        }

                        MPI_Recv(&fieldsOutnt[comp][localCoreStartnt], localCoreSizent, MPI_DOUBLE, j, comp, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }

                
                vector<double> monopole_field(nPos); 
                    
                // Output fields and R values to separate files
                for (j = 0; j < nPos; j++) {

                    // Compute R values
                    R1nt = 2 * (fieldsOutnt[0][j] * fieldsOutnt[4][j] + fieldsOutnt[1][j] * fieldsOutnt[5][j] + fieldsOutnt[2][j] * fieldsOutnt[6][j] + fieldsOutnt[3][j] * fieldsOutnt[7][j]);
                    R2nt = 2 * (fieldsOutnt[0][j] * fieldsOutnt[5][j] + fieldsOutnt[2][j] * fieldsOutnt[7][j] - fieldsOutnt[1][j] * fieldsOutnt[4][j] - fieldsOutnt[3][j] * fieldsOutnt[6][j]);
                    R3nt = pow(fieldsOutnt[0][j], 2) + pow(fieldsOutnt[1][j], 2) + pow(fieldsOutnt[2][j], 2) + pow(fieldsOutnt[3][j], 2) - pow(fieldsOutnt[4][j], 2) - pow(fieldsOutnt[5][j], 2) - pow(fieldsOutnt[6][j], 2) - pow(fieldsOutnt[7][j], 2);

                    // Write R values to R values file, ensuring explicit output of 0.0
                    (*rValuesStreamPtr) << R1nt << "," << R2nt << "," << R3nt << "\n";

                    if (ic_type == "monopole") {
                    monopole_field[j] = R1nt*R1nt + R2nt*R2nt + R3nt*R3nt;
                    }
                }

                if (TimeStep % R_saveFreq == 0) {
                    rValuesFile.close();
                }

                                // Monopole tracking for monopole initial conditions
                if (ic_type == "monopole") {
                    
                    // Create monopole tracking file if this is the first save
                    string monopoleTrackingPath = out_path + "monopole_tracking_" +  outTag + ".csv";
                    ofstream monopoleFile;
                    
                    if (TimeStep == 0) {
                        monopoleFile.open(monopoleTrackingPath.c_str());
                        monopoleFile << "timestep,x1_center,y1_center,z1_center,x2_center,y2_center,z2_center" << endl;
                    } else {
                        monopoleFile.open(monopoleTrackingPath.c_str(), ios::app);
                    }
                    
                    // Simple approach: find two lowest values with separation constraint
                    min1_value = 1e10;
                    min2_value = 1e10;
                    min1_idx = -1;
                    min2_idx = -1;
                    
                    // First pass: find absolute minimum
                    for (j = 0; j < nPos; j++) {
                        if (monopole_field[j] < min1_value) {
                            min1_value = monopole_field[j];
                            min1_idx = j;
                        }
                    }
                    

                    // Second pass: find second minimum with separation constraint
                    if (min1_idx != -1) {
                        // Convert min1_idx to 3D coordinates
                        k1 = min1_idx % nz;
                        j1 = (min1_idx / nz) % ny;
                        i1 = min1_idx / (ny * nz);
                        
                        // Find maximum value within 2 grid points of first minimum
                        max_near_min1 = 0.0;
                        
                                                
                        for (j = 0; j < nPos; j++) {
                            k_j = j % nz;
                            j_j = (j / nz) % ny;
                            i_j = j / (ny * nz);

                            dx_diff = i_j - i1;
                            dy_diff = j_j - j1;
                            dz_diff = k_j - k1;
                            distance_squared = dx_diff*dx_diff + dy_diff*dy_diff + dz_diff*dz_diff;
                            
                            if (distance_squared <= 4.0) {
                                max_near_min1 = max(max_near_min1, monopole_field[j]);
                            }
                        }
                        
                        // Find second minimum outside 5 grid point radius and below max_near_min1
                        for (j = 0; j < nPos; j++) {
                            k_j = j % nz;
                            j_j = (j / nz) % ny;
                            i_j = j / (ny * nz);

                            dx_diff = i_j - i1;
                            dy_diff = j_j - j1;
                            dz_diff = k_j - k1;
                            distance_squared = dx_diff*dx_diff + dy_diff*dy_diff + dz_diff*dz_diff;

                            if (distance_squared > 25.0 && monopole_field[j] < min2_value && monopole_field[j] < max_near_min1) {
                                min2_value = monopole_field[j];
                                min2_idx = j;
                            }
                        }
                    }
                    
                    // Convert indices to physical coordinates and output
                    double x1_center = -1, y1_center = -1, z1_center = -1;
                    double x2_center = -1, y2_center = -1, z2_center = -1;
                    
                    if (min1_idx != -1) {
                        
                        x1_center = i1 * dx;
                        y1_center = j1 * dy;
                        z1_center = k1 * dz;
                    }
                    
                    if (min2_idx != -1) {
                        int k2 = min2_idx % nz;
                        int j2 = (min2_idx / nz) % ny;
                        int i2 = min2_idx / (ny * nz);
                        
                        x2_center = i2 * dx;
                        y2_center = j2 * dy;
                        z2_center = k2 * dz;
                    }
                    
                    // Output to file (using -1 instead of NaN for compatibility)
                    monopoleFile << TimeStep << "," << x1_center << "," << y1_center << "," << z1_center << "," 
                                << x2_center << "," << y2_center << "," << z2_center << endl;
                    
                    monopoleFile.close();
                }
            }

            else {
                // Send field data to rank 0
                for (comp = 0; comp < nb_fields; comp++) {
                    MPI_Send(&fields[comp][frontHaloSize], coreSize, MPI_DOUBLE, 0, comp, MPI_COMM_WORLD);
                }
            }
        }

        



	
	// Simulation Progress Output
        if (rank == 0 and TimeStep % countRate == 0) {
        
            cout << "\rTimestep " << TimeStep << " completed.";
        
        }

        // Barrier before going to the next timestep.
        MPI_Barrier(MPI_COMM_WORLD);

    }

    if (rank == 0) {
        gettimeofday(&evolution_end, NULL);
        double evolution_time = (evolution_end.tv_sec - evolution_start.tv_sec) + (evolution_end.tv_usec - evolution_start.tv_usec)/1000000.0;
        
        cout << "\rTimestep " << nt << " completed." << endl;
        cout << "STEP 21: Main evolution loop completed" << endl;
        cout << "Field evolution time: " << evolution_time << "s" << endl;
        cout << "Average time per timestep: " << evolution_time/nt << "s" << endl;

        gettimeofday(&end, NULL);
        double total_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1000000.0;

        cout << "=== TIMING SUMMARY ===" << endl;
        cout << "Setup time: " << (setup_end.tv_sec - setup_start.tv_sec) + (setup_end.tv_usec - setup_start.tv_usec)/1000000.0 << "s" << endl;
        cout << "Initial conditions time: " << (ic_end.tv_sec - ic_start.tv_sec) + (ic_end.tv_usec - ic_start.tv_usec)/1000000.0 << "s" << endl;
        cout << "Evolution time: " << evolution_time << "s (" << (evolution_time/total_time)*100 << "% of total)" << endl;
        cout << "Total simulation time: " << total_time << "s" << endl;
        cout << "STEP 22: Timing analysis completed" << endl;
    }

    // Deletes redundent outpur files if not used:
    if (rank == 0) {

        if (!finalOut) {
            finalFields.close();
            remove(finalFieldPath.c_str());
        }

        if (!calcEnergy and !wallDetect) { 
            valsPerLoop.close();
            remove(valsPerLoopPath.c_str());
        }
        
        cout << "STEP 23: File cleanup completed" << endl;
    }

    MPI_Finalize();

    if (rank == 0) {
        cout << "STEP 24: MPI finalized - program completed successfully" << endl;
    }

    return 0;
}