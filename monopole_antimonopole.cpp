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

const long long int nx = 64; // Grid Dimensions
const long long int ny = 64;
const long long int nz = 64; // Set nz = 1 for 2D.
const long long int nPos = nx * ny * nz;

const double dx = 0.5; //Grid Spacings
const double dy = 0.5;
const double dz = 0.5;
const double dt = 0.1; //..KEEP 1 TO 5 RATIO, KEEP BELOW 0.5

// const int nt = (nx * dx / (2 * dt)); // nt required for sim to end at light crossing time is nx*dx/(2*dt)
const int nt = (nx * dx / (2 * dt));

const int seed = 73;

const string outTag = "_nx" + to_string(nx) + "_nt" + to_string(nt) + "_seed" + to_string(seed) + "_Z2";

const bool calcEnergy = true; // Output Choices
const bool wallDetect = false;
const bool finalOut = true;

const bool monopoleDetect = false;

const bool makeGif = true;
const int saveFreq = 2;

const string file_path = __FILE__;
const string dir_path = "./Data/"; // Data Directory Location - fixed path

const int countRate = 20; // Increments for simulation progress status output.

const string ic_type = "monopole";
const string bc_type = "fixed";

const int nb_fields = 8; // Number of fields in simulation





// 2HDM Z_2 Symmetric Potential Set-Up:
 
// Mass and Energy Paramaters (CAN be chosen)
const long double m_h = 125;
const long double V_sm = 246;
const long double m_H = 0;
const long double m_A = 0;
const long double m_H_pm = 200;

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
    double totalLocalEnergy, localNDW, localADW_simple, localADW_full, localNM, x1, y1, z1, x2, y2, z2;
    long long int i, j, k, TimeStep, tNow, tPast, comp, imx, ipx, imy, ipy, imz, ipz, ipxmy, ipxmz, imxpy, ipymz, imxpz, imypz, imxx, ipxx, imyy, ipyy, imzz, ipzz, ipxpy, ipxpz, ipypz, ipxpypz;

    vector<vector<double>> k_kp(4, vector<double>(2 * totSize, 0.0)); 
    vector<vector<double>> g_gp(4, vector<double>(2 * totSize, 0.0));
    
    if (rank == 0) {
        cout << "STEP 5: Field arrays allocated" << endl;
    }

    struct timeval start, end;
    if (rank == 0) { gettimeofday(&start, NULL); }

    stringstream ss;

    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "STEP 6: MPI barrier passed, starting file operations" << endl;
    }

    //Creates Output Files if required
    string icPath = dir_path + "ic.txt";
    ifstream ic(icPath.c_str());

    string finalFieldPath = dir_path + "vortices_ gif_finalFields" + outTag + ".txt";
    ofstream finalFields(finalFieldPath.c_str());

    string valsPerLoopPath = dir_path + "t_gamma=2pi_3_energy" + outTag + ".txt";
    ofstream valsPerLoop(valsPerLoopPath.c_str());

    string monopoleNumberPath = dir_path + "2m_monopoleNumber" + outTag + ".txt";
    ofstream monopoleNumber(monopoleNumberPath.c_str());

    if (rank == 0) {
        cout << "STEP 7: Output files created" << endl;
    }

    //DISTANCE BETWEEN MONOPOLE AND ANTIMONOPOLE
    // Index values (not neccessarily on grid and hence not integers) of the zero coordinate.
    x1 = 0.5 * (nx - 1);
    y1 = 0.5 * (ny - 1);
    z1 = 0.5 * (nz + 23); // z1 = 76

    x2 = 0.5 * (nx - 1);
    y2 = 0.5 * (ny - 1);
    z2 = 0.5 * (nz - 25); // z2 = 52

    if (rank == 0) {
        cout << "STEP 8: Monopole positions calculated" << endl;
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
        }

        string fields_ic_data = dir_path + "SOR_Fields.txt";
        
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


        for (i = frontHaloSize; i < coreSize + frontHaloSize; i++) {

            if (rank == 0 && i == frontHaloSize) {
                cout << "STEP 9d: Starting main monopole calculation loop" << endl;
            }

            //First monopole

            double x_1 = ( (i+dataStart)/(ny*nz) - x1 )*dx;
            double y_1 = ( ((i+dataStart)/nz)%ny - y1 )*dy;
            double z_1 = ( (i+dataStart)%nz - z1 )*dz;

            double r_1 = pow(x_1*x_1 + y_1*y_1 + z_1*z_1, 0.5); // Calculate r_pos
            double r_pos_1 = r_1 / 0.01; //Position of r in the smaller grid
            int r_c_1 = static_cast<int>(round(r_pos_1)); 
            int r_m_1 = r_c_1 - 1;
            int r_p_1 = r_c_1 + 1;

            // Debugging output to check bounds and values
            if (r_c_1 < 0) {

                cout << "Error: Index out of bounds at process " << rank 
                    << " with i=" << i << ", r_c_1=" << r_c_1 
                    << ", r_p_1=" << r_p_1 << ", x_1=" << x_1 << ", y_1=" << y_1 
                    << ", z_1=" << z_1 << ", r_1=" << r_1 << ", r_pos_1=" << r_pos_1 << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Declare k_r and k_p_r here so they are accessible later
            double k_1 = 0.0;
            double k_1_p = 0.0;
            
            if (r_p_1 >= (k.size())) {

                k_1 = 1.0;
                k_1_p = 0.0;

            } else if (r_c_1 == 0) {

                // Values of k and k+ at r_value
                k_1 = ((( - (r_c_1 - r_pos_1) * k[r_p_1] )) 
                        + ((r_p_1 - r_pos_1) * k[r_c_1]));
                k_1_p = ((( - (r_c_1 - r_pos_1) * k_p[r_p_1] )) 
                        + ((r_p_1 - r_pos_1) * k_p[r_c_1]));

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

            double r_2 = pow(x_2*x_2 + y_2*y_2 + z_2*z_2, 0.5); // Calculate r_pos
            double r_pos_2 = r_2 / 0.01; //Position of r in the smaller grid
            int r_c_2 = static_cast<int>(round(r_pos_2)); 
            int r_m_2 = r_c_2 - 1;
            int r_p_2 = r_c_2 + 1;

            // Debugging output to check bounds and values
            if (r_c_2 < 0) {

                cout << "Error: Index out of bounds at process " << rank 
                    << " with i=" << i << ", r_c_2=" << r_c_2 
                    << ", r_p_2=" << r_p_2 << ", x_2=" << x_2 << ", y_2=" << y_2 
                    << ", z_2=" << z_2 << ", r_2=" << r_2 << ", r_pos_2=" << r_pos_2 << endl;
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

            if ( z_1 == r_1 ) {

                u_1[0][0] = complex<double>(1.0, 0.0);  
                u_1[0][1] = complex<double>(0.0, 0.0); 
                u_1[1][0] = complex<double>(0.0, 0.0); 
                u_1[1][1] = complex<double>(1.0, 0.0);

                u_2[0][0] = complex<double>(0.0, 0.0);  
                u_2[0][1] = complex<double>(-1.0, 0.0); 
                u_2[1][0] = complex<double>(1.0, 0.0); 
                u_2[1][1] = complex<double>(0.0, 0.0);

                
            } else if ( z_2 == -r_2 ) {

                u_1[0][0] = complex<double>(0.0, 0.0);  
                u_1[0][1] = complex<double>(-1.0, 0.0); 
                u_1[1][0] = complex<double>(1.0, 0.0); 
                u_1[1][1] = complex<double>(0.0, 0.0);

                u_2[0][0] = complex<double>(-1.0, 0.0);  
                u_2[0][1] = complex<double>(0.0, 0.0); 
                u_2[1][0] = complex<double>(0.0, 0.0); 
                u_2[1][1] = complex<double>(-1.0, 0.0);
                
                        
            } else if ( z_1 != r_1 and z_2 != -r_2 and z_2 == r_2) {

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

                cos_1 = pow(0.5 * (1 + (z_1 / r_1)), 0.5);  // cos(theta_1 / 2)


                u_1[0][0] = complex<double>(cos_1, 0.0);  
                u_1[0][1] = complex<double>(- x_1 / (2 * r_1 * cos_1), y_1 / (2 * r_1 * cos_1)); 
                u_1[1][0] = complex<double>(x_1 / (2 * r_1 * cos_1), y_1 / (2 * r_1 * cos_1)); 
                u_1[1][1] = complex<double>(cos_1, 0.0);

                double sin_2;

                sin_2 = pow(0.5 * (1 - (z_2 / r_2)), 0.5);

                u_2[0][0] = complex<double>(- sin_2, 0.0);  
                u_2[0][1] = complex<double>(- x_2 / (2 * r_2 * sin_2), y_2 / (2 * r_2 * sin_2)); 
                u_2[1][0] = complex<double>(x_2 / (2 * r_2 * sin_2), y_2 / (2 * r_2 * sin_2)); 
                u_2[1][1] = complex<double>(- sin_2, 0.0);

            }

            double gamma = (2 * pi / 3);

            // Define the T matrix
            complex<double> T[2][2] = {
                {exp(complex<double>(0, 0.5 * gamma)), complex<double>(0.0, 0.0)},
                {complex<double>(0.0, 0.0), exp(complex<double>(0, -0.5 * gamma))}
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

            

            // Multiply the tensor product by the prefactor v_sm / sqrt(2)
            double prefactor = (pow(2, -1.5));
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    TP[r][c] *= prefactor;
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
            fields[0][i] = phi[0].real();  
            fields[1][i] = phi[0].imag();  
            fields[2][i] = phi[1].real();  
            fields[3][i] = phi[1].imag();  
            fields[4][i] = phi[2].real();  
            fields[5][i] = phi[2].imag();  
            fields[6][i] = phi[3].real();  
            fields[7][i] = phi[3].imag();


            fields[0][totSize + i] = fields[0][i];
            fields[1][totSize + i] = fields[1][i];
            fields[2][totSize + i] = fields[2][i];
            fields[3][totSize + i] = fields[3][i];
            fields[4][totSize + i] = fields[4][i];
            fields[5][totSize + i] = fields[5][i];
            fields[6][totSize + i] = fields[6][i];
            fields[7][totSize + i] = fields[7][i];

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
            
            string test_kValuesPath = dir_path + "test_m_am_kValues" + outTag + ".txt";
            string test_gValuesPath = dir_path + "test_m_am_gValues" + outTag + ".txt";
            
            ofstream test_kValuesFile(test_kValuesPath.c_str());
            ofstream test_gValuesFile(test_gValuesPath.c_str());

            test_kValuesFile << "k1" << " " << "k1p" << " " << "k2" << " " << "k2p" << endl;
            test_gValuesFile << "- g1p * g2p" << " " << "g1 * g2 " << " " << "- g1 * g2" << " " << "g1p * g2p" << endl;

            vector<vector<double>> kOut(4, vector<double>(nPos, 0.0));
            vector<vector<double>> gOut(4, vector<double>(nPos, 0.0));

            // Gather field data from all processes
            for (comp = 0; comp < 4; comp++) {

                for (j = 0; j < coreSize; j++) { 
                    
                    kOut[comp][j] = k_kp[comp][frontHaloSize + j]; 
                    gOut[comp][j] = g_gp[comp][frontHaloSize + j]; 
                    
                }

                for (j = 1; j < size; j++) {

                    int localCoreStartnt;
                    int localCoreSizent;
                    if (j < chunkRem) { localCoreStartnt = j * (chunk + 1); localCoreSizent = chunk + 1; }
                    else { localCoreStartnt = j * chunk + chunkRem; localCoreSizent = chunk; }

                    MPI_Recv(&kOut[comp][localCoreStartnt], localCoreSizent, MPI_DOUBLE, j, comp, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&gOut[comp][localCoreStartnt], localCoreSizent, MPI_DOUBLE, j, comp, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }

            // Output fields and R values to separate files
            for (j = 0; j < nPos; j++) {


                // Write R values to R values file, ensuring explicit output of 0.0
                test_kValuesFile << kOut[0][j] << " " << kOut[1][j] << " " << kOut[2][j] << " " << kOut[3][j] << endl;
                test_gValuesFile << gOut[0][j] << " " << gOut[1][j] << " " << gOut[2][j] << " " << gOut[3][j] << endl;
                
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
            string test_fieldsPath = dir_path + "test_m_am_fieldValues" + outTag + ".txt";
            string test_rValuesPath = dir_path + "test_m_am_RValues" + outTag + ".txt";
            
            ofstream test_fieldsFile(test_fieldsPath.c_str());
            ofstream test_rValuesFile(test_rValuesPath.c_str());

            // Headers for fields and R values
            test_fieldsFile << "fieldsOutnt[0] fieldsOutnt[1] fieldsOutnt[2] fieldsOutnt[3] "
                    << "fieldsOutnt[4] fieldsOutnt[5] fieldsOutnt[6] fieldsOutnt[7]" << endl;

            test_rValuesFile << "R0nt R1nt R2nt R3nt" << endl;

            vector<vector<double>> fieldsOutnt(nb_fields, vector<double>(nPos, 0.0));
            double R0nt, R1nt, R2nt, R3nt;

            // Gather field data from all processes
            for (comp = 0; comp < nb_fields; comp++) {

                for (j = 0; j < coreSize; j++) { fieldsOutnt[comp][j] = fields[comp][frontHaloSize + j]; }

                for (j = 1; j < size; j++) {

                    int localCoreStartnt;
                    int localCoreSizent;
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
                R3nt = pow(fieldsOutnt[0][j], 2) + pow(fieldsOutnt[1][j], 2) + pow(fieldsOutnt[2][j], 2) + pow(fieldsOutnt[3][j], 2)
                    - pow(fieldsOutnt[4][j], 2) - pow(fieldsOutnt[5][j], 2) - pow(fieldsOutnt[6][j], 2) - pow(fieldsOutnt[7][j], 2);
                R0nt = pow(fieldsOutnt[0][j], 2) + pow(fieldsOutnt[1][j], 2) + pow(fieldsOutnt[2][j], 2) + pow(fieldsOutnt[3][j], 2) + pow(fieldsOutnt[4][j], 2) + pow(fieldsOutnt[5][j], 2) + pow(fieldsOutnt[6][j], 2) + pow(fieldsOutnt[7][j], 2);

                // Write field values to fields file, ensuring explicit output of 0.0
                test_fieldsFile << (fieldsOutnt[0][j] == 0.0 ? "0.0" : to_string(fieldsOutnt[0][j])) << " "
                        << (fieldsOutnt[1][j] == 0.0 ? "0.0" : to_string(fieldsOutnt[1][j])) << " "
                        << (fieldsOutnt[2][j] == 0.0 ? "0.0" : to_string(fieldsOutnt[2][j])) << " "
                        << (fieldsOutnt[3][j] == 0.0 ? "0.0" : to_string(fieldsOutnt[3][j])) << " "
                        << (fieldsOutnt[4][j] == 0.0 ? "0.0" : to_string(fieldsOutnt[4][j])) << " "
                        << (fieldsOutnt[5][j] == 0.0 ? "0.0" : to_string(fieldsOutnt[5][j])) << " "
                        << (fieldsOutnt[6][j] == 0.0 ? "0.0" : to_string(fieldsOutnt[6][j])) << " "
                        << (fieldsOutnt[7][j] == 0.0 ? "0.0" : to_string(fieldsOutnt[7][j])) << endl;

                // Write R values to R values file, ensuring explicit output of 0.0
                test_rValuesFile << (R0nt == 0.0 ? "0.0" : to_string(R0nt)) << " "
                            << (R1nt == 0.0 ? "0.0" : to_string(R1nt)) << " "
                            << (R2nt == 0.0 ? "0.0" : to_string(R2nt)) << " "
                            << (R3nt == 0.0 ? "0.0" : to_string(R3nt)) << endl;
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

    if (rank == 0) { cout << "Initial data loaded/generated in: " << end.tv_sec - start.tv_sec << "s" << endl; }




    // Main for loop that evolves the fields:
    for (TimeStep = 0; TimeStep < nt; TimeStep++) {

        if (rank == 0 && TimeStep == 0) {
            cout << "STEP 11: Starting main evolution loop" << endl;
        }

        double fric, tau;


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

        //Loops over and evolves all assigned data points
        for (i = frontHaloSize; i < coreSize + frontHaloSize; i++) { 
            
            //Fixed BCs

            double i_coord = (i + dataStart) / (ny * nz);
            double j_coord = (((i + dataStart) / nz) % ny);
            double k_coord = ((i + dataStart) % nz);

            if (bc_type == "fixed" and (i_coord == 0 or i_coord == nx-1 or j_coord == 0 or j_coord == ny-1 or k_coord == 0 or k_coord == nz-1)) {

                continue;
            }
            
            //Encoding the periodic boundary conditions for the evolving data point:
            
            // No need to worry about periodicity with the x neighbours because halo is designed to contain them
            imx = i - ny * nz;
            ipx = i + ny * nz;
            imxx = i - 2 * ny * nz;
            ipxx = i + 2 * ny * nz;


            // Need to account for the periodicity of the space for the other two directions:
            // Convert to global position in array to do modulo arithmetic. The second to last term gives ny*nz*floor((i+dataStart)/(ny*nz)). The last term converts back to the position in the local array 
            imy = (i + dataStart - nz + ny * nz) % (ny * nz) + ((i + dataStart) / (ny * nz)) * ny * nz - dataStart;
            ipy = (i + dataStart + nz) % (ny * nz) + ((i + dataStart) / (ny * nz)) * ny * nz - dataStart;
            imyy = (i + dataStart - 2 * nz + ny * nz) % (ny * nz) + ((i + dataStart) / (ny * nz)) * ny * nz - dataStart;
            ipyy = (i + dataStart + 2 * nz) % (ny * nz) + ((i + dataStart) / (ny * nz)) * ny * nz - dataStart;

            imz = (i + dataStart - 1 + nz) % nz + ((i + dataStart) / nz) * nz - dataStart;
            ipz = (i + dataStart + 1) % nz + ((i + dataStart) / nz) * nz - dataStart;
            imzz = (i + dataStart - 2 + nz) % nz + ((i + dataStart) / nz) * nz - dataStart;
            ipzz = (i + dataStart + 2) % nz + ((i + dataStart) / nz) * nz - dataStart;

            // Additionally needed for wilson loop calculations. Avoid using x shifted points first as this makes the calculations more complicated and some of these points aren't in the correct positions
            ipxmy = imy + ny * nz;
            ipxmz = imz + ny * nz;
            imxpy = ipy - ny * nz;
            ipymz = (ipy + dataStart - 1 + nz) % nz + ((ipy + dataStart) / nz) * nz - dataStart;
            imxpz = ipz - ny * nz;
            imypz = (imy + dataStart + 1) % nz + ((imy + dataStart) / nz) * nz - dataStart;


            //Used for monopole detection

            ipxpy = (ipy + ny * nz); // One unit in the positive x and y directions
            ipxpz = (ipz + ny * nz); // One unit in the positive x and z directions
            ipypz = (ipy + dataStart + 1) % nz + ((ipy + dataStart) / nz) * nz - dataStart; // One unit in the positive y and z directions
            ipxpypz = ipypz + ny * nz; // One unit in the positive x, y, and z directions




            // Define in Shorthand the current value of each field at the evolving point:
            double f0 = fields[0][totSize * tNow + i];
            double f1 = fields[1][totSize * tNow + i];
            double f2 = fields[2][totSize * tNow + i];
            double f3 = fields[3][totSize * tNow + i];
            double f4 = fields[4][totSize * tNow + i];
            double f5 = fields[5][totSize * tNow + i];
            double f6 = fields[6][totSize * tNow + i];
            double f7 = fields[7][totSize * tNow + i];





            // Calculate the second order time derivatives and update the fields:


            //First Doublet:
            for (comp = 0; comp <= 3; comp++) {

                //2nd Spatial Deriviitive Calculations (central finite difference method) - to 4th Order:

                if (bc_type == "fixed" and (i_coord == 1 or i_coord == nx-2)) {

                    fieldxx = (fields[comp][totSize * tNow + ipx] - 2 * fields[comp][totSize * tNow + i] + fields[comp][totSize * tNow + imx]) / (dx * dx);

                } else {

                    fieldxx = (16 * (fields[comp][totSize * tNow + ipx] + fields[comp][totSize * tNow + imx]) - 30 * fields[comp][totSize * tNow + i] - fields[comp][totSize * tNow + ipxx] - fields[comp][totSize * tNow + imxx]) / (12 * dx * dx);

                }

                if (bc_type == "fixed" and (j_coord == 1 or j_coord == ny-2)) {

                    fieldyy = (fields[comp][totSize * tNow + ipy] - 2 * fields[comp][totSize * tNow + i] + fields[comp][totSize * tNow + imy]) / (dy * dy);

                } else {

                    fieldyy = (16 * (fields[comp][totSize * tNow + ipy] + fields[comp][totSize * tNow + imy]) - 30 * fields[comp][totSize * tNow + i] - fields[comp][totSize * tNow + ipyy] - fields[comp][totSize * tNow + imyy]) / (12 * dy * dy);

                }

                if (bc_type == "fixed" and (k_coord == 1 or k_coord == nz-2)) {

                    fieldzz = (fields[comp][totSize * tNow + ipz] - 2 * fields[comp][totSize * tNow + i] + fields[comp][totSize * tNow + imz]) / (dz * dz);

                } else {

                    fieldzz = (16 * (fields[comp][totSize * tNow + ipz] + fields[comp][totSize * tNow + imz]) - 30 * fields[comp][totSize * tNow + i] - fields[comp][totSize * tNow + ipzz] - fields[comp][totSize * tNow + imzz]) / (12 * dz * dz);

                }


                //1st Temporal Derivitive Calculation (only required for damping):
                fieldt[comp] = (fields[comp][totSize * tNow + i] - fields[comp][totSize * tPast + i]) / dt;


                //2nd Temporal Derivitive Calculation - Using EoM:
                fieldtt[comp] = fieldxx + fieldyy + fieldzz - fric * fieldt[comp] - pow(pow(tau, scaling), bbeta) * (-mu_1_sq * fields[comp][totSize * tNow + i] + 2 * lambda_1 * fields[comp][totSize * tNow + i] * (pow(f0, 2) + pow(f1, 2) + pow(f2, 2) + pow(f3, 2)) + lambda_3 * fields[comp][totSize * tNow + i] * (pow(f4, 2) + pow(f5, 2) + pow(f6, 2) + pow(f7, 2)));

                //Different potential contributions dependant upon the component of the field vector:
                if (comp == 0) {

                    fieldtt[comp] += -pow(pow(tau, scaling), bbeta) * ((l4_m_l5)*f4 * (f0 * f4 + f1 * f5 + f2 * f6 + f3 * f7));
                    fieldtt[comp] += -pow(pow(tau, scaling), bbeta) * ((l4_p_l5)*f5 * (f0 * f5 - f1 * f4 + f2 * f7 - f3 * f6));

                }

                if (comp == 1) {

                    fieldtt[comp] += -pow(pow(tau, scaling), bbeta) * ((l4_m_l5)*f5 * (f0 * f4 + f1 * f5 + f2 * f6 + f3 * f7));
                    fieldtt[comp] += -pow(pow(tau, scaling), bbeta) * ((l4_p_l5) * -f4 * (f0 * f5 - f1 * f4 + f2 * f7 - f3 * f6));

                }

                if (comp == 2) {

                    fieldtt[comp] += -pow(pow(tau, scaling), bbeta) * ((l4_m_l5)*f6 * (f0 * f4 + f1 * f5 + f2 * f6 + f3 * f7));
                    fieldtt[comp] += -pow(pow(tau, scaling), bbeta) * ((l4_p_l5)*f7 * (f0 * f5 - f1 * f4 + f2 * f7 - f3 * f6));

                }

                if (comp == 3) {

                    fieldtt[comp] += -pow(pow(tau, scaling), bbeta) * ((l4_m_l5)*f7 * (f0 * f4 + f1 * f5 + f2 * f6 + f3 * f7));
                    fieldtt[comp] += -pow(pow(tau, scaling), bbeta) * ((l4_p_l5) * -f6 * (f0 * f5 - f1 * f4 + f2 * f7 - f3 * f6));

                }

                // Updates value of the field (2nd Order centralfinite difference method):
                fields[comp][totSize * tPast + i] = 2 * fields[comp][totSize * tNow + i] - fields[comp][totSize * tPast + i] + dt * dt * fieldtt[comp];

            }

            //Second Doublet:
            for (comp = 4; comp <= 7; comp++) {

                //2nd Spatial Deriviitive Calculations (central finite difference method) - to 4th Order:
                if (bc_type == "fixed" and (i_coord == 1 or i_coord == nx-2)) {

                    fieldxx = (fields[comp][totSize * tNow + ipx] - 2 * fields[comp][totSize * tNow + i] + fields[comp][totSize * tNow + imx]) / (dx * dx);

                } else {

                    fieldxx = (16 * (fields[comp][totSize * tNow + ipx] + fields[comp][totSize * tNow + imx]) - 30 * fields[comp][totSize * tNow + i] - fields[comp][totSize * tNow + ipxx] - fields[comp][totSize * tNow + imxx]) / (12 * dx * dx);

                }

                if (bc_type == "fixed" and (j_coord == 1 or j_coord == ny-2)) {

                    fieldyy = (fields[comp][totSize * tNow + ipy] - 2 * fields[comp][totSize * tNow + i] + fields[comp][totSize * tNow + imy]) / (dy * dy);

                } else {

                    fieldyy = (16 * (fields[comp][totSize * tNow + ipy] + fields[comp][totSize * tNow + imy]) - 30 * fields[comp][totSize * tNow + i] - fields[comp][totSize * tNow + ipyy] - fields[comp][totSize * tNow + imyy]) / (12 * dy * dy);

                }

                if (bc_type == "fixed" and (k_coord == 1 or k_coord == nz-2)) {

                    fieldzz = (fields[comp][totSize * tNow + ipz] - 2 * fields[comp][totSize * tNow + i] + fields[comp][totSize * tNow + imz]) / (dz * dz);

                } else {

                    fieldzz = (16 * (fields[comp][totSize * tNow + ipz] + fields[comp][totSize * tNow + imz]) - 30 * fields[comp][totSize * tNow + i] - fields[comp][totSize * tNow + ipzz] - fields[comp][totSize * tNow + imzz]) / (12 * dz * dz);

                }

                //1st Temporal Derivitive Calculation (only required for damping):
                fieldt[comp] = (fields[comp][totSize * tNow + i] - fields[comp][totSize * tPast + i]) / dt;

                //2nd Temporal Derivitive Calculation - Using EoM:
                fieldtt[comp] = fieldxx + fieldyy + fieldzz - fric * fieldt[comp] - pow(pow(tau, scaling), bbeta) * (-mu_2_sq * fields[comp][totSize * tNow + i] + lambda_3 * fields[comp][totSize * tNow + i] * (pow(f0, 2) + pow(f1, 2) + pow(f2, 2) + pow(f3, 2)) + 2 * lambda_2 * fields[comp][totSize * tNow + i] * (pow(f4, 2) + pow(f5, 2) + pow(f6, 2) + pow(f7, 2)));

                //Different potential contributions dependant upon the component of the field vector:
                if (comp == 4) {

                    fieldtt[comp] += -pow(pow(tau, scaling), bbeta) * ((l4_m_l5)*f0 * (f0 * f4 + f1 * f5 + f2 * f6 + f3 * f7));
                    fieldtt[comp] += -pow(pow(tau, scaling), bbeta) * ((l4_p_l5) * -f1 * (f0 * f5 - f1 * f4 + f2 * f7 - f3 * f6));

                }

                if (comp == 5) {

                    fieldtt[comp] += -pow(pow(tau, scaling), bbeta) * ((l4_m_l5)*f1 * (f0 * f4 + f1 * f5 + f2 * f6 + f3 * f7));
                    fieldtt[comp] += -pow(pow(tau, scaling), bbeta) * ((l4_p_l5)*f0 * (f0 * f5 - f1 * f4 + f2 * f7 - f3 * f6));

                }

                if (comp == 6) {

                    fieldtt[comp] += -pow(pow(tau, scaling), bbeta) * ((l4_m_l5)*f2 * (f0 * f4 + f1 * f5 + f2 * f6 + f3 * f7));
                    fieldtt[comp] += -pow(pow(tau, scaling), bbeta) * ((l4_p_l5) * -f3 * (f0 * f5 - f1 * f4 + f2 * f7 - f3 * f6));

                }

                if (comp == 7) {

                    fieldtt[comp] += -pow(pow(tau, scaling), bbeta) * ((l4_m_l5)*f3 * (f0 * f4 + f1 * f5 + f2 * f6 + f3 * f7));
                    fieldtt[comp] += -pow(pow(tau, scaling), bbeta) * ((l4_p_l5)*f2 * (f0 * f5 - f1 * f4 + f2 * f7 - f3 * f6));

                }

                // Updates value of the field (2nd Order centralfinite difference method):
                fields[comp][totSize * tPast + i] = 2 * fields[comp][totSize * tNow + i] - fields[comp][totSize * tPast + i] + dt * dt * fieldtt[comp];

            }






            // Calculate the energy contained in this process's domain
            if (calcEnergy) {

                for (comp = 0; comp < nb_fields; comp++) {
                    fieldx[comp] = (fields[comp][totSize * tNow + i] - fields[comp][totSize * tNow + imx]) / dx;
                    fieldy[comp] = (fields[comp][totSize * tNow + i] - fields[comp][totSize * tNow + imy]) / dy;
                    fieldz[comp] = (fields[comp][totSize * tNow + i] - fields[comp][totSize * tNow + imz]) / dz;

                    totalLocalEnergy += (pow(fieldx[comp], 2) + pow(fieldy[comp], 2) + pow(fieldz[comp], 2)) * dx * dy * dz;
                }

                //Potential Terms
                totalLocalEnergy += -mu_1_sq * (pow(f0, 2) + pow(f1, 2) + pow(f2, 2) + pow(f3, 2)) - mu_2_sq * (pow(f4, 2) + pow(f5, 2) + pow(f6, 2) + pow(f7, 2));
                totalLocalEnergy += lambda_1 * pow((pow(f0, 2) + pow(f1, 2) + pow(f2, 2) + pow(f3, 2)), 2);
                totalLocalEnergy += lambda_2 * pow((pow(f4, 2) + pow(f5, 2) + pow(f6, 2) + pow(f7, 2)), 2);
                totalLocalEnergy += lambda_3 * ((pow(f0, 2) + pow(f1, 2) + pow(f2, 2) + pow(f3, 2)) * (pow(f4, 2) + pow(f5, 2) + pow(f6, 2) + pow(f7, 2)));
                totalLocalEnergy += (l4_m_l5)*pow((f0 * f4 + f1 * f5 + f2 * f6 + f3 * f7), 2);
                totalLocalEnergy += (l4_p_l5)*pow((f0 * f5 - f1 * f4 + f2 * f7 - f3 * f6), 2);

            }

            // If the sign of phi flips between any two neighbours, consider that as a wall detection. Sum this up. Only look at forward neighbours so I'm not double counting.
            if (wallDetect) {

                double R1_i = 2 * (fields[0][totSize * tNow + i] * fields[4][totSize * tNow + i] + fields[1][totSize * tNow + i] * fields[5][totSize * tNow + i] + fields[2][totSize * tNow + i] * fields[6][totSize * tNow + i] + fields[3][totSize * tNow + i] * fields[7][totSize * tNow + i]);
                double R1_ipx = 2 * (fields[0][totSize * tNow + ipx] * fields[4][totSize * tNow + ipx] + fields[1][totSize * tNow + ipx] * fields[5][totSize * tNow + ipx] + fields[2][totSize * tNow + ipx] * fields[6][totSize * tNow + ipx] + fields[3][totSize * tNow + ipx] * fields[7][totSize * tNow + ipx]);
                double R1_ipy = 2 * (fields[0][totSize * tNow + ipy] * fields[4][totSize * tNow + ipy] + fields[1][totSize * tNow + ipy] * fields[5][totSize * tNow + ipy] + fields[2][totSize * tNow + ipy] * fields[6][totSize * tNow + ipy] + fields[3][totSize * tNow + ipy] * fields[7][totSize * tNow + ipy]);
                double R1_ipz = 2 * (fields[0][totSize * tNow + ipz] * fields[4][totSize * tNow + ipz] + fields[1][totSize * tNow + ipz] * fields[5][totSize * tNow + ipz] + fields[2][totSize * tNow + ipz] * fields[6][totSize * tNow + ipz] + fields[3][totSize * tNow + ipz] * fields[7][totSize * tNow + ipz]);

                double R1_imx = 2 * (fields[0][totSize * tNow + imx] * fields[4][totSize * tNow + imx] + fields[1][totSize * tNow + imx] * fields[5][totSize * tNow + imx] + fields[2][totSize * tNow + imx] * fields[6][totSize * tNow + imx] + fields[3][totSize * tNow + imx] * fields[7][totSize * tNow + imx]);
                double R1_imy = 2 * (fields[0][totSize * tNow + imy] * fields[4][totSize * tNow + imy] + fields[1][totSize * tNow + imy] * fields[5][totSize * tNow + imy] + fields[2][totSize * tNow + imy] * fields[6][totSize * tNow + imy] + fields[3][totSize * tNow + imy] * fields[7][totSize * tNow + imy]);
                double R1_imz = 2 * (fields[0][totSize * tNow + imz] * fields[4][totSize * tNow + imz] + fields[1][totSize * tNow + imz] * fields[5][totSize * tNow + imz] + fields[2][totSize * tNow + imz] * fields[6][totSize * tNow + imz] + fields[3][totSize * tNow + imz] * fields[7][totSize * tNow + imz]);



                // x neighbour
                if (R1_i * R1_ipx < 0) {

                    localNDW += 1;
                    localADW_simple += 2.0 * dy * dz / 3.0;

                    double R1x = (R1_i - R1_imx) / dx;
                    double R1y = (R1_i - R1_imy) / dy;
                    double R1z = (R1_i - R1_imz) / dz;
                    localADW_full += dy * dz * sqrt(pow(R1x, 2) + pow(R1y, 2) + pow(R1z, 2)) / (abs(R1x) + abs(R1y) + abs(R1z));
                }

                // y neighbour
                if (R1_i * R1_ipy < 0) {

                    localNDW += 1;
                    localADW_simple += 2.0 * dy * dz / 3.0;

                    double R1x = (R1_i - R1_imx) / dx;
                    double R1y = (R1_i - R1_imy) / dy;
                    double R1z = (R1_i - R1_imz) / dz;
                    localADW_full += dx * dz * sqrt(pow(R1x, 2) + pow(R1y, 2) + pow(R1z, 2)) / (abs(R1x) + abs(R1y) + abs(R1z));
                }

                // z neighbour
                if (R1_i * R1_ipz < 0) {

                    localNDW += 1;
                    localADW_simple += 2.0 * dy * dz / 3.0;

                    double R1x = (R1_i - R1_imx) / dx;
                    double R1y = (R1_i - R1_imy) / dy;
                    double R1z = (R1_i - R1_imz) / dz;
                    localADW_full += dx * dy * sqrt(pow(R1x, 2) + pow(R1y, 2) + pow(R1z, 2)) / (abs(R1x) + abs(R1y) + abs(R1z));
                }

            }

            if (monopoleDetect) {

                double R1_i = 2 * (fields[0][totSize * tNow + i] * fields[4][totSize * tNow + i] + fields[1][totSize * tNow + i] * fields[5][totSize * tNow + i] + fields[2][totSize * tNow + i] * fields[6][totSize * tNow + i] + fields[3][totSize * tNow + i] * fields[7][totSize * tNow + i]);
                double R1_ipx = 2 * (fields[0][totSize * tNow + ipx] * fields[4][totSize * tNow + ipx] + fields[1][totSize * tNow + ipx] * fields[5][totSize * tNow + ipx] + fields[2][totSize * tNow + ipx] * fields[6][totSize * tNow + ipx] + fields[3][totSize * tNow + ipx] * fields[7][totSize * tNow + ipx]);
                double R1_ipy = 2 * (fields[0][totSize * tNow + ipy] * fields[4][totSize * tNow + ipy] + fields[1][totSize * tNow + ipy] * fields[5][totSize * tNow + ipy] + fields[2][totSize * tNow + ipy] * fields[6][totSize * tNow + ipy] + fields[3][totSize * tNow + ipy] * fields[7][totSize * tNow + ipy]);
                double R1_ipz = 2 * (fields[0][totSize * tNow + ipz] * fields[4][totSize * tNow + ipz] + fields[1][totSize * tNow + ipz] * fields[5][totSize * tNow + ipz] + fields[2][totSize * tNow + ipz] * fields[6][totSize * tNow + ipz] + fields[3][totSize * tNow + ipz] * fields[7][totSize * tNow + ipz]);
                double R1_ipxpy = 2 * (fields[0][totSize * tNow + ipxpy] * fields[4][totSize * tNow + ipxpy] + fields[1][totSize * tNow + ipxpy] * fields[5][totSize * tNow + ipxpy] + fields[2][totSize * tNow + ipxpy] * fields[6][totSize * tNow + ipxpy] + fields[3][totSize * tNow + ipxpy] * fields[7][totSize * tNow + ipxpy]);
                double R1_ipxpz = 2 * (fields[0][totSize * tNow + ipxpz] * fields[4][totSize * tNow + ipxpz] + fields[1][totSize * tNow + ipxpz] * fields[5][totSize * tNow + ipxpz] + fields[2][totSize * tNow + ipxpz] * fields[6][totSize * tNow + ipxpz] + fields[3][totSize * tNow + ipxpz] * fields[7][totSize * tNow + ipxpz]);
                double R1_ipypz = 2 * (fields[0][totSize * tNow + ipypz] * fields[4][totSize * tNow + ipypz] + fields[1][totSize * tNow + ipypz] * fields[5][totSize * tNow + ipypz] + fields[2][totSize * tNow + ipypz] * fields[6][totSize * tNow + ipypz] + fields[3][totSize * tNow + ipypz] * fields[7][totSize * tNow + ipypz]);
                double R1_ipxpypz = 2 * (fields[0][totSize * tNow + ipxpypz] * fields[4][totSize * tNow + ipxpypz] + fields[1][totSize * tNow + ipxpypz] * fields[5][totSize * tNow + ipxpypz] + fields[2][totSize * tNow + ipxpypz] * fields[6][totSize * tNow + ipxpypz] + fields[3][totSize * tNow + ipxpypz] * fields[7][totSize * tNow + ipxpypz]);

                double R2_i = 2 * (fields[0][totSize * tNow + i] * fields[5][totSize * tNow + i] + fields[2][totSize * tNow + i] * fields[7][totSize * tNow + i] - fields[1][totSize * tNow + i] * fields[4][totSize * tNow + i] - fields[3][totSize * tNow + i] * fields[6][totSize * tNow + i]);
                double R2_ipx = 2 * (fields[0][totSize * tNow + ipx] * fields[5][totSize * tNow + ipx] + fields[2][totSize * tNow + ipx] * fields[7][totSize * tNow + ipx] - fields[1][totSize * tNow + ipx] * fields[4][totSize * tNow + ipx] - fields[3][totSize * tNow + ipx] * fields[6][totSize * tNow + ipx]);
                double R2_ipy = 2 * (fields[0][totSize * tNow + ipy] * fields[5][totSize * tNow + ipy] + fields[2][totSize * tNow + ipy] * fields[7][totSize * tNow + ipy] - fields[1][totSize * tNow + ipy] * fields[4][totSize * tNow + ipy] - fields[3][totSize * tNow + ipy] * fields[6][totSize * tNow + ipy]);
                double R2_ipz = 2 * (fields[0][totSize * tNow + ipz] * fields[5][totSize * tNow + ipz] + fields[2][totSize * tNow + ipz] * fields[7][totSize * tNow + ipz] - fields[1][totSize * tNow + ipz] * fields[4][totSize * tNow + ipz] - fields[3][totSize * tNow + ipz] * fields[6][totSize * tNow + ipz]);
                double R2_ipxpy = 2 * (fields[0][totSize * tNow + ipxpy] * fields[5][totSize * tNow + ipxpy] + fields[2][totSize * tNow + ipxpy] * fields[7][totSize * tNow + ipxpy] - fields[1][totSize * tNow + ipxpy] * fields[4][totSize * tNow + ipxpy] - fields[3][totSize * tNow + ipxpy] * fields[6][totSize * tNow + ipxpy]);
                double R2_ipxpz = 2 * (fields[0][totSize * tNow + ipxpz] * fields[5][totSize * tNow + ipxpz] + fields[2][totSize * tNow + ipxpz] * fields[7][totSize * tNow + ipxpz] - fields[1][totSize * tNow + ipxpz] * fields[4][totSize * tNow + ipxpz] - fields[3][totSize * tNow + ipxpz] * fields[6][totSize * tNow + ipxpz]);
                double R2_ipypz = 2 * (fields[0][totSize * tNow + ipypz] * fields[5][totSize * tNow + ipypz] + fields[2][totSize * tNow + ipypz] * fields[7][totSize * tNow + ipypz] - fields[1][totSize * tNow + ipypz] * fields[4][totSize * tNow + ipypz] - fields[3][totSize * tNow + ipypz] * fields[6][totSize * tNow + ipypz]);
                double R2_ipxpypz = 2 * (fields[0][totSize * tNow + ipxpypz] * fields[5][totSize * tNow + ipxpypz] + fields[2][totSize * tNow + ipxpypz] * fields[7][totSize * tNow + ipxpypz] - fields[1][totSize * tNow + ipxpypz] * fields[4][totSize * tNow + ipxpypz] - fields[3][totSize * tNow + ipxpypz] * fields[6][totSize * tNow + ipxpypz]);

                double R3_i = pow(fields[0][totSize * tNow + i], 2) + pow(fields[1][totSize * tNow + i], 2) + pow(fields[2][totSize * tNow + i], 2) + pow(fields[3][totSize * tNow + i], 2) - pow(fields[4][totSize * tNow + i], 2) - pow(fields[5][totSize * tNow + i], 2) - pow(fields[6][totSize * tNow + i], 2) - pow(fields[7][totSize * tNow + i], 2);
                double R3_ipx = pow(fields[0][totSize * tNow + ipx], 2) + pow(fields[1][totSize * tNow + ipx], 2) + pow(fields[2][totSize * tNow + ipx], 2) + pow(fields[3][totSize * tNow + ipx], 2) - pow(fields[4][totSize * tNow + ipx], 2) - pow(fields[5][totSize * tNow + ipx], 2) - pow(fields[6][totSize * tNow + ipx], 2) - pow(fields[7][totSize * tNow + ipx], 2);
                double R3_ipy = pow(fields[0][totSize * tNow + ipy], 2) + pow(fields[1][totSize * tNow + ipy], 2) + pow(fields[2][totSize * tNow + ipy], 2) + pow(fields[3][totSize * tNow + ipy], 2) - pow(fields[4][totSize * tNow + ipy], 2) - pow(fields[5][totSize * tNow + ipy], 2) - pow(fields[6][totSize * tNow + ipy], 2) - pow(fields[7][totSize * tNow + ipy], 2);
                double R3_ipz = pow(fields[0][totSize * tNow + ipz], 2) + pow(fields[1][totSize * tNow + ipz], 2) + pow(fields[2][totSize * tNow + ipz], 2) + pow(fields[3][totSize * tNow + ipz], 2) - pow(fields[4][totSize * tNow + ipz], 2) - pow(fields[5][totSize * tNow + ipz], 2) - pow(fields[6][totSize * tNow + ipz], 2) - pow(fields[7][totSize * tNow + ipz], 2);
                double R3_ipxpy = pow(fields[0][totSize * tNow + ipxpy], 2) + pow(fields[1][totSize * tNow + ipxpy], 2) + pow(fields[2][totSize * tNow + ipxpy], 2) + pow(fields[3][totSize * tNow + ipxpy], 2) - pow(fields[4][totSize * tNow + ipxpy], 2) - pow(fields[5][totSize * tNow + ipxpy], 2) - pow(fields[6][totSize * tNow + ipxpy], 2) - pow(fields[7][totSize * tNow + ipxpy], 2);
                double R3_ipxpz = pow(fields[0][totSize * tNow + ipxpz], 2) + pow(fields[1][totSize * tNow + ipxpz], 2) + pow(fields[2][totSize * tNow + ipxpz], 2) + pow(fields[3][totSize * tNow + ipxpz], 2) - pow(fields[4][totSize * tNow + ipxpz], 2) - pow(fields[5][totSize * tNow + ipxpz], 2) - pow(fields[6][totSize * tNow + ipxpz], 2) - pow(fields[7][totSize * tNow + ipxpz], 2);
                double R3_ipypz = pow(fields[0][totSize * tNow + ipypz], 2) + pow(fields[1][totSize * tNow + ipypz], 2) + pow(fields[2][totSize * tNow + ipypz], 2) + pow(fields[3][totSize * tNow + ipypz], 2) - pow(fields[4][totSize * tNow + ipypz], 2) - pow(fields[5][totSize * tNow + ipypz], 2) - pow(fields[6][totSize * tNow + ipypz], 2) - pow(fields[7][totSize * tNow + ipypz], 2);
                double R3_ipxpypz = pow(fields[0][totSize * tNow + ipxpypz], 2) + pow(fields[1][totSize * tNow + ipxpypz], 2) + pow(fields[2][totSize * tNow + ipxpypz], 2) + pow(fields[3][totSize * tNow + ipxpypz], 2) - pow(fields[4][totSize * tNow + ipxpypz], 2) - pow(fields[5][totSize * tNow + ipxpypz], 2) - pow(fields[6][totSize * tNow + ipxpypz], 2) - pow(fields[7][totSize * tNow + ipxpypz], 2);


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
            if (calcEnergy and wallDetect) { valsPerLoop << "Energy" << " " << "NDW" << " " << "ADW_Simple" << " " << "ADW_Full" << endl; }
            else {
                if (calcEnergy) { valsPerLoop << "Energy" << endl; }
                if (wallDetect) { valsPerLoop << "NDW" << " " << "ADW_Simple" << " " << "ADW_Full" << endl; }
            }
        }

        if (TimeStep == 0 and rank == 0) {
            if(monopoleDetect) { monopoleNumber << "NM" << endl; } 
            }

        if (rank == 0 && TimeStep == 0) {
            cout << "STEP 11: Starting main evolution loop" << endl;
        }

        // If calculating the energy, add it all up and output to text
        if (calcEnergy) {

            if (rank == 0) {

                double energy = totalLocalEnergy; // Initialise the energy as the energy in the domain of this process. Then add the energy in the regions of the other processes.

                for (i = 1; i < size; i++) { MPI_Recv(&totalLocalEnergy, 1, MPI_DOUBLE, i, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  energy += totalLocalEnergy; }

                valsPerLoop << energy << " ";

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

                valsPerLoop << NDW << " " << ADW_simple << " " << ADW_full;

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

        if (rank == 0 and monopoleDetect) { monopoleNumber << endl; }


        if (rank == 0 and (calcEnergy or wallDetect)) { valsPerLoop << endl; }


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
                finalFields << "R0" << " " << "R1" << " " << "R2" << " " << "R3" << " " << "R4" << " " << "R5" << " " << "n1" << " " << "n2" << " " << "n3" << endl;

                vector<vector<double>> fieldsOut(nb_fields, vector<double>(nPos, 0.0));
                double R0;
                double R1;
                double R2;
                double R3;
                double R4;
                double R5;

                double n1;
                double n2;
                double n3;


                for (comp = 0; comp < nb_fields; comp++) {

                    for (i = 0; i < coreSize; i++) { fieldsOut[comp][i] = fields[comp][frontHaloSize + i]; }

                    for (i = 1; i < size; i++) {

                        int localCoreStart;
                        int localCoreSize;
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


                    finalFields << R0 << " " << R1 << " " << R2 << " " << R3 << " " << R4 << " " << R5 << " " << n1 << " " << n2 << " " << n3 << endl;

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
                string TimeStepPath = dir_path + "fields_timestep=" + to_string(TimeStep) + outTag + ".txt";
                ofstream Gif(TimeStepPath.c_str());
                Gif << "R1" << " " << "R2" << " " << "R3" << endl;

                vector<vector<double>> fieldsOutnt(nb_fields, vector<double>(nPos, 0.0));
                double R1nt;
                double R2nt;
                double R3nt;

                for (comp = 0; comp < nb_fields; comp++) {

                    for (j = 0; j < coreSize; j++) { fieldsOutnt[comp][j] = fields[comp][frontHaloSize + j]; }

                    for (j = 1; j < size; j++) {

                        int localCoreStartnt;
                        int localCoreSizent;
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


        

        if (makeGif and TimeStep % saveFreq == 0) {

            if (rank == 0) {
                // Create files for fields and R values

                string rValuesPath = dir_path + "t_gamma=2pi_3_R_values_timestep=" + to_string(TimeStep) + outTag + ".txt";
                

                ofstream rValuesFile(rValuesPath.c_str());

                // Headers for fields and R values

                rValuesFile << "R1nt R2nt R3nt" << endl;

                vector<vector<double>> fieldsOutnt(nb_fields, vector<double>(nPos, 0.0));
                double R1nt, R2nt, R3nt;

                // Gather field data from all processes
                for (comp = 0; comp < nb_fields; comp++) {

                    for (j = 0; j < coreSize; j++) { fieldsOutnt[comp][j] = fields[comp][frontHaloSize + j]; }

                    for (j = 1; j < size; j++) {

                        int localCoreStartnt;
                        int localCoreSizent;
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
                    R3nt = pow(fieldsOutnt[0][j], 2) + pow(fieldsOutnt[1][j], 2) + pow(fieldsOutnt[2][j], 2) + pow(fieldsOutnt[3][j], 2)
                        - pow(fieldsOutnt[4][j], 2) - pow(fieldsOutnt[5][j], 2) - pow(fieldsOutnt[6][j], 2) - pow(fieldsOutnt[7][j], 2);

                    // Write R values to R values file, ensuring explicit output of 0.0
                    rValuesFile << (R1nt == 0.0 ? "0.0" : to_string(R1nt)) << " "
                                << (R2nt == 0.0 ? "0.0" : to_string(R2nt)) << " "
                                << (R3nt == 0.0 ? "0.0" : to_string(R3nt)) << endl;
                }



                rValuesFile.close();
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

        cout << "\rTimestep " << nt << " completed." << endl;
        cout << "STEP 21: Main evolution loop completed" << endl;

        gettimeofday(&end, NULL);

        cout << "Time taken: " << end.tv_sec - start.tv_sec << "s" << endl;
        cout << "STEP 22: Timing completed" << endl;

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