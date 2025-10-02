#include <iostream>
#include <math.h>
#include <fstream>
#include <iomanip>
#include <sys/time.h>
#include <omp.h>
#include <stdlib.h>

using namespace std;

// Calculating the profile functions for topological defects in the 2HDM using SOR.

const double pi = 4*atan(1);

const int nx = 10001;
const double h=0.01;
const double w=1.5; // Relaxation factor > 1. Trial and error to get better convergence.
const int maxIter = 10000000;
const double tol = 1e-3;
const int ic_type = 0;  // 0 is a guess at the solutions, 1 continues from a previous code output

// Model parameters using the physical parameters - masses and mixing angles.

const double Mh = 1; // First two can always be set to 1 by rescaling
const double vSM = 1;
const double alpha = 0.25*pi;
const double bbeta = 0.25*pi;
const double MH = 0;
const double MHpm = 1.0;

// Parameters that matter if I include an additional scalar

const double lambda_s = 1;
const double eta_s = 1;
const double beta1 = 0.1;
const double beta2 = 0.11;
const double beta3 = 0.09;

const double chiFix = 0;

const bool qFixed = false; // Fixes the charge per unit length rather than chi when solving for the solution
const double q = 1;
const double k = 0;

const int n = 1; 

// Either U(1) or SO(3) for 2HDM defects. NO or tHP for Nielson-Oleson vortices or t'Hooft-Polyakov monopoles respectively. Put glob in front for global model (not working for U(1) yet) U(1) alt for my ansatz
// Other options are "glob U(1) w cond" or "glob SO(3) w cond" for U(1) or SO(3) 2HDM with an attempted ansatz for a charge breaking condensate
// New option is to have "extra scalar" which is like the glob U(1) w cond but now there is an additional complex scalar charged under U(1)_PQ only
const string symType = "glob SO(3) w cond";
const double vacAng = 0;//pi/4.0; // Used with glob U(1) w cond setting. Sets angle between vacuum being used and the vacuum where only bottom of both doublets is non-zero.
const bool switchOffConds = false; // Sets the two condensate type fields to zero, everywhere, initially so they should never become non-zero and should regain bare string sol.
const bool altU1 = false; // Trys an alternate U(1) condensate ansatz
const string BC = "neumann"; // fixed or Neumann

// const bool stringInf1 = false; // For use with U(1) condensate. True if the winding is in f1 (and g2) or false if it's in f2 (and g1)

const bool calcEnergy = false;

int main(){

	if(symType!="U(1)" and symType!="SO(3)" and symType!="NO" and symType!="tHP" and symType!="glob U(1) w cond" and symType!="glob SO(3) w cond" and symType!="U(1) alt" and
	   symType!="glob SO(3)" and symType!="glob U(1) w cond alt" and symType!="extra scalar"){ cout << "Error: Incorrect symmetry type specified." << endl; }
	// cout << "Lambda: " << lambda << endl;

	struct timeval start, end;
    gettimeofday(&start, NULL);

    string file_path = __FILE__;
    string dir_path = file_path.substr(0,file_path.find_last_of('/'));

    string valsPath = dir_path + "/Data/singleVals.txt";
    string SOR_FieldsPath = dir_path + "/Data/SOR_Fields.txt";
    string icPath = dir_path + "/Data/SOR_Fields.txt";
    //string aniPath = dir_path + "/ani.txt";
    //string icPath = dir_path + "/SavedData/P4_nx10001_h0005_chi04338.txt";

    ifstream ic (icPath.c_str());

    // Initialise the field arrays

	double f1[nx], f2[nx], h1[nx], h2[nx], Ff1[nx][2], Ff2[nx][2], Fh1[nx][2], Fh2[nx][2], tolTest, E, v1, v2, Sigma_2, chi, phi[2][nx], Fphi[2][nx][2], vs;

	int i,j,n1,n2;
	long long int iterNum;

	// if(stringInf1){ n1 = n; n2 = 0; }
	// else{ n1 = 0; n2 = 1; }

	// Calculate the original model parameters:
	double mu_1 = 0.5*( pow(Mh*cos(alpha),2) + pow(MH*sin(alpha),2) + (Mh*Mh - MH*MH)*tan(bbeta)*cos(alpha)*sin(alpha) ); // Technically mu_1^2
	double mu_2 = 0.5*( pow(Mh*sin(alpha),2) + pow(MH*cos(alpha),2) + (Mh*Mh - MH*MH)*(1/tan(bbeta))*cos(alpha)*sin(alpha) ); // ^^
	double lambda_1 = ( pow(Mh*cos(alpha),2) + pow(MH*sin(alpha),2) )/( 2*pow(vSM*cos(bbeta),2) );
	double lambda_2 = ( pow(Mh*sin(alpha),2) + pow(MH*cos(alpha),2) )/( 2*pow(vSM*sin(bbeta),2) );
	double lambda_3 = ( (Mh*Mh - MH*MH)*cos(alpha)*sin(alpha) + 2*MHpm*MHpm*sin(bbeta)*cos(bbeta) )/( vSM*vSM*sin(bbeta)*cos(bbeta) );
	double lambda_4 = -2*pow(MHpm/vSM,2);



	if(ic_type==0){

		// Determine the vacuum values of the fields
		if(symType=="U(1)" or symType=="glob U(1) w cond" or symType=="U(1) alt" or symType=="glob U(1) w cond alt"){

			if(4*lambda_1*lambda_2==pow(lambda_3+lambda_4,2)){ v1 = 0.5; v2 = 0.5;} // Just set vacuum to 1/sqrt(2) for both as an initial (probably bad) guess. May need to alter this in future
			else{

				v1 = ( 4*lambda_2*mu_1 - 2*(lambda_3+lambda_4)*mu_1 )/( 4*lambda_1*lambda_2 - pow(lambda_3+lambda_4,2) );
				v2 = ( 4*lambda_1*mu_2 - 2*(lambda_3+lambda_4)*mu_2 )/( 4*lambda_1*lambda_2 - pow(lambda_3+lambda_4,2) );

				if(v1<0){ v1 = 0; v2 = mu_2/lambda_2; }
				else if(v2<0){ v1 = mu_1/lambda_1; v2 = 0; }

				cout << v1 << " " << v2 << endl;

			}

		}

		if(symType=="extra scalar"){

			// NEED TO DO THE NEW CALCULATION OF THE VEVs
			// For now just putting this guess in
			v1 = ( 4*lambda_2*mu_1 - 2*(lambda_3+lambda_4)*mu_1 )/( 4*lambda_1*lambda_2 - pow(lambda_3+lambda_4,2) );
			v2 = ( 4*lambda_1*mu_2 - 2*(lambda_3+lambda_4)*mu_2 )/( 4*lambda_1*lambda_2 - pow(lambda_3+lambda_4,2) );
			vs = eta_s;


		}

		for(i=0;i<nx;i++){

			if(symType=="U(1)" or symType=="glob U(1) w cond" or symType=="U(1) alt" or symType=="glob U(1) w cond alt" or symType=="extra scalar"){

				// Leave the gauge fields as initialised to zero

				f1[i] = sqrt(v1)*tanh(h*i);
				f2[i] = sqrt(v2)*tanh(h*i);

				if(symType=="glob U(1) w cond" or symType=="extra scalar"){

					if(altU1){

						f1[i] = sqrt(v1);
						h1[i] = 0;
						h2[i] = sqrt(0.5)/cosh(0.5*h*i);

					} else{

						f1[i] = sqrt(v1)*( cos(vacAng) + sqrt(0.5)/cosh(0.5*h*i) );
						f2[i] = sqrt(v2)*cos(vacAng)*tanh(h*i);
						h1[i] = -sqrt(v1)*sin(vacAng)*tanh(h*i);
						h2[i] = sqrt(v2)*( -sin(vacAng) + sqrt(0.5)/cosh(0.5*h*i) );

						// if(stringInf1){

						// 	h1[i] = sqrt(0.5)/cosh(0.5*h*i);
						// 	h2[i] = 0;

						// } else{

						// 	h1[i] = 0;
						// 	h2[i] = sqrt(0.5)/cosh(0.5*h*i);

						// }

						// if(switchOffConds){ h1[i] = 0; h2[i] = 0; }

					}

					if(symType=="extra scalar"){

						if (beta3>=0){

							phi[0][i] = -sqrt(vs)*tanh(h*i);
							phi[1][i] = 0;

						} else{

							phi[0][i] = sqrt(vs)*tanh(h*i);
							phi[1][i] = 0;

						}

					}


				} else if(symType=="glob U(1) w cond alt"){

					// h1 is f_+ and h2 is gamma

					h1[i] = 0;
					h2[i] = vacAng*tanh(h*i);

				}	else{

					h1[i] = 0;
					h2[i] = 0;

				}

			} else{

				f1[i] = tanh(h*i);
				if(symType=="glob SO(3) w cond"){

					f1[i] += sqrt(0.5)/cosh(0.5*h*i);
					h1[i] = sqrt(0.5)/cosh(0.5*h*i);


					// h1[i] = 1/cosh(0.5*h*i); // Old version

				} else{ h1[i] = 0; }

				// The other two will be unused in this case

			}

		}

	} else if(ic_type==1){

		for(i=0;i<nx;i++){

			if(symType=="extra scalar"){

				ic >> f1[i] >> f2[i] >> h1[i] >> h2[i] >> phi[0][i] >> phi[1][i];

			} else if(symType=="U(1)" or symType=="glob U(1) w cond" or symType=="glob U(1) w cond alt"){

				ic >> f1[i] >> f2[i] >> h1[i] >> h2[i];

			} else if(symType=="U(1) alt"){

				ic >> f1[i] >> f2[i] >> h1[i];

			} else{

				ic >> f1[i] >> h1[i];

			}

		}

	}

    ofstream vals (valsPath.c_str());
    ofstream SOR_Fields (SOR_FieldsPath.c_str());


	tolTest = 1;
	iterNum = 0;
	while(tolTest>tol and iterNum<maxIter){

		double prevTolTest = tolTest;
		tolTest = 0;
		iterNum += 1;

		if((1000*iterNum)%maxIter==0){

			cout << "\rIteration number: " << iterNum << ", tol: " << prevTolTest;
			cout << flush;

		}

		if(iterNum==maxIter){

			cout << "\rMaximum number of iterations reached" << flush;

		}

		// Set boundary conditions

		if(BC=="neumann"){

			f1[nx-1] = f1[nx-2];
			f2[nx-1] = f2[nx-2];
			h1[nx-1] = h1[nx-2];
			h2[nx-1] = h2[nx-2];
			phi[0][nx-1] = phi[0][nx-2];
			phi[1][nx-1] = phi[1][nx-2];

		} else if(BC=="fixed"){

			f1[nx-1] = sqrt(v1)*cos(vacAng);
			f2[nx-1] = sqrt(v2)*cos(vacAng);
			h1[nx-1] = -sqrt(v1)*sin(vacAng);
			h2[nx-1] = -sqrt(v2)*sin(vacAng);

		}

		//f1[nx-1] = 1;
		//f2[nx-1] = 0;

		if(symType=="U(1)"){

			// Calculate next iteration of f1[0] using boundary conditions f1'(r=0) = 0  and all other fields are zero here.

			Ff1[0][0] = 2*(f1[1] - f1[0])/(h*h) - f1[0]*(lambda_1*pow(f1[0],2) - mu_1);
			Ff1[0][1] = -2/(h*h) - 3*lambda_1*pow(f1[0],2) - mu_1;

			f1[0] = f1[0] - w*Ff1[0][0]/Ff1[0][1];


			if(abs(Ff1[0][0])>tolTest){ tolTest = abs(Ff1[0][0]); }


		}else if(symType=="glob U(1) w cond" or symType=="extra scalar"){

			// Overule derivative boundary conditions by fixing fields to the vacuum values at end of array
			//f1[nx-1] = sqrt(v1);
			//f2[nx-1] = sqrt(v2);
			//h1[nx-1] = 0;
			//h2[nx-1] = 0;

			if(qFixed){

				Sigma_2 = 0;
				for(i=1;i<nx;i++){ Sigma_2 += 2*pi*h*i*h*(pow(h1[i],2)+pow(h2[i],2)); }

				chi = pow(q/Sigma_2,2) - pow(k,2);

			} else{ chi = chiFix; }

			Ff1[0][0] = 2*(f1[1] - f1[0])/(h*h) - f1[0]*(lambda_1*pow(f1[0],2) - mu_1 + 0.5*lambda_3*pow(h2[0],2));
			Ff1[0][1] = -2/(h*h) - 3*lambda_1*pow(f1[0],2) + mu_1 - 0.5*lambda_3*pow(h2[0],2);

			Fh2[0][0] = 2*(h2[1] - h2[0])/(h*h) - h2[0]*(lambda_2*pow(h2[0],2) - mu_2 + 0.5*lambda_3*pow(f1[0],2) - chi);
			Fh2[0][1] = -2/(h*h) - 3*lambda_2*pow(h2[0],2) + mu_2 - 0.5*lambda_3*pow(f1[0],2) + chi;

			f1[0] += -w*Ff1[0][0]/Ff1[0][1];
			h2[0] += -w*Fh2[0][0]/Fh2[0][1];

			// if(altU1){

			// 	Ff1[0][0] = 2*(f1[1] - f1[0])/(h*h) - f1[0]*(zeta*pow(f1[0],2) - eta + 0.5*zeta3*pow(h2[0],2));
			// 	Ff1[0][1] = -2/(h*h) - 3*zeta*pow(f1[0],2) + eta - 0.5*zeta3*pow(h2[0],2);

			// 	Fh2[0][0] = 2*(h2[1] - h2[0])/(h*h) - h2[0]*(pow(h2[0],2) - 1 + 0.5*zeta3*pow(f1[0],2));
			// 	Fh2[0][1] = -2/(h*h) - 3*pow(h2[0],2) + 1 - 0.5*zeta3*pow(f1[0],2);

			// 	f1[0] += -w*Ff1[0][0]/Ff1[0][1];
			// 	h2[0] += -w*Fh2[0][0]/Fh2[0][1];

			// } else{

			// 	if(stringInf1){

			// 		Ff2[0][0] = 2*(f2[1] - f2[0])/(h*h) - f2[0]*(pow(f2[0],2) - 1 + 0.5*zeta3*pow(h1[0],2));
			// 		Ff2[0][1] = -2/(h*h) - 3*pow(f2[0],2) - 1 + 0.5*zeta3*pow(h1[0],2);

			// 		Fh1[0][0] = 2*(h1[1] - h1[0])/(h*h) - h1[0]*(zeta*pow(h1[0],2) - eta + 0.5*zeta3*pow(f2[0],2) - chi);
			// 		Fh1[0][1] = -2/(h*h) - 3*zeta*pow(h1[0],2) - eta + 0.5*zeta3*pow(f2[0],2) - chi;

			// 		f2[0] = f2[0] - w*Ff2[0][0]/Ff2[0][1];
			// 		h1[0] = h1[0] - w*Fh1[0][0]/Fh1[0][1];

			// 	}else{

			// 		// Need to do the same for f1 and h2 (h corresponds to condensate, not gauge fields here)

			// 		Ff1[0][0] = 2*(f1[1] - f1[0])/(h*h) - f1[0]*(zeta*pow(f1[0],2) - eta + 0.5*zeta3*pow(h2[0],2));
			// 		Ff1[0][1] = -2/(h*h) - 3*zeta*pow(f1[0],2) - eta + 0.5*zeta3*pow(h2[0],2);

			// 		Fh2[0][0] = 2*(h2[1] - h2[0])/(h*h) - h2[0]*(pow(h2[0],2) - 1 + 0.5*zeta3*pow(f1[0],2) - chi);
			// 		Fh2[0][1] = -2/(h*h) - 3*pow(h2[0],2) - 1 + 0.5*zeta3*pow(f1[0],2) - chi;

			// 		f1[0] = f1[0] - w*Ff1[0][0]/Ff1[0][1];
			// 		h2[0] = h2[0] - w*Fh2[0][0]/Fh2[0][1];

			// 	}

			// }

		}else if(symType=="glob U(1) w cond alt"){

			// Ff1[0][0] = 2*(f1[1] - f1[0])/(h*h) - f1[0]*(zeta*pow(f1[0],2) - eta + 0.5*zeta3*pow(h1[0],2) + pow(h2[1]/h,2));
			// Ff1[0][1] = -2/(h*h) - 3*zeta*pow(f1[0],2) + eta - 0.5*zeta3*pow(h2[0],2) - pow(h2[1]/h,2);

			// Fh1[0][0] = 2*(h1[1] - h1[0])/(h*h) - h1[0]*(pow(h1[0],2) - 1 + 0.5*zeta3*pow(f1[0],2) + pow(h2[1]/h,2));
			// Fh1[0][1] = -2/(h*h) - 3*pow(h2[0],2) + 1 - 0.5*zeta3*pow(f1[0],2);

			// f1[0] += -w*Ff1[0][0]/Ff1[0][1];
			// h1[0] += -w*Fh1[0][0]/Fh1[0][1];

		}else if(symType=="glob SO(3) w cond"){

			f1[0] = 0.5*(f1[1]+h1[1]);
			h1[0] = f1[0];

		} // Otherwise f1 is fixed at 0 at r=0.


		// Looping over all other positions in grid
		# pragma omp parallel for default(none) shared(f1,f2,h1,h2,tolTest,Ff1,Ff2,Fh1,Fh2,symType,n1,n2,chi,phi,Fphi,mu_1,mu_2,lambda_1,lambda_2,lambda_3,lambda_4)
		for(j=1;j<nx-1;j++){

			if(symType=="U(1)"){

				// Ff1[j][0] = (f1[j+1] - 2*f1[j] + f1[j-1])/(h*h) + (f1[j+1] - f1[j-1])/(2*j*h*h) - pow( n*(h1[j]-h2[j])/(2*j*h) ,2)*f1[j] - lambda*f1[j]*( zeta*pow(f1[j],2) - eta + alpha*pow(f2[j],2) );
				// Ff1[j][1] = -2/(h*h) - pow( n*(h1[j]-h2[j])/(2*j*h) ,2) - lambda*( 3*zeta*pow(f1[j],2) - eta + alpha*pow(f2[j],2) );

				// Ff2[j][0] = (f2[j+1] - 2*f2[j] + f2[j-1])/(h*h) + (f2[j+1] - f2[j-1])/(2*j*h*h) - pow( n*(h1[j]+h2[j]-2)/(2*j*h) ,2)*f2[j] - lambda*f2[j]*( pow(f2[j],2) - 1 + alpha*pow(f1[j],2) );
				// Ff2[j][1] = -2/(h*h) - pow( n*(h1[j]+h2[j]-2)/(2*j*h) ,2) - lambda*( 3*pow(f2[j],2) - 1 + alpha*pow(f1[j],2) );

				// Fh1[j][0] = (h1[j+1] - 2*h1[j] + h1[j-1])/(h*h) - (h1[j+1] - h1[j-1])/(2*j*h*h) - ( pow(f1[j],2)*(h1[j]-h2[j]) + pow(f2[j],2)*(h1[j]+h2[j]-2) )/(4*g);
				// Fh1[j][1] = -2/(h*h) - ( pow(f1[j],2) + pow(f2[j],2) )/(4*g);

				// Fh2[j][0] = (h2[j+1] - 2*h2[j] + h2[j-1])/(h*h) - (h2[j+1] - h2[j-1])/(2*j*h*h) + ( pow(f1[j],2)*(h1[j]-h2[j]) - pow(f2[j],2)*(h1[j]+h2[j]-2) )/4;
				// Fh2[j][1] = -2/(h*h) - ( pow(f1[j],2) + pow(f2[j],2) )/4;

				// // Update with SOR

				// f1[j] = f1[j] - w*Ff1[j][0]/Ff1[j][1];
				// f2[j] = f2[j] - w*Ff2[j][0]/Ff2[j][1];
				// h1[j] = h1[j] - w*Fh1[j][0]/Fh1[j][1];
				// h2[j] = h2[j] - w*Fh2[j][0]/Fh2[j][1];

			} else if(symType=="SO(3)" or symType=="glob SO(3)"){

				// Ff1[j][0] = (f1[j+1] - 2*f1[j] + f1[j-1])/(h*h) + (f1[j+1] - f1[j-1])/(j*h*h) - 2*f1[j]*pow( (1-h1[j])/(j*h) ,2) - lambda*f1[j]*(pow(f1[j],2) - 1);
				// Ff1[j][1] = -2/(h*h) - 2*pow( (1-h1[j])/(j*h) ,2) - lambda*(3*pow(f1[j],2) - 1);

				// if(symType=="SO(3)"){

				// 	Fh1[j][0] = (h1[j+1] - 2*h1[j] + h1[j-1])/(h*h) + (1-h1[j])*pow(f1[j],2)/4 - 2*h1[j]*(1-h1[j])*(1-2*h1[j])/pow(j*h,2);
				// 	Fh1[j][1] = -2/(h*h) - pow(f1[j],2)/4 - ( 12*h1[j]*(h1[j]-1) + 2 )/pow(j*h,2);

				// }

				// // Update with SOR

				// f1[j] = f1[j] - w*Ff1[j][0]/Ff1[j][1];
				// if(symType=="SO(3)"){ h1[j] = h1[j] - w*Fh1[j][0]/Fh1[j][1]; }

			} else if(symType=="NO"){

				// Ff1[j][0] = (f1[j+1] - 2*f1[j] + f1[j-1])/(h*h) + (f1[j+1] - f1[j-1])/(2*j*h*h) - pow( n*(1-h1[j])/(j*h) ,2)*f1[j] - lambda*f1[j]*(pow(f1[j],2) - 1);
				// Ff1[j][1] = -2/(h*h) - pow( n*(1-h1[j])/(j*h) ,2) - lambda*(3*pow(f1[j],2) - 1);

				// Fh1[j][0] = (h1[j+1] - 2*h1[j] + h1[j-1])/(h*h) - (h1[j+1] - h1[j-1])/(2*j*h*h) + (1-h1[j])*pow(f1[j],2);
				// Fh1[j][1] = -2/(h*h) - pow(f1[j],2);

				// // Update with SOR

				// f1[j] = f1[j] - w*Ff1[j][0]/Ff1[j][1];
				// h1[j] = h1[j] - w*Fh1[j][0]/Fh1[j][1]; 

			} else if(symType=="tHP"){

				// Ff1[j][0] = (f1[j+1] - 2*f1[j] + f1[j-1])/(h*h) + (f1[j+1] - f1[j-1])/(j*h*h) - 2*f1[j]*pow( (1-h1[j])/(j*h) ,2) - lambda*f1[j]*(pow(f1[j],2) - 1);
				// Ff1[j][1] = -2/(h*h) - 2*pow( (1-h1[j])/(j*h) ,2) - lambda*(3*pow(f1[j],2) - 1);

				// Fh1[j][0] = (h1[j+1] - 2*h1[j] + h1[j-1])/(h*h) + (1-h1[j])*pow(f1[j],2) - h1[j]*(2-h1[j])*(1-h1[j])/pow(j*h,2);
				// Fh1[j][1] = -2/(h*h) - pow(f1[j],2) - ( 3*h1[j]*(h1[j]-2) + 2 )/pow(j*h,2);

				// // Update with SOR

				// f1[j] = f1[j] - w*Ff1[j][0]/Ff1[j][1];
				// h1[j] = h1[j] - w*Fh1[j][0]/Fh1[j][1];

			} else if(symType=="glob U(1) w cond"){

				if(altU1){

					// double mod1 = pow(f1[j],2);
					// double mod2 = pow(f2[j],2) + pow(h2[j],2);

					// Ff1[j][0] = (f1[j+1] - 2*f1[j] + f1[j-1])/(h*h) + (f1[j+1] - f1[j-1])/(2*j*h*h) - f1[j]*( zeta*mod1 - eta + 0.5*zeta3*mod2 + 0.5*zeta4*pow(f2[j],2) );
					// Ff1[j][1] = -2/(h*h) - 3*zeta*mod1 + eta - 0.5*zeta3*mod2 - 0.5*zeta4*pow(f2[j],2);

					// Ff2[j][0] = (f2[j+1] - 2*f2[j] + f2[j-1])/(h*h) + (f2[j+1] - f2[j-1])/(2*j*h*h) - pow(n/(j*h),2)*f2[j] - f2[j]*( mod2 - 1 + 0.5*zeta3*mod1 + 0.5*zeta4*mod1 );
					// Ff2[j][1] = -2/(h*h) - pow(n/(j*h),2) - 3*pow(f2[j],2) - pow(h2[j],2) + 1 - 0.5*zeta3*mod1 - 0.5*zeta4*mod1;

					// Fh2[j][0] = (h2[j+1] - 2*h2[j] + h2[j-1])/(h*h) + (h2[j+1] - h2[j-1])/(2*j*h*h) - h2[j]*( mod2 - 1 + 0.5*zeta3*mod1 );
					// Fh2[j][1] = -2/(h*h) - 3*pow(h2[j],2) - pow(f2[j],2) + 1 - 0.5*zeta3*mod1;

					// Fh1[j][0] = 0; // Just to satisfy tolerance tests

					// // Update with SOR

					// f1[j] += -w*Ff1[j][0]/Ff1[j][1];
					// f2[j] += -w*Ff2[j][0]/Ff2[j][1];
					// h2[j] += -w*Fh2[j][0]/Fh2[j][1];

				} else{

					// Actually these are magnitude squared

					double mod1 = pow(f1[j],2) + pow(h1[j],2);
					double mod2 = pow(f2[j],2) + pow(h2[j],2);
					double mix12 = f1[j]*f2[j] + h1[j]*h2[j];

					Ff1[j][0] = (f1[j+1] - 2*f1[j] + f1[j-1])/(h*h) + (f1[j+1] - f1[j-1])/(2*j*h*h) - f1[j]*( lambda_1*mod1 - mu_1 + 0.5*lambda_3*mod2 ) - 0.5*lambda_4*mix12*f2[j];
					Ff1[j][1] = -2/(h*h) - ( lambda_1*(2*pow(f1[j],2)+mod1) - mu_2 + 0.5*lambda_3*mod2 ) - 0.5*lambda_4*pow(f2[j],2);

					Ff2[j][0] = (f2[j+1] - 2*f2[j] + f2[j-1])/(h*h) + (f2[j+1] - f2[j-1])/(2*j*h*h) - pow(n/(j*h),2)*f2[j] - f2[j]*( lambda_2*mod2 - mu_2 + 0.5*lambda_3*mod1 ) - 0.5*lambda_4*mix12*f1[j];
					Ff2[j][1] = -2/(h*h) - pow(n/(j*h),2) - ( lambda_2*(2*pow(f2[j],2)+mod2) - mu_2 + 0.5*lambda_3*mod1 ) - 0.5*lambda_4*pow(f1[j],2);

					Fh1[j][0] = (h1[j+1] - 2*h1[j] + h1[j-1])/(h*h) + (h1[j+1] - h1[j-1])/(2*j*h*h) - pow(n/(j*h),2)*h1[j] - h1[j]*( lambda_1*mod1 - mu_1 + 0.5*lambda_3*mod2 - chi ) - 0.5*lambda_4*mix12*h2[j];
					Fh1[j][1] = -2/(h*h) - pow(n/(j*h),2) - ( lambda_1*(mod1+2*pow(h1[j],2)) - mu_1 + 0.5*lambda_3*mod2 - chi ) - 0.5*lambda_4*pow(h2[j],2);

					Fh2[j][0] = (h2[j+1] - 2*h2[j] + h2[j-1])/(h*h) + (h2[j+1] - h2[j-1])/(2*j*h*h) - h2[j]*( lambda_2*mod2 - mu_2 + 0.5*lambda_3*mod1 - chi ) - 0.5*lambda_4*mix12*h1[j];
					Fh2[j][1] = -2/(h*h) - ( lambda_2*(mod2+2*pow(h2[j],2)) - mu_2 + 0.5*lambda_3*mod1 - chi ) - 0.5*lambda_4*pow(h1[j],2);


					// Update with SOR

					f1[j] = f1[j] - w*Ff1[j][0]/Ff1[j][1];
					f2[j] = f2[j] - w*Ff2[j][0]/Ff2[j][1];
					h1[j] = h1[j] - w*Fh1[j][0]/Fh1[j][1];
					h2[j] = h2[j] - w*Fh2[j][0]/Fh2[j][1];

				}

			} else if(symType=="extra scalar"){

				// Actually these are magnitude squared

				// double mod1 = pow(f1[j],2) + pow(h1[j],2);
				// double mod2 = pow(f2[j],2) + pow(h2[j],2);
				// double mix12 = f1[j]*f2[j] + h1[j]*h2[j];
				// double mods = pow(phi[0][j],2) + pow(phi[1][j],2);

				// Ff1[j][0] = (f1[j+1] - 2*f1[j] + f1[j-1])/(h*h) + (f1[j+1] - f1[j-1])/(2*j*h*h) - f1[j]*( zeta*mod1 - eta + 0.5*zeta3*mod2 + 0.5*beta1*mods ) - ( 0.5*zeta4*mix12 + 0.5*beta3*phi[0][j] )*f2[j];
				// Ff1[j][1] = -2/(h*h) - ( zeta*(2*pow(f1[j],2)+mod1) - eta + 0.5*zeta3*mod2 + 0.5*beta1*mods ) - 0.5*zeta4*pow(f2[j],2);

				// Ff2[j][0] = (f2[j+1] - 2*f2[j] + f2[j-1])/(h*h) + (f2[j+1] - f2[j-1])/(2*j*h*h) - pow(n/(j*h),2)*f2[j] - f2[j]*( mod2 - 1 + 0.5*zeta3*mod1 + 0.5*beta2*mods ) - ( 0.5*zeta4*mix12 + 0.5*beta3*phi[0][j] )*f1[j];
				// Ff2[j][1] = -2/(h*h) - pow(n/(j*h),2) - ( 2*pow(f2[j],2)+mod2 - 1 + 0.5*zeta3*mod1 + 0.5*beta2*mods ) - 0.5*zeta4*pow(f1[j],2);

				// Fh1[j][0] = (h1[j+1] - 2*h1[j] + h1[j-1])/(h*h) + (h1[j+1] - h1[j-1])/(2*j*h*h) - pow(n/(j*h),2)*h1[j] - h1[j]*( zeta*mod1 - eta + 0.5*zeta3*mod2 - chi + 0.5*beta1*mods ) - ( 0.5*zeta4*mix12 + 0.5*beta3*phi[0][j] )*h2[j];
				// Fh1[j][1] = -2/(h*h) - pow(n/(j*h),2) - ( zeta*(mod1+2*pow(h1[j],2)) - eta + 0.5*zeta3*mod2 - chi + 0.5*beta1*mods ) - 0.5*zeta4*pow(h2[j],2);

				// Fh2[j][0] = (h2[j+1] - 2*h2[j] + h2[j-1])/(h*h) + (h2[j+1] - h2[j-1])/(2*j*h*h) - h2[j]*( mod2 - 1 + 0.5*zeta3*mod1 - chi + 0.5*beta2*mods ) - ( 0.5*zeta4*mix12 + 0.5*beta3*phi[0][j] )*h1[j];
				// Fh2[j][1] = -2/(h*h) - ( mod2+2*pow(h2[j],2) - 1 + 0.5*zeta3*mod1 - chi + 0.5*beta1*mods ) - 0.5*zeta4*pow(h1[j],2);

				// Fphi[0][j][0] = (phi[0][j+1] - 2*phi[0][j] + phi[0][j-1])/(h*h) + (phi[0][j+1] - phi[0][j-1])/(2*j*h*h) - pow(n/(j*h),2)*phi[0][j] - phi[0][j]*( lambda_s*(mods-eta_s) + 0.5*beta1*mod1 + 0.5*beta2*mod2 + 0.5*beta3*mix12 );
				// Fphi[0][j][1] = -2/(h*h) - pow(n/(j*h),2) - ( lambda_s*(3*pow(phi[0][j],2)+mods-eta_s) + 0.5*beta1*mod1 + 0.5*beta2*mod2 + 0.5*beta3*mix12 );

				// Fphi[1][j][0] = (phi[1][j+1] - 2*phi[1][j] + phi[1][j-1])/(h*h) + (phi[1][j+1] - phi[1][j-1])/(2*j*h*h) - pow(n/(j*h),2)*phi[1][j] - phi[1][j]*( lambda_s*(mods-eta_s) + 0.5*beta1*mod1 + 0.5*beta2*mod2 );
				// Fphi[1][j][1] = -2/(h*h) - pow(n/(j*h),2) - ( lambda_s*(3*pow(phi[1][j],2)+mods-eta_s) + 0.5*beta1*mod1 + 0.5*beta2*mod2 );


				// // Update with SOR

				// f1[j] = f1[j] - w*Ff1[j][0]/Ff1[j][1];
				// f2[j] = f2[j] - w*Ff2[j][0]/Ff2[j][1];
				// h1[j] = h1[j] - w*Fh1[j][0]/Fh1[j][1];
				// h2[j] = h2[j] - w*Fh2[j][0]/Fh2[j][1];
				// phi[0][j] = phi[0][j] - w*Fphi[0][j][0]/Fphi[0][j][1];
				// phi[1][j] = phi[1][j] - w*Fphi[1][j][0]/Fphi[1][j][1];

			} else if(symType=="glob SO(3) w cond"){

				Ff1[j][0] = (f1[j+1] - 2*f1[j] + f1[j-1])/(h*h) + (f1[j+1] - f1[j-1])/(j*h*h) - (f1[j]-h1[j])/pow(j*h,2) - ( lambda_1*(pow(f1[j],2) + pow(h1[j],2)) - mu_1 - 0.5*lambda_4*pow(h1[j],2) )*f1[j];
				Ff1[j][1] = -2/(h*h) - 1/pow(j*h,2) - ( lambda_1*(3*pow(f1[j],2) + pow(h1[j],2)) - mu_1 - 0.5*lambda_4*pow(h1[j],2) );

				Fh1[j][0] = (h1[j+1] - 2*h1[j] + h1[j-1])/(h*h) + (h1[j+1] - h1[j-1])/(j*h*h) + (f1[j]-h1[j])/pow(j*h,2) - ( lambda_1*(pow(f1[j],2) + pow(h1[j],2)) - mu_1 - 0.5*lambda_4*pow(f1[j],2) )*h1[j];
				Fh1[j][1] = -2/(h*h) - 1/pow(j*h,2) - ( lambda_1*(pow(f1[j],2) + 3*pow(h1[j],2)) - mu_1 - 0.5*lambda_4*pow(f1[j],2) );

				// Update with SOR

				f1[j] += - w*Ff1[j][0]/Ff1[j][1];
				h1[j] += - w*Fh1[j][0]/Fh1[j][1];

			} else if(symType=="U(1) alt"){

				// Ff1[j][0] = (f1[j+1] - 2*f1[j] + f1[j-1])/(h*h) + (f1[j+1] - f1[j-1])/(2*j*h*h) - pow(n*(1-h1[j])/(j*h),2)*f1[j] - lambda*f1[j]*( zeta*pow(f1[j],2) - eta + alpha*pow(f2[j],2) );
				// Ff1[j][1] = -2/(h*h) - pow(n*(1-h1[j])/(j*h),2) - lambda*( 3*zeta*pow(f1[j],2) - eta + alpha*pow(f2[j],2) );

				// Ff2[j][0] = (f2[j+1] - 2*f2[j] + f2[j-1])/(h*h) + (f2[j+1] - f2[j-1])/(2*j*h*h) - pow(n*(1-h1[j])/(j*h),2)*f2[j] - lambda*f2[j]*( pow(f2[j],2) - 1 + alpha*pow(f1[j],2) );
				// Ff2[j][1] = -2/(h*h) - pow(n*(1-h1[j])/(j*h),2) - lambda*( 3*pow(f2[j],2) - 1 + alpha*pow(f1[j],2) );

				// Fh1[j][0] = (h1[j+1] - 2*h1[j] + h1[j-1])/(h*h) - (h1[j+1] - h1[j-1])/(2*j*h*h) + 0.25*(pow(f1[j],2) + pow(f2[j],2))*(1-h1[j]);
				// Fh1[j][1] = -2/(h*h) - 0.25*(pow(f1[j],2) + pow(f2[j],2));

				// Update with SOR

				f1[j] = f1[j] - w*Ff1[j][0]/Ff1[j][1];
				f2[j] = f2[j] - w*Ff2[j][0]/Ff2[j][1];
				h1[j] = h1[j] - w*Fh1[j][0]/Fh1[j][1];

			}

		}

		// Do this outside of parallel region

		E = 0;
		double mu_b = 0;
		double mu_t = 0;
		Sigma_2 = 0;

		for(j=1;j<nx-1;j++){

			if(symType=="U(1)" or symType=="glob U(1) w cond"){

				if(abs(Ff1[j][0])>tolTest){ tolTest = abs(Ff1[j][0]); }
				if(abs(Ff2[j][0])>tolTest){ tolTest = abs(Ff2[j][0]); }
				if(abs(Fh1[j][0])>tolTest){ tolTest = abs(Fh1[j][0]); }
				if(abs(Fh2[j][0])>tolTest){ tolTest = abs(Fh2[j][0]); }

			} else if(symType=="U(1) alt"){

				if(abs(Ff1[j][0])>tolTest){ tolTest = abs(Ff1[j][0]); }
				if(abs(Ff2[j][0])>tolTest){ tolTest = abs(Ff2[j][0]); }
				if(abs(Fh1[j][0])>tolTest){ tolTest = abs(Fh1[j][0]); }

			} else{

				if(abs(Ff1[j][0])>tolTest){ tolTest = abs(Ff1[j][0]); }
				if(abs(Fh1[j][0])>tolTest){ tolTest = abs(Fh1[j][0]); }

			}

			if(calcEnergy){

				if(symType=="U(1)"){

					// E += j*h*h*( 0.5*( pow((f1[j+1]-f1[j-1])/(2*h),2) + pow(n*f1[j]*(h1[j]-h2[j])/(2*j*h),2) + pow((f2[j+1]-f2[j-1])/(2*h),2) + pow(n*f2[j]*(h1[j]+h2[j]-2)/(2*j*h),2) )
					//            + 0.5*pow(n/(j*h),2)*( g*pow((h1[j+1]-h1[j-1])/(2*h),2) + pow((h2[j+1]-h2[j-1])/(2*h),2) ) 
					//            + 0.25*lambda*( zeta*pow(f1[j],4) + pow(f2[j],4) - 2*eta*pow(f1[j],2) - 2*pow(f2[j],2) + 2*alpha*pow(f1[j]*f2[j],2) )
					//            - 0.25*lambda*( zeta*pow(v1,2) + pow(v2,2) - 2*eta*v1 - 2*v2 + 2*alpha*v1*v2 ) ); // This line subtracts the vacuum energy so that the energy density is zero in vacuum.

				} else if(symType=="SO(3)" or symType=="glob SO(3)"){

					// E += pow(j*h,2)*h*( 0.5*pow((f1[j+1]-f1[j-1])/(2*h),2) + pow(f1[j]*(1-h1[j])/(j*h),2) + 0.25*lambda*pow( pow(f1[j],2)-1 ,2) 
					// 				  + 4*( pow((h1[j+1]-h1[j-1])/(2*h),2) + 2*pow(h1[j]*(1-h1[j])/(j*h),2) )/pow(j*h,2) );

				} else if(symType=="NO"){

					//E += j*h*h*( 0.5*( pow((f1[j+1]-f1[j-1])/(2*h),2) + pow(n*(h1[j+1]-h1[j-1])/(2*j*h*h),2) + pow(n*f1[j]*(1-h1[j])/(j*h),2)) ) + 0.25*lambda*pow( pow(f1[j],2)-1 ,2);

				} else if(symType=="tHP"){

					// E += pow(j*h,2)*h*( 0.5*pow((f1[j+1]-f1[j-1])/(2*h),2) + pow(f1[j]*(1-h1[j])/(j*h),2) + 0.25*lambda*pow( pow(f1[j],2)-1 ,2)
					// 				  + ( pow((h1[j+1]-h1[j-1])/(2*h),2) + 0.5*pow(h1[j]*(2-h1[j])/(j*h),2) )/pow(j*h,2) );

				} else if(symType=="glob U(1) w cond"){

					// Actually these are magnitude squared

					double mod1 = pow(f1[j],2) + pow(h1[j],2);
					double mod2 = pow(f2[j],2) + pow(h2[j],2);
					double mix12 = f1[j]*f2[j] + h1[j]*h2[j];

					E += 2*pi*j*h*h*( 0.5*( pow((f1[j+1]-f1[j-1])/(2*h),2) + pow((f2[j+1]-f2[j-1])/(2*h),2) + pow((h1[j+1]-h1[j-1])/(2*h),2) + pow((h2[j+1]-h2[j-1])/(2*h),2)
									 + pow(n*f2[j]/(j*h),2) + pow(n*h1[j]/(j*h),2) )
					           + 0.25*( lambda_1*pow(mod1,2) + lambda_2*pow(mod2,2) - 2*mu_1*mod1 - 2*mu_2*mod2 + lambda_3*mod1*mod2 + lambda_4*pow(mix12,2) )
					           - 0.25*( lambda_1*pow(v1,2) + lambda_2*pow(v2,2) - 2*mu_1*v1 - 2*mu_2*v2 + (lambda_3+lambda_4)*v1*v2 ) ); // This line subtracts the vacuum energy.

					mu_b += 2*pi*j*h*h*( 0.5*( pow((f1[j+1]-f1[j-1])/(2*h),2) + pow((f2[j+1]-f2[j-1])/(2*h),2) + pow(n*f2[j]/(j*h),2) )
					            	   + 0.25*( lambda_1*pow(f1[j],4) + lambda_2*pow(f2[j],4) - 2*mu_1*pow(f1[j],2) - 2*mu_2*pow(f2[j],2) + (lambda_3+lambda_4)*pow(f1[j]*f2[j],2) )
					            	   - 0.25*( lambda_1*pow(v1,2) + lambda_2*pow(v2,2) - 2*mu_1*v1 - 2*mu_2*v2 + (lambda_3+lambda_4)*v1*v2 ) );

					mu_t += 0.5*pi*j*h*h*( lambda_1*pow(h1[j],4) + lambda_2*pow(h2[j],4) + (lambda_3+lambda_4)*pow(h1[j]*h2[j],2) );



					Sigma_2 += 2*pi*j*h*h*(pow(h1[j],2)+pow(h2[j],2));

				} else if(symType=="glob SO(3) w cond"){

					E+= pow(j*h,2)*h*( 0.5*( pow((f1[j+1]-f1[j-1])/(2*h),2) + pow((h1[j+1]-h1[j-1])/(2*h),2) + pow((f1[j]-h1[j])/(j*h),2) )
									 - 0.5*mu_1*(pow(f1[j],2)+pow(h1[j],2)) + 0.25*lambda_1*pow( pow(f1[j],2)+pow(h1[j],2) ,2) - 0.25*lambda_4*pow(f1[j],2)*pow(h1[j],2) + 0.25 );

				} else if(symType=="U(1) alt"){

					// E += j*h*h*( 0.5*( pow((f1[j+1]-f1[j-1])/(2*h),2) + pow((f2[j+1]-f2[j-1])/(2*h),2) + pow(n*(1-h1[j])/(j*h),2)*(pow(f1[j],2)+pow(f2[j],2)) ) + 2*pow(n/(j*h),2)*pow((h1[j+1]-h1[j-1])/(2*h),2)
					//            + 0.25*lambda*( zeta*pow(f1[j],4) + pow(f2[j],4) - 2*eta*pow(f1[j],2) - 2*pow(f2[j],2) + 2*alpha*pow(f1[j]*f2[j],2) )
					//            - 0.25*lambda*( zeta*pow(v1,2) + pow(v2,2) - 2*eta*v1 - 2*v2 + 2*alpha*v1*v2 ) );

				}

			}

		}

		vals << E << " ";

		if(symType=="glob U(1) w cond"){ vals << mu_b << " " << mu_t << " " << Sigma_2 << " " ; }

		vals << endl;

	}

	cout << "\rNumber of iterations for Static EoMs: " << iterNum << "                                      " << endl;

	for(i=0;i<nx;i++){

		if(symType=="extra scalar"){

			SOR_Fields << f1[i] << " " << f2[i] << " " << h1[i] << " " << h2[i] << " " << phi[0][i] << " " << phi[1][i] << endl;

		}else if(symType=="U(1)" or symType=="glob U(1) w cond"){

			SOR_Fields << f1[i] << " " << f2[i] << " " << h1[i] << " " << h2[i] << endl;

		} else if(symType=="U(1) alt"){

			SOR_Fields << f1[i] << " " << f2[i] << " " << h1[i] << endl;

		} else{

			SOR_Fields << f1[i] << " " << h1[i] << endl;

		}

	}


	gettimeofday(&end, NULL);

    cout << "Time taken: " << end.tv_sec - start.tv_sec << "s" << endl;

	return 0;

}