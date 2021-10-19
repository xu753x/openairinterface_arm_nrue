 
/*! \file openairinterface5g/openair1/PHY/BF/cu_function.cpp
 * \brief merge ISIP beamforming and MVDR algorithm
 * \author NCTU OpinConnect Terng-Yin Hsu, Sendren Xu, WEI-YING LIN, Min-Hsun Wu
 * \email  a22490010@gmail.com
 * \date   19-10-2021
 * \version 1.2
 * \note
 * \warning
 */

// C++
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <random>
#include <ccomplex>
// C
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
// CUDA
// #include <cuda_runtime.h>
// #include <cuda.h>
// #include <cublas_v2.h>
// #include <cusolverDn.h>
// #include <cuComplex.h>

#include "cu_function.h"

// #include "matplotlib-cpp/matplotlibcpp.h"

#define PI acos(-1)
#define BLOCK_SIZE 16
#define PRINT_RESULT 1
//#define PLOT_RESULT

using namespace std::literals::complex_literals;
//using namespace std::complex_literals;
// namespace plt = matplotlibcpp;


// print complex matrix matlab
void print_complex_matrix_matlab(std::complex<double> *matA, int rowA, int colA) {
    std::cout << "[";
    for(int i = 0; i < rowA; ++i) {
        for(int j = 0; j < colA; ++j) {
            std::cout << std::setprecision(16) << matA[i * colA + j].real() << "+" << matA[i * colA + j].imag() << "i ";
        }
        std::cout << ";" << std::endl;
    }
    std::cout << "]" << std::endl;
}


// print complex matrix
void print_complex_matrix(std::complex<double> *matA, int rowA, int colA) {
    for(int i = 0; i < rowA; ++i) {
        for(int j = 0; j < colA; ++j) {
            std::cout << std::fixed << std::setprecision(6) << std::setw(27) << matA[i * colA + j] << " ";
            
        }
        std::cout << std::endl;
    }
}


// generate random number with normal_distribution
std::complex<double> randn() {
    std::random_device randomness_device{};
    std::mt19937 pseudorandom_generator{randomness_device()};
    auto mean = 0.0;
    auto std_dev = 1.0;
    std::normal_distribution<> distribution{mean, std_dev};
    auto sample = distribution(pseudorandom_generator);
    return (std::complex<double>)(sample);
}

// add white gaussian noise
void awgn(std::complex<double> *input_signal, std::complex<double> *output_signal, int snr, int row, int col) {
    std::complex<double> Esym;
    std::complex<double> No;
    std::complex<double> noiseSigma;
    std::complex<double> n;
    for(int i = 0; i < row * col; i++) {
        Esym += pow(abs(input_signal[i]), 2) / std::complex<double>(row * col);
        No = Esym / std::complex<double>(snr);
        noiseSigma = sqrt(No / std::complex<double>(2));
        n = noiseSigma * (randn() + randn() * 1i);
        output_signal[i] = input_signal[i] + n;
    }
}


// complex matrix addition
void complex_matrix_addition(std::complex<double> *matA, std::complex<double> *matB, int rowA, int colA) {
    for(int i = 0; i < rowA; ++i) {
        for(int j = 0; j < colA; ++j) {
            matA[i * colA + j].real(matA[i * colA + j].real() + matB[i * colA + j].real());
            matA[i * colA + j].imag(matA[i * colA + j].imag() + matB[i * colA + j].imag());
        }
    }
}


// complex matrix subtraction
void complex_matrix_subtraction(std::complex<double> *matA, std::complex<double> *matB, int rowA, int colA) {
    for(int i = 0; i < rowA; ++i) {
        for(int j = 0; j < colA; ++j) {
            matA[i * colA + j].real(matA[i * colA + j].real() - matB[i * colA + j].real());
            matA[i * colA + j].imag(matA[i * colA + j].imag() - matB[i * colA + j].imag());
        }
    }
}


// complex matrix multiplication
void complex_matrix_multiplication(std::complex<double> *matA, std::complex<double> *matB, std::complex<double> *matC, int rowA, int rowB, int colB) {
    memset(matC, 0, rowA * colB * sizeof(std::complex<double>));
    for(int i = 0; i < rowA; ++i) {
        for(int j = 0; j < colB; ++j) {
            for(int k = 0; k < rowB; ++k) {
                matC[i * colB + j] += matA[i * rowB + k] * matB[k * colB + j];
            }
        }
    }
}


// get complex matrix by column
void complex_matrix_get_columns(std::complex<double> *matA, std::complex<double> *matCol, int rowA, int colA, int colTarget) {
    for(int i = 0; i < rowA; ++i) {
        matCol[i] = matA[i * colA + colTarget];
    }
}


// get complex matrix by row
void complex_matrix_get_rows(std::complex<double> *matA, std::complex<double> *matRow, int rowA, int colA, int rowTarget) {
    for(int i = 0; i < colA; ++i) {
        matRow[i] = matA[rowTarget * colA + i];
    }
}


// complex matrix conjugate transpose
void complex_matrix_conjugate_transpose(std::complex<double> *matA, int rowA, int colA) {
    std::complex<double> *temp = (std::complex<double>*)malloc(colA * rowA * sizeof(std::complex<double>));
    memcpy(temp, matA, (rowA * colA * sizeof(std::complex<double>)));
    for(int i = 0; i < rowA; ++i) {
        for(int j = 0; j < colA; ++j) {
            matA[j * rowA + i].real(temp[i * colA + j].real());
            matA[j * rowA + i].imag(-temp[i * colA + j].imag());
        }
    }
    free(temp);
}


// complex matrix conjugate transpose and multiplication
void complex_matrix_conjugate_transpose_multiplication(std::complex<double> *matA, std::complex<double> *matB, int rowA, int colA) {
    std::complex<double> *temp = (std::complex<double>*)malloc(colA * rowA * sizeof(std::complex<double>));
    memcpy(temp, matA, (rowA * colA * sizeof(std::complex<double>)));
    complex_matrix_conjugate_transpose(temp, rowA, colA);
    complex_matrix_multiplication(matA, temp, matB, rowA, colA, rowA);
    free(temp);
}


// compute Pn: matlab co.de: (Pn=Pn+vet_noise(:,ii)*vet_noise(:,ii)';), where (ii=1:length(vet_noise(1,:)))
void compute_Pn(std::complex<double> *Pn, std::complex<double> *vet_noise, int M, int len_t_theta) {
    std::complex<double> *vet_noise_temp = (std::complex<double>*)malloc(M * sizeof(std::complex<double>));
    std::complex<double> *Pn_temp = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
    for(int i = 0; i < M - len_t_theta; ++i) {
        complex_matrix_get_columns(vet_noise, vet_noise_temp, M, M - len_t_theta, i);
        complex_matrix_conjugate_transpose_multiplication(vet_noise_temp, Pn_temp, M, 1);
        complex_matrix_addition(Pn, Pn_temp, M, M);
    }
    free(vet_noise_temp);
    free(Pn_temp);
}


// compute S_MUSIC: matlab code: (S_MUSIC(i)=1/(a_vector'*Pn*a_vector))
std::complex<double> compute_S_MUSIC(std::complex<double> *a_vector, std::complex<double> *Pn, int M) {
    std::complex<double> *Pn_a_vector_temp = (std::complex<double>*)malloc(M * sizeof(std::complex<double>));
    std::complex<double> *S_MUSIC_temp = (std::complex<double>*)malloc(sizeof(std::complex<double>));
    complex_matrix_multiplication(Pn, a_vector, Pn_a_vector_temp, M, M, 1);
    complex_matrix_conjugate_transpose(a_vector, M, 1);
    complex_matrix_multiplication(a_vector, Pn_a_vector_temp, S_MUSIC_temp, 1, M, 1);
    std::complex<double> S_MUSIC = std::complex<double>(1) / S_MUSIC_temp[0];
    free(Pn_a_vector_temp);
    free(S_MUSIC_temp);
    return S_MUSIC;
}


// QR decomposer for c code
void qr(std::complex<double> *A, std::complex<double> *Q, std::complex<double> *R, int row, int col) {
    std::complex<double> *Q_col = (std::complex<double>*)malloc(row * 1 * sizeof(std::complex<double>));
    std::complex<double> *vector_cur = (std::complex<double>*)malloc(row * 1 * sizeof(std::complex<double>));
    std::complex<double> *Qvector_cur = (std::complex<double>*)malloc(row * 1 * sizeof(std::complex<double>));
    std::complex<double> *power_cur = (std::complex<double>*)malloc(sizeof(std::complex<double>));
    std::complex<double> *power_val = (std::complex<double>*)malloc(sizeof(std::complex<double>));
    std::complex<double> *proj_val = (std::complex<double>*)malloc(sizeof(std::complex<double>));
    std::complex<double> *proj_Qvector_cur = (std::complex<double>*)malloc(row * 1 * sizeof(std::complex<double>));
    for(int i = 0; i < row * col; i += (col + 1)) {
        Q[i].real(1);
        R[i].real(1);
    }
    for(int i = 0; i < col; ++i) {
        for(int m = 0; m < row; ++m) {
            Q[m * col + i] = A[m * col + i];
        }
        complex_matrix_get_columns(Q, Q_col, row, col, i);
        // conjuate Q_col
        complex_matrix_conjugate_transpose(Q_col, row, 1);
        memset(power_cur, 0, sizeof(std::complex<double>));
        complex_matrix_conjugate_transpose_multiplication(Q_col, power_cur, 1, row);
        power_cur[0] = sqrt(power_cur[0]);
        if(i > 0) {
            complex_matrix_get_columns(A, vector_cur, row, col, i);
            std::complex<double> *Q_col_proj = (std::complex<double>*)malloc(row * i * sizeof(std::complex<double>));
            std::complex<double> *proj_vector = (std::complex<double>*)malloc(i * sizeof(std::complex<double>));
            memset(proj_vector, 0, i * sizeof(std::complex<double>));
            for(int j = 0; j < i; ++j) {
                for(int m = 0; m < row; ++m) {
                    Q_col_proj[m * i + j] = Q[m * col + j];
                }
            }
            complex_matrix_conjugate_transpose(Q_col_proj, row, i);
            complex_matrix_multiplication(Q_col_proj, vector_cur, proj_vector, i, row, 1);
            complex_matrix_conjugate_transpose(Q_col_proj, i, row);
            memset(Q_col, 0, row * 1 * sizeof(std::complex<double>));
            complex_matrix_multiplication(Q_col_proj, proj_vector, Q_col, row, i, 1);
            complex_matrix_subtraction(vector_cur, Q_col, row, 1);
            for(int m = 0; m < row; ++m) {
                Q[m * col + i] = vector_cur[m];
            }
            for(int j = 0; j < i; ++j) {
                R[i + col * j] = proj_vector[j];
            }
            free(Q_col_proj);
            free(proj_vector);
        }
        complex_matrix_get_columns(Q, Q_col, row, col, i);
        // conjuate Q_col
        complex_matrix_conjugate_transpose(Q_col, row, 1);
        memset(power_val, 0, sizeof(std::complex<double>));
        complex_matrix_conjugate_transpose_multiplication(Q_col, power_val, 1, row);
        power_val[0] = sqrt(power_val[0]);

        //1e-4 = 0.0001
        if(power_val[0].real() / power_cur[0].real() < 1e-4) {
            R[i * row + i] = 0;
            // span again
            for(int m = 0; m < row; ++m) {
                Q[m * col + i] = 0;
            }
            Q[i * row + i].real(1);
            complex_matrix_get_columns(Q, vector_cur, row, col, i);
            for(int j = 0; j < i; ++j) {
                complex_matrix_get_columns(Q, Qvector_cur, row, col, j);
                memset(proj_val, 0, sizeof(std::complex<double>));
                complex_matrix_conjugate_transpose(Qvector_cur, row, 1);
                complex_matrix_multiplication(Qvector_cur, vector_cur, proj_val, 1, row, 1);
                complex_matrix_conjugate_transpose(Qvector_cur, 1, row);
                complex_matrix_get_columns(Q, Q_col, row, col, i);
                memset(proj_Qvector_cur, 0, row * 1 * sizeof(std::complex<double>));
                complex_matrix_multiplication(Qvector_cur, proj_val, proj_Qvector_cur, row, 1, 1);
                complex_matrix_subtraction(Q_col, proj_Qvector_cur, row, 1);
                for(int m = 0; m < row; ++m) {
                    Q[m * col + i] = Q_col[m];
                }
            }
            complex_matrix_get_columns(Q, Q_col, row, col, i);
            complex_matrix_conjugate_transpose(Q_col, row, 1);
            memset(power_val, 0, sizeof(std::complex<double>));
            complex_matrix_conjugate_transpose_multiplication(Q_col, power_val, 1, row);
            power_val[0] = sqrt(power_val[0]);
            complex_matrix_conjugate_transpose(Q_col, 1, row);
            for(int m = 0; m < row; ++m) {
                Q[m * col + i] /= power_val[0]; // Q[m * col + i] = Q[m * col + i] / power_val[0]
            }
        } else {
            R[i * row + i] = power_val[0];
            for(int m = 0; m < row; ++m) {
                Q[m * col + i] /= power_val[0];
            }
        }
    }
    free(Q_col);
    free(vector_cur);
    free(Qvector_cur);
    free(power_cur); 
    free(power_val);
    free(proj_val);
    free(proj_Qvector_cur);
}



// compute eigen upper triangular
void eigen_upper_triangular(std::complex<double> *A, std::complex<double> *eigenvalue, std::complex<double> *eigenvector, int row, int col) {
    std::complex<double> *vector_cur = (std::complex<double>*)malloc(row * 1 * sizeof(std::complex<double>));
    std::complex<double> *eigen_element_cur = (std::complex<double>*)malloc(sizeof(std::complex<double>));
    std::complex<double> *vector_cur_temp = (std::complex<double>*)malloc(sizeof(std::complex<double>));
    std::complex<double> *A_col = (std::complex<double>*)malloc(1 * col * sizeof(std::complex<double>));
    std::complex<double> diff_eigen_value = 0;
    for(int i = 0; i < row; ++i) {
        for(int j = 0; j < col; ++j) {
			if(i > j){
				A[i * col + j].real(0);
				A[i * col + j].imag(0);
			} 
            if(i == j) {
                eigenvalue[i * col + j] = A[i * col + j];
                eigenvector[i * col + j].real(1);
            }
        }
    }
    for(int i = 0; i < col; ++i) {
        complex_matrix_get_columns(eigenvector, vector_cur, row, col, i);
        for(int j = i - 1; j > -1; --j) {
            diff_eigen_value = eigenvalue[i * col + i] - eigenvalue[j * col + j];
            if(diff_eigen_value.real() < 1e-8) eigen_element_cur[0] = 0;
            else {
                complex_matrix_get_rows(A, A_col, row, col, j);
                complex_matrix_multiplication(A_col, vector_cur, eigen_element_cur, 1, row, 1);
                eigen_element_cur[0] = eigen_element_cur[0] / diff_eigen_value;
            }
            vector_cur[j] = eigen_element_cur[0];
        }
        complex_matrix_conjugate_transpose(vector_cur, row, 1);
        complex_matrix_conjugate_transpose_multiplication(vector_cur, vector_cur_temp, 1, row);
        vector_cur_temp[0] = sqrt(vector_cur_temp[0]);
        complex_matrix_conjugate_transpose(vector_cur, 1, row);
        for(int m = 0; m < row; ++m) {
            eigenvector[m * col + i] = vector_cur[m] / vector_cur_temp[0];
        }
    }
    free(vector_cur);
    free(eigen_element_cur);
    free(vector_cur_temp);
    free(A_col);
}


// compute complex eigenvector and eigenvalue for c code
void eigen(std::complex<double> *A, std::complex<double> *Ve, std::complex<double> *De, int row, int col, int iter) {
    std::complex<double> *Q = (std::complex<double>*)calloc(row * col, sizeof(std::complex<double>));
    std::complex<double> *R = (std::complex<double>*)calloc(row * col, sizeof(std::complex<double>));
    std::complex<double> *Q_temp = (std::complex<double>*)calloc(row * col, sizeof(std::complex<double>));
    std::complex<double> *Q_temp_clone = (std::complex<double>*)calloc(row * col, sizeof(std::complex<double>));
    for(int i = 0; i < row * col; i += (col + 1)) {
        Q_temp[i].real(1);
    }
    for(int i = 0; i < iter; ++i) {
        qr(A, Q, R, row, col);
        complex_matrix_multiplication(R, Q, A, row, row, col);
        complex_matrix_multiplication(Q_temp, Q, Q_temp_clone, row, row, col);
        memcpy(Q_temp, Q_temp_clone, row * col * sizeof(std::complex<double>));
    }
    for(int i = 0; i < row; ++i) {
        for(int j = 0; j < col; ++j) {
            if(i > j) A[i * col + j] = 0;
        }
    }
    std::complex<double> *YY0 = (std::complex<double>*)calloc(row * col, sizeof(std::complex<double>));
    std::complex<double> *XX0 = (std::complex<double>*)calloc(row * col, sizeof(std::complex<double>));
    eigen_upper_triangular(A, YY0, XX0, row, col);
    memcpy(De, YY0, row * col * sizeof(std::complex<double>));
    complex_matrix_multiplication(Q_temp, XX0, Ve, row, row, col);
    free(Q);
    free(R);
    free(Q_temp);
    free(Q_temp_clone);
    free(YY0);
    free(XX0);
}


// compute the MVDR DOA in one dimension on CPU
void MVDR_DOA_1D_CPU(int M, int snr, int qr_iter, int multi_input, float *result) {
    #ifdef PRINT_RESULT
    printf("--Parameter--\n");
    printf("Antenna count:\t\t%d\n", M);
    printf("SNR:\t\t\t%d\n", snr);
    printf("QR iteration:\t\t%d\n", qr_iter);
    printf("Multiple input size:\t%d\n", multi_input);
    #endif
// generate the signal
    // time initial
    float timeStart, timeEnd;
    // parameter setting
    const int fc = 180e+6;
    const int c = 3e+8;
    const double lemda = (double)c / (double)fc;
    std::complex<double> d(lemda * 0.5);
    std::complex<double> kc(2.0 * PI / lemda);
    const int nd = 500;
    // angle setting
    const int len_t_theta = 1;
    std::complex<double> *t_theta = (std::complex<double>*)malloc(len_t_theta * sizeof(std::complex<double>));
    t_theta[0].real(3);
    t_theta[1].real(12);
    t_theta[2].real(20);
    #ifdef PRINT_RESULT
    std::cout << "Theta(degree):\t\t[";
    for(int i = 0; i < len_t_theta; ++i) {
        if(i != len_t_theta - 1) std::cout << t_theta[i].real() << ", ";
        else std::cout << t_theta[i].real() << "]\n\n";
    }
    std::cout << "---Time---" << std::endl;
    #endif
    // A_theta matrix (M, length of t_theta)
    std::complex<double> *A_theta = (std::complex<double>*)malloc(M * len_t_theta * sizeof(std::complex<double>));
    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < len_t_theta; ++j) {
            A_theta[i * len_t_theta + j] = exp(1i * kc * std::complex<double>(i) * d * sin(t_theta[j] * std::complex<double>(PI / 180)));
        }
    }
    // t_sig matrix (length of t_theta, nd)
    std::complex<double> *t_sig = (std::complex<double>*)malloc(len_t_theta * nd * sizeof(std::complex<double>));
    for(int i = 0; i < len_t_theta; ++i) {
        for(int j = 0; j < nd; ++j) {
            t_sig[i * nd + j] = (randn() + randn() * 1i) / std::complex<double>(sqrt(2));
            // if(i == 0) t_sig[i * nd + j] *= (std::complex<double>)2;
        }
    }
    // sig_co matrix (M, nd)
    std::complex<double> *sig_co = (std::complex<double>*)malloc(M * nd * sizeof(std::complex<double>));
    // compute sig_co
    complex_matrix_multiplication(A_theta, t_sig, sig_co, M, len_t_theta, nd);

// receiver
    // x_r matrix (M, nd)
    std::complex<double> *x_r = (std::complex<double>*)malloc(M * nd * sizeof(std::complex<double>));
    // memcpy(x_r, sig_co, M * nd * sizeof(std::complex<double>));
    // add noise to the signal
    awgn(sig_co, x_r, snr, M, nd);

// mvdr algorithm
    // R_xx matrix (M, M)
    std::complex<double> *R_xx = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
    // matlab code:  (R_xx = 1 / M * x_r * x_r')
    complex_matrix_conjugate_transpose_multiplication(x_r, R_xx, M, nd);
    for(int i = 0; i < M * M; ++i) R_xx[i] /= std::complex<double>(M);
    // compute eigenvector Ve (M, M)
    std::complex<double> *Ve = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
    std::complex<double> *De = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
    // timestamp start
    timeStart = clock();
    eigen(R_xx, Ve, De, M, M, qr_iter);
	std::complex<double> *R_xx_inv_1 = (std::complex<double>*)malloc(M * M * sizeof(std::complex<double>));
	std::complex<double> *Pn = (std::complex<double>*)calloc(M * M, sizeof(std::complex<double>));
	for(int i = 0; i < M * M; i += (M + 1)) {
		if(abs(De[i])<0.00000000001) {
			De[i].real(1000000);
			De[i].imag(0);
		}
		else De[i]= std::complex <double> (1)/De[i];
	}
	
	complex_matrix_multiplication(Ve, De, R_xx_inv_1, M, M, M);
	complex_matrix_conjugate_transpose(Ve, M, M);
	complex_matrix_multiplication(R_xx_inv_1, Ve, Pn, M, M, M);

    timeEnd = clock();
    #ifdef PRINT_RESULT
    std::cout << std::setprecision(6) << "MVDR (cpu):\t\t" << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
    #endif
    result[2] += (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000;

// array pattern
    // parameter setting
    const int len_dth = 401;
    double *dth = (double*)malloc(len_dth * sizeof(double));
    double *dr = (double*)malloc(len_dth * sizeof(double));
    for(int i = 0; i < len_dth; ++i) { // do only one time, no need to be paralleled
        dth[i] = -10 + 0.1 * i;
        dr[i] = dth[i] * PI / 180;
    }
    // compute S_MVDR_dB
    std::complex<double> *a_vector = (std::complex<double>*)malloc(M * sizeof(std::complex<double>));
    std::complex<double> *S_MVDR = (std::complex<double>*)malloc(len_dth * sizeof(std::complex<double>));
    double *S_MVDR_dB = (double*)malloc(len_dth * sizeof(double));
    // timestamp start
    timeStart = clock();
    for(int i = 0; i < len_dth; ++i) { // can be paralleled to compute S_MVDR_dB
        for(int j = 0; j < M; ++j) {
            a_vector[j] = exp(1i * kc * (std::complex<double>)j * d * sin(dr[i]));
        }
        S_MVDR[i] = compute_S_MUSIC(a_vector, Pn, M);
        // compute S_MVDR_dB
        S_MVDR_dB[i] = 20 * log10(abs(S_MVDR[i]));
    }
    // find Max and position
    double max_temp = S_MVDR_dB[0];
    int position = 0;
    for(int i = 0; i < len_dth; ++i) {
        if(S_MVDR_dB[i] > max_temp) {
            max_temp = S_MVDR_dB[i];
            position = i;
        }
    }
    // timestamp end
    timeEnd = clock();
    #ifdef PRINT_RESULT
    std::cout << std::setprecision(6) << "Array pattern (cpu):\t" << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
    printf("\n--Result--\n");
    std::cout << "Theta estimation:\t" << dth[position] << std::endl;
    std::cout << std::endl << "-----------------------------------------" << std::endl << std::endl;
    #endif
    float error;
    error = abs(dth[position] - t_theta[0].real());

    if(error > result[0]) result[0] = error;
    if (error != 0) result[1] += pow(error, 2);

// plot the result
    #ifdef PLOT_RESULT
    std::vector<double> S_MVDR_dB_vec(S_MVDR_dB, S_MVDR_dB + len_dth);
    std::vector<double> dth_vec(dth, dth + len_dth);
    plt::plot(dth_vec, S_MVDR_dB_vec, "blue");
    plt::title("MUSIC DOA Estimation");
    plt::xlabel("Theta (degree)");
    plt::ylabel("Power Spectrum (dB)");
    plt::xlim(dth[0], dth[len_dth - 1]);
    plt::grid(true);
    plt::show();
    #endif

// free memory
    free(t_theta);
    free(A_theta);
    free(t_sig);
    free(sig_co);
    free(x_r);
    free(R_xx);
    free(Ve);
    free(De);
    // free(vet_noise);
    free(Pn);
    free(dth);
    free(dr);
    free(a_vector);
    free(S_MVDR);
    free(S_MVDR_dB);
}



extern "C" 
void qr_test(double *matA ,int rowA, int colA) {
    
    std::complex<double> *A=(std::complex<double>*)malloc(rowA* colA * sizeof(std::complex<double>));
    std::complex<double> *Q=(std::complex<double>*)malloc(rowA* colA * sizeof(std::complex<double>));
    std::complex<double> *R=(std::complex<double>*)malloc(rowA* colA * sizeof(std::complex<double>));
    for (int i = 0; i < rowA*colA; i++)
    {
        A[i].real(matA[i]);
    }
    
    qr(A,Q,R,2,2);
    printf("A:\n");
    print_complex_matrix(A ,rowA,colA);
    printf("Q:\n");
    print_complex_matrix(Q ,rowA,colA);
    printf("R:\n");
    print_complex_matrix(R ,rowA,colA);
    
    free(A);
    free(Q);
    free(R);

}

// test
// int main(void){
//     // double a[]={100,200,300,400};
//     // qr_test(a ,2,2);
//     int row=2;
//     int col=2;
//     int iter=10;


//     std::complex<double> A[]={100,200,300,400};
//     std::complex<double> *Ve=(std::complex<double>*)malloc(row* col * sizeof(std::complex<double>));
//     std::complex<double> *De=(std::complex<double>*)malloc(row* col * sizeof(std::complex<double>));

//     eigen(A,Ve,De,row,col,iter);
//     print_complex_matrix(Ve ,row,col);
//     print_complex_matrix(De ,row,col);

//     free(Ve);
//     free(De);
    
//     return 0;
// }


