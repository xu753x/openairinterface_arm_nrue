 
/*! \file openairinterface5g/openair1/PHY/BF/cu_function.cpp
 * \brief merge ISIP beamforming and QR decomposer
 * \author NCTU OpinConnect Terng-Yin Hsu, Sendren Xu, WEI-YING LIN, Min-Hsun Wu
 * \email  a22490010@gmail.com
 * \date   25-9-2021
 * \version 1.0
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


