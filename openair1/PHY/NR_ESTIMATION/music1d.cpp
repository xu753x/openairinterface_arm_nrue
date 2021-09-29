/* music1d.cpp */

#include "music1d.h"
#include <iostream>
#include <Eigen/Dense>
#include <complex>
#include <cmath>

extern "C" {

  short music1d(short **arg) {
    using namespace Eigen;
    short snapcount = 10;
    MatrixXcf data(2,snapcount);
    float wavelength = 2.99792458e8/3.6192e9;
    RowVectorXf doas(180);
    MatrixXf::Index maxIndex;
    Vector2cf Un;

    for (short k = 0; k<180; k++){
        doas(k) = (k/90.0-1)*M_PI_2;
    }

    MatrixXcf steering_matrix = exp((2*M_PI*Vector2f (0,0.0428)*doas.array().sin().matrix()/wavelength).array()*std::complex<float> (0,1));

    for (short i = 0; i < 2; i++) {
      for (short j = 0;j<snapcount;j++){
        data(i,j) = std::complex<float> (*(arg[i]+2*j),*(arg[i]+2*j+1));
      }
    }
    MatrixXcf dataT = data.adjoint();
    MatrixXcf R = data*dataT/snapcount;
    MatrixXcf RT = R.adjoint();
    SelfAdjointEigenSolver<MatrixXcf> complexeigensolver((R+RT)*0.5);

    if (complexeigensolver.info() != Success) abort();
    VectorXcf U = complexeigensolver.eigenvectors().col(1);
    Un(0) = U(1); Un(1) = -U(0);

    ArrayXcf V = Un.adjoint()*steering_matrix;
    VectorXf Vr = 1.0/real(V*conj(V));
    Vr.maxCoeff(&maxIndex);
    int aoa = maxIndex;
    return aoa;
    // std::cout << "-----------------------------------------------------------\n" << std::endl;
    // std::cout << "(music) Current Angle is:" << aoa << "\n" << std::endl;
  }

} /* extern "C" */