#include <iostream>

#include "libmv/multiview/fundamental_kernel.h"
#include "libmv/multiview/projection.h"
#include "libmv/multiview/test_data_sets.h"
#include "libmv/numeric/numeric.h"
#include "testing/testing.h"
#include "libmv/multiview/nviewtriangulation.h" 


using namespace libmv;
  
void dotMultiplication(const Mat & x1, const Mat & x2, Mat * Out)
{
  assert(x1.rows() == x2.rows());
  assert(x1.cols() == x2.cols());

  (*Out).resize(x1.rows(),x1.cols());
  (*Out) = x1.array() * x2.array();
}

void buildTrifocalMatrix(const Mat & x1, const Mat & x2, const Mat & x3, Mat * Amat)
{
  Mat x1H, x2H, x3H;
  EuclideanToHomogeneous(x1,&x1H);
  EuclideanToHomogeneous(x2,&x2H);
  EuclideanToHomogeneous(x3,&x3H);

  Mat oneMat (Mat::Ones(1, 3));

  Mat & A = *Amat;
  const int npts = 7;
  A = Mat(4*npts,27);
  A.setZero();

  // Fill the A matrix (The linear problem Ax = B):
  int n=0;
  for (int I = 1; I<=2; ++I)  {
    for (int L = 1; L<=2; ++L)  {
      n++;
      int rmin = (n-1)*npts;
      //int rmax = n*npts;
      int c1 = 3*(I-1)+L; // = [1 2 4 5]
      int c2 = 3*(I-1)+3; // = [3 3 6 6]
      int c3 = L+6; // = [7 8 7 8]
      int c4 = 9; // = [9 9 9 9]

      // Constraint 1 :
      A.block(rmin, c1-1 ,npts, 1) = x1.row(0).transpose();
      A.block(rmin, c1-1+9 ,npts, 1) = x1.row(1).transpose();
      A.block(rmin, c1-1+18 ,npts, 1).fill(1.0);

      //-- Constraint 2 END
      {
        Mat Constraint2( 7,3 );
        dotMultiplication(-x1H.transpose(), ( x3H.row(L-1).transpose() * oneMat), &Constraint2 );

        A.block(rmin, c2-1 ,npts, 1) = Constraint2.col(0);
        A.block(rmin, c2-1+9 ,npts, 1) = Constraint2.col(1);
        A.block(rmin, c2-1+18 ,npts, 1) = Constraint2.col(2);
      }

      //-- Constraint 3
      {
        Mat Constraint3( 7,3 );
        dotMultiplication(-x1H.transpose(), ( x2H.row(I-1).transpose() * oneMat), &Constraint3 );

        A.block(rmin, c3-1 ,npts, 1) = Constraint3.col(0);
        A.block(rmin, c3-1+9 ,npts, 1) = Constraint3.col(1);
        A.block(rmin, c3-1+18 ,npts, 1) = Constraint3.col(2);
      }

      //-- Constraint4
      {
        Mat outA;
        dotMultiplication( x2H.row(I-1).transpose(), x3H.row(L-1).transpose(), &outA);

        Mat Constraint4;
        dotMultiplication( x1H.transpose(), outA * oneMat , &Constraint4);

        A.block(rmin, c4-1 ,npts, 1) = Constraint4.col(0);
        A.block(rmin, c4-1+9 ,npts, 1) = Constraint4.col(1);
        A.block(rmin, c4-1+18 ,npts, 1) = Constraint4.col(2);
      }
    }
  }
}

//%--------------------------------------------------------------------------
//% subfunction to retrieve one epipole at a time.
void epipoleFromTrifocalTensor( Mat3 t0, Mat3 t1, Mat3 t2, Vec3 * epipole)
{
  Mat3 L;
  Mat3 * tabT[3] = {&t0,&t1,&t2};
  for (int i=0; i<3; ++i) {
    Mat3 & T = *tabT[i];
    //Extract epipolar lines from the column of V corresponding to
    //smallest singular value.
    Vec q;
    Nullspace(&T,&q);
    L.row(i) = q;
  }
  Vec q;
  Nullspace(&L,&q);
  *epipole = q/q(2); //-- Homogenous to euclidian
}

void epipolesFromTrifocalTensor( Mat3 t0, Mat3 t1, Mat3 t2, Vec3 * e2, Vec3 * e3)
{
  epipoleFromTrifocalTensor(t0.transpose(), t1.transpose(), t2.transpose(), &(*e2));
  epipoleFromTrifocalTensor(t0, t1, t2, &(*e3));
}

//%--------------------------------------------------------------------------
//% Function to build the relationship matrix which represent
//% T_i^jk = a_i^j * b_4^k - a_4^j * b_i^k
//% as t = E * aa, where aa = [a'(:) ; b'(:)], (note: for representation only)

Mat E_from_ee(const Vec3 & e2,const Vec3 & e3)
{
  Mat e2Block(9,3);
  e2Block.block(0,0,3,3) = -e2(0) * Mat::Identity(3,3);
  e2Block.block(3,0,3,3) = -e2(1) * Mat::Identity(3,3);
  e2Block.block(6,0,3,3) = -e2(2) * Mat::Identity(3,3);

  Mat e3Block(9,3);
  e3Block.setZero();
  e3Block.block(0,0,3,1) = e3;
  e3Block.block(3,1,3,1) = e3;
  e3Block.block(6,2,3,1) = e3;

  Mat E = Mat::Zero(27,18);
  E.block(0,0,9,3) = e3Block;
  E.block(0,9,9,3) = e2Block;

  E.block(9,3,9,3) = e3Block;
  E.block(9,12,9,3) = e2Block;
  E.block(18,6,9,3)= e3Block;
  E.block(18,15,9,3)= e2Block;
  return E;
}

void minAlg_5p6(const Mat & A, const Mat & G, Mat * x)
{
  //Compute the SVD of G
  Eigen::JacobiSVD<Mat> svd(G, Eigen::ComputeThinU);

  int rank = 15;
  //Optional computing the rank via LU decomposition
  Eigen::FullPivLU<Mat> lu_decomp(G);
  rank = static_cast<int>(lu_decomp.rank());
  //End optional

  Mat U = svd.matrixU();

  //% Extract U2 from U
  Mat U2 = U.block(0,0,U.rows(),rank);

  // Find unit vector x2 that minimizes ||A*U2*x2||
  Vec V;
  Mat A2 = (A*U2);
  Nullspace(&A2, &V);

  //-- Compute the required solution
  (*x) = U2 * V;
}


TEST(Fundamental, TrifocalMatrixBuilder) {

  Mat x1(2,7);
  x1 << 141, 83, 84, 79, 158, 189, 125,
         98, 122, 9, 96, 154, 149, 67;

  Mat x2(2,7);
  x2 << 139, 81, 84, 78, 157, 188, 125,
         98, 122, 9, 96, 154, 149, 67;

  Mat x3(2,7);
  x3 << 138, 80, 83, 78, 156, 187, 124,
         97, 121, 8, 95, 154, 148, 67;

  Mat A;
  buildTrifocalMatrix(x1,x2,x3, &A);

  //Extract trifocal tensor from the column of V corresponding to
  //smallest singular value.
  Vec q;
  Nullspace(&A, &q);

  // Reshape it as a 3*3*3 matrix T=[t0,t1,t2]

  Mat3 t0,t1,t2;
  t0 = Map<RMat3>(q.block<9,1>(0,0).data(),3,3);
  t1 = Map<RMat3>(q.block<9,1>(9,0).data(),3,3);
  t2 = Map<RMat3>(q.block<9,1>(18,0).data(),3,3);
  
  Vec3 e2,e3;
  epipolesFromTrifocalTensor( t0, t1, t2, &e2, &e3);
  
  Mat E = E_from_ee(e2,e3);

  Mat t;
  minAlg_5p6(A,E, &t);

  t0 = Map<RMat3>(t.block<9,1>(0,0).data(),3,3);
  t1 = Map<RMat3>(t.block<9,1>(9,0).data(),3,3);
  t2 = Map<RMat3>(t.block<9,1>(18,0).data(),3,3);
  
  //Denormalisation

  //Computing camera matrices
  // HZ 15.1  page 375 2nd Edition : (P retrieval from trifocal tensor)
  Vec3 e2n = e2.normalized();
  Vec3 e3n = e3.normalized();
  Mat34 P0 = HStack(Mat3::Identity(), Mat::Zero(3,1));
  Mat34 P1;
  P1.col(0) = t0 * e3n;
  P1.col(1) = t1 * e3n;
  P1.col(2) = t2 * e3n;
  P1.col(3) = e2n;
  Mat34 P2;
  Mat3 factor = e3n*e3n.transpose()-Mat3::Identity();
  P2.col(0) = factor * t0 * e2n;
  P2.col(1) = factor * t1 * e2n;
  P2.col(2) = factor * t2 * e2n;
  P2.col(3) = e3n;
}

// Trifocal with normalized coordinates
TEST(Fundamental, TrifocalMatrixBuilder_normalizedCoordinates) {

  Mat x1(2,7);
  x1 << 141, 83, 84, 79, 158, 189, 125,
    98, 122, 9, 96, 154, 149, 67;

  Mat x2(2,7);
  x2 << 139, 81, 84, 78, 157, 188, 125,
    98, 122, 9, 96, 154, 149, 67;

  Mat x3(2,7);
  x3 << 138, 80, 83, 78, 156, 187, 124,
    97, 121, 8, 95, 154, 148, 67;

  Mat3 H1,H2,H3;
  Mat x1_n, x2_n, x3_n;
  NormalizePoints(x1, &x1_n, &H1);
  NormalizePoints(x2, &x2_n, &H2);
  NormalizePoints(x3, &x3_n, &H3);

  Mat A;
  buildTrifocalMatrix(x1_n, x2_n, x3_n, &A);

  //Extract trifocal tensor from the column of V corresponding to
  //smallest singular value.
  Vec q;
  Nullspace(&A, &q);

  // Reshape it as a 3*3*3 matrix T=[t0,t1,t2]
  Mat3 t0,t1,t2;
  t0 = Map<RMat3>(q.block<9,1>(0,0).data(),3,3);
  t1 = Map<RMat3>(q.block<9,1>(9,0).data(),3,3);
  t2 = Map<RMat3>(q.block<9,1>(18,0).data(),3,3);

  //Computes epipoles
  Vec3 e2,e3;
  epipolesFromTrifocalTensor( t0, t1, t2, &e2, &e3);

  //Minimize geometric error
  Mat E = E_from_ee(e2,e3);
  Mat t;
  minAlg_5p6(A,E, &t);

  t0 = Map<RMat3>(t.block<9,1>(0,0).data(),3,3);
  t1 = Map<RMat3>(t.block<9,1>(9,0).data(),3,3);
  t2 = Map<RMat3>(t.block<9,1>(18,0).data(),3,3);

  //Denormalisation
  t0 = H2.inverse() * t0 * (H3.inverse()).transpose();
  t1 = H2.inverse() * t1 * (H3.inverse()).transpose();
  t2 = H2.inverse() * t2 * (H3.inverse()).transpose();

  Mat3 t0d = H1(0,0)*t0 + H1(1,0)*t1 + H1(2,0)*t2;
  Mat3 t1d = H1(0,1)*t0 + H1(1,1)*t1 + H1(2,1)*t2;
  Mat3 t2d = H1(0,2)*t0 + H1(1,2)*t1 + H1(2,2)*t2;
  
  t0 = t0d;
  t1 = t1d;
  t2 = t2d;

  //Computing camera matrices
  // HZ 15.1  page 375 2nd Edition : (P retrieval from trifocal tensor)
  Vec3 e2n = e2.normalized();
  Vec3 e3n = e3.normalized();
  Mat34 P0 = HStack(Mat3::Identity(), Mat::Zero(3,1));
  Mat34 P1;
  P1.col(0) = t0 * e3n;
  P1.col(1) = t1 * e3n;
  P1.col(2) = t2 * e3n;
  P1.col(3) = e2n;
  Mat34 P2;
  Mat3 factor = e3n*e3n.transpose()-Mat3::Identity();
  P2.col(0) = factor * t0 * e2n;
  P2.col(1) = factor * t1 * e2n;
  P2.col(2) = factor * t2 * e2n;
  P2.col(3) = e3n;
}

// Trifocal with synthetic data
TEST(Fundamental, TrifocalMatrixBuilder_syntheticData) {

  //http://staff.science.uva.nl/~leo/hz/chap11_13.pdf
  //http://www.csse.uwa.edu.au/~du/Software/compvis/
  int nviews = 6;
  int npoints = 7;
  NViewDataSet d = NRealisticCamerasFull(nviews, npoints);
  d.ExportToPLY("basicTrifocal.ply");

  //Collect point of each view
  Mat x1 = d.x[0], x2 = d.x[1], x3 = d.x[2];

  Mat3 H1,H2,H3;
  Mat x1_n, x2_n, x3_n;
  NormalizePoints(x1, &x1_n, &H1);
  NormalizePoints(x2, &x2_n, &H2);
  NormalizePoints(x3, &x3_n, &H3);

  Mat A;
  buildTrifocalMatrix(x1_n, x2_n, x3_n, &A);

  //Extract trifocal tensor from the column of V corresponding to
  //smallest singular value.
  Vec q;
  Nullspace(&A, &q);

  // Reshape it as a 3*3*3 matrix T=[t0,t1,t2]
  Mat3 t0,t1,t2;
  t0 = Map<RMat3>(q.block<9,1>(0,0).data(),3,3);
  t1 = Map<RMat3>(q.block<9,1>(9,0).data(),3,3);
  t2 = Map<RMat3>(q.block<9,1>(18,0).data(),3,3);

  //Computes epipoles
  Vec3 e2,e3;
  epipolesFromTrifocalTensor( t0, t1, t2, &e2, &e3);

  //Minimize geometric error
  Mat E = E_from_ee(e2,e3);
  Mat t;
  minAlg_5p6(A,E, &t);

  t0 = Map<RMat3>(t.block<9,1>(0,0).data(),3,3);
  t1 = Map<RMat3>(t.block<9,1>(9,0).data(),3,3);
  t2 = Map<RMat3>(t.block<9,1>(18,0).data(),3,3);

  //Denormalisation
  t0 = H2.inverse() * t0 * (H3.inverse()).transpose();
  t1 = H2.inverse() * t1 * (H3.inverse()).transpose();
  t2 = H2.inverse() * t2 * (H3.inverse()).transpose();

  Mat3 t0d = H1(0,0)*t0 + H1(1,0)*t1 + H1(2,0)*t2;
  Mat3 t1d = H1(0,1)*t0 + H1(1,1)*t1 + H1(2,1)*t2;
  Mat3 t2d = H1(0,2)*t0 + H1(1,2)*t1 + H1(2,2)*t2;

  t0 = t0d;
  t1 = t1d;
  t2 = t2d;

  //Check algebraic errors:
  //[x']_x (E_i x_i T_i) [x'']_x = 0_3x3
  std::cout << "\n Trifocal tensor Algebraic error: \n";
  for (size_t i = 0; i < npoints; ++i)
  {
    Vec3 x( d.x[0].col(i)(0), d.x[0].col(i)(1), 1.0);
    Vec3 xPrime( d.x[1].col(i)(0), d.x[1].col(i)(1), 1.0);
    Vec3 xPrimePrime( d.x[2].col(i)(0), d.x[2].col(i)(1), 1.0);

    Mat3 res = CrossProductMatrix(xPrime) *
      ( x(0)* t0 + x(1)* t1 + x(2)* t2)
      * CrossProductMatrix(xPrimePrime);
    std::cout << res.array().sum() << '\n';
  }
  std::cout << std::endl;

  //(update Epipole with the unormalized tensor)
  epipolesFromTrifocalTensor( t0, t1, t2, &e2, &e3);
  //Computing camera matrices 
  // HZ 15.1  page 375 2nd Edition : (P retrieval from trifocal tensor)
  // Be carefull epipoles must be normalized
  Vec3 e2n = e2.normalized();
  Vec3 e3n = e3.normalized();
  Mat34 P0 = HStack(Mat3::Identity(), Mat::Zero(3,1));
  Mat34 P1;
  P1.col(0) = t0 * e3n;
  P1.col(1) = t1 * e3n;
  P1.col(2) = t2 * e3n;
  P1.col(3) = e2n;
  Mat34 P2;
  Mat3 factor = e3n*e3n.transpose() - Mat3::Identity();
  std::cout << "\n Factor : \n" << factor;
  std::cout << "\n Epipole2 : \n" << e2n;
  std::cout << "\n Epipole3 : \n" << e3n;
  P2.col(0) = factor * t0.transpose() * e2n;
  P2.col(1) = factor * t1.transpose() * e2n;
  P2.col(2) = factor * t2.transpose() * e2n;
  P2.col(3) = e3n;

  // I don't understand but it seems not work...
  //P matrices don't seems to be good.

  std::cout << "\n P0 : \n" << P0;
  std::cout << "\n P1 : \n" << P1;
  std::cout << "\n P2: \n" << P2;

    NViewDataSet d2 =d;
  d2.C.resize(3);
  d2.K.resize(3);
  d2.t.resize(3);
  //-- Modify d2 and export the results to compare
  KRt_From_P(P0, &d2.K[0], &d2.R[0], &d2.t[0]);
  KRt_From_P(P1, &d2.K[1], &d2.R[1], &d2.t[1]);
  KRt_From_P(P2, &d2.K[2], &d2.R[2], &d2.t[2]);

  std::cout << "\n K0 : \n" << d2.K[0];
  std::cout << "\n R0 : \n" << d2.R[0];
  std::cout << "\n t0: \n" << d2.t[0];


  std::cout << "\n K1 : \n" << d2.K[1];
  std::cout << "\n R1 : \n" << d2.R[1];
  std::cout << "\n t1: \n" << d2.t[1];


  std::cout << "\n K2 : \n" << d2.K[2];
  std::cout << "\n R2 : \n" << d2.R[2];
  std::cout << "\n t2: \n" << d2.t[2];

  //Retriangulate the 3D points
  /*{
    std::vector<Mat34> vec_Ps;
    vec_Ps.push_back(P0);
    vec_Ps.push_back(P1);
    vec_Ps.push_back(P2);
    Vec4 X;

    for (size_t i = 0; i < npoints; ++i)
    {
      Mat2X xproj(2, nviews);
      for (int j = 0; j < nviews; ++j) {
        xproj.col(j) = d._x[j].col(i);
      }
      NViewTriangulate(xproj, vec_Ps, &X);
      Vec3 Xs = X.head<3>()/X(3);
      std::cout << Xs.transpose() << '\n' << X << '\n';
      d2._X.col(i) = Xs;
    }
  }*/

  //d2._t.resize(0);

  d2.X.resize(0,0);
  d2.ExportToPLY("AfterTrifocal.ply");
}
