#include "config.h"

#include<dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include<dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/io.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/preconditioners.hh>

extern "C" {
    // int test_istl(double* val);
    int istl_solve_block(int n, int nnz, int* row_size, int* BI, int* BJ, 
                     double* V1, double* V2, double* V3, double* V4, 
                     double* r1, double* r2, double* dx1, double* dx2,
                     double tol, int max_iter);
}

namespace Dune
{
  using Mat1 = BCRSMatrix<FieldMatrix<double,2,2>>;
  using Vec1 = BlockVector<FieldVector<double,2>>;

  // explicit template instantiation of all iterative solvers
  // field_type = double
  template class SeqILU<Mat1, Vec1, Vec1>;
  template class RestartedGMResSolver<Vec1>;
//   template class RestartedFlexibleGMResSolver<Vec1>;
  template class BiCGSTABSolver<Vec1>;
} // end namespace Dune

using namespace Dune;


int istl_solve_block(int n, int nnz, int* row_size, int* BI, int* BJ, 
                     double* V1, double* V2, double* V3, double* V4, 
                     double* r1, double* r2, double* dx1, double* dx2,
                     double tol, int max_iter){
    
    const int BS = 2;

    typedef FieldVector<double,BS> VectorBlock;
    typedef FieldMatrix<double,BS,BS> MatrixBlock;
    typedef BlockVector<VectorBlock> BVector;
    typedef BCRSMatrix<MatrixBlock> BCRSMat;
    typedef MatrixAdapter<BCRSMat,BVector,BVector> Operator;

    BCRSMat Jac(n,n,BCRSMat::random);
    Operator fop(Jac);

    for(int i=0; i<n; i++)
        Jac.setrowsize(i,row_size[i]);
    Jac.endrowsizes();

    for(int i=0; i<nnz; i++){
        Jac.addindex(BI[i], BJ[i]);
    }
    Jac.endindices();

    for(int i=0; i<nnz; i++){
        Jac[BI[i]][BJ[i]][0][0] = V1[i];
        Jac[BI[i]][BJ[i]][0][1] = V2[i];
        Jac[BI[i]][BJ[i]][1][0] = V3[i];
        Jac[BI[i]][BJ[i]][1][1] = V4[i];
    }
    
    InverseOperatorResult res;
    BVector rhs(n), x(n);
    x = 0;
    for (int i = 0; i < n; i++){
        rhs[i][0] = r1[i];
        rhs[i][1] = r2[i];
    }
        
    // printvector(std::cout, rhs, "rhs", "row");
    // printvector(std::cout, x, "initial guess", "row");
    
    SeqILU<BCRSMat,BVector,BVector> prec0(Jac, true);
    // RestartedGMResSolver<BVector> solver(fop, prec0, 1e-3,5,20,2);
    BiCGSTABSolver<BVector> solver(fop, prec0, tol,max_iter,0);
    solver.apply(x, rhs, res);
    // printvector(std::cout, x, "solution", "row");
    // printvector(std::cout, rhs, "final rhs", "row");

    for (int i = 0; i < n; i++){
        dx1[i] = x[i][0];
        dx2[i] = x[i][1];
    }
        
    // printvector(std::cout, rhs_final, "final residual", "row");
    // std::cout << "Solving time:" << res.elapsed << std::endl;

    return 1;
}

