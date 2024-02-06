#include <iostream>
#include <cmath>
#include "sparsebase/bases/iobase.h"

using namespace sparsebase;
using namespace bases;

/*                                                                              */
/* Computation of the imbalance factor (max nnzs in a warp / avg nnzs in a warp)*/
/*                                                                              */
float imbalanceFactor(long int warps, long int n, int *row_ptr){
  long int local_max = 0, max = 0, avg = 0;

  for(int i = 0; i < warps; i++){
    for(int j = 0; j < 32; j++){
      if (i*32+j >= n)
        break;
      if (row_ptr[i*32+j+1]-row_ptr[i*32+j] > local_max)
        local_max = row_ptr[i*32+j+1]-row_ptr[i*32+j];
    }
    if (local_max > max)
      max = local_max;
    avg += local_max;
  }

  avg = avg/warps;

  return (float) max/avg;
}

int main(int argc, char * argv[]){

  if (argc < 2){
    std::cout << "Please enter the name of the edgelist file as a parameter\n";
    return 1;
  }

  /* Read mtx file to CSR format */
  auto csr = IOBase::ReadMTXToCSR<int,int,float>(argv[1]);

  /* Get CSR parameters */
  long int n = csr->get_dimensions()[0];
  int *row_ptr = csr->get_row_ptr();

  /* Get number of warps used */
  long int warps = static_cast<int>(std::ceil(static_cast<double>(n) / 32));
  float imb_factor;
  std::cout<<"number of warps: "<<warps<<std::endl;

  /* Compute imbalance factor */
  imb_factor = imbalanceFactor(warps, n, row_ptr);
  std::cout<<"imbalance factor: "<<imb_factor<<std::endl;

  return 0;
}