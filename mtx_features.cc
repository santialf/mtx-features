#include <iostream>
#include <cmath>
#include "sparsebase/bases/iobase.h"

using namespace sparsebase;
using namespace bases;

/*                                                                               */
/* Computation of the imbalance factor (max nnzs in a warp / avg nnzs in a warp) */
/*                                                                               */
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
    local_max = 0;
  }

  avg = avg/warps;

  return (float) max/avg;
}

/*                                                                               */
/* Computation of the imbalance factor inside each warp and then average it out  */
/*                                                                               */
float warpImbalanceFactor(long int warps, long int n, int *row_ptr){
  
  float avg = 0, local_avg = 0, local_max = 0;

  for(int i = 0; i < warps; i++){
    for(int j = 0; j < 32; j++){
      if (i*32+j >= n)
        break;
      if (row_ptr[i*32+j+1]-row_ptr[i*32+j] > local_max)
        local_max = row_ptr[i*32+j+1]-row_ptr[i*32+j];
      local_avg += row_ptr[i*32+j+1]-row_ptr[i*32+j];
    }
    local_avg = local_avg/32;
    if (local_avg == 0)
      avg += 1;
    else
      avg += local_max/local_avg;
    local_avg = 0;
    local_max = 0;
  }
  
  return (float) avg/warps;
}

/*                                                                               */
/* Computation of the imbalance factor over all threads                          */
/*                                                                               */
float threadImbalanceFactor(long int warps, long int n, int *row_ptr){
  
  float avg = 0, max = 0;

  for(int i = 0; i < n; i++){
    if (row_ptr[i+1]-row_ptr[i] > max)
      max = row_ptr[i+1]-row_ptr[i];
    avg += row_ptr[i+1]-row_ptr[i];
  }

  avg = avg/n;
  
  return (float) max/avg;
}


long int bandwidth(long int n, int *row_ptr, int *cols){
  long int bw = 0;

  /* Loop over each row */
  for (int i = 0; i < n; i++) {
    /* Loop over the non-zero elements in the current row */
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
      /* Compute the absolute distance from the diagonal */
      long int dist = std::abs(i - cols[j]);
      if (dist>bw) {
        bw = dist;
      }
    }
  }
  return bw;
}

long int offDiagonal(long int n, int *row_ptr, int *cols, int blocks, long int nnz){
  long int rows_per_block = static_cast<int>(std::ceil(static_cast<double>(n) / blocks));
  long int count = 0;

  for (int i = 0; i < blocks; i++) {
    int left = 0, right = rows_per_block;
    for (int j = 0; j < rows_per_block; j++) {
      long int id = rows_per_block*i + j;
      if (id >= n)
        break;

      //printf("%d\n", id);
      for (int k = row_ptr[id]; k < row_ptr[id + 1]; k++) {
        if ((id-left <= cols[k]) && (cols[k] < id+right))
          count++;
      }
      left++;
      right--;
    }
  }
  return nnz-count;
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
  long int nnz = csr->get_num_nnz();
  auto *row_ptr = csr->get_row_ptr();
  auto *cols = csr->get_col();

  /* Get number of warps used */
  long int warps = static_cast<int>(std::ceil(static_cast<double>(n) / 32));

  for (int i = 1; i < argc; i++) {
    std::string function_name = argv[i];

    if (function_name == "--bandwidth") {
      /* Compute bandwidth */
      long int bw;
      bw = bandwidth(n, row_ptr, cols);
      std::cout<<"bandwidth: "<<bw<<std::endl;

    } else if (function_name == "--offDiagonal") {
      /* Compute off diagonal nnzs */
      long int offCount;
      offCount = offDiagonal(n, row_ptr, cols, 64, nnz);
      std::cout<<"off Diagonal nnzs: "<<offCount<<std::endl;

    } else if (function_name == "--imbWarp") {
      /* Compute imbalance factor */
      float imb_factor;
      imb_factor = imbalanceFactor(warps, n, row_ptr);
      std::cout<<"imbalance factor across warps: "<<imb_factor<<std::endl;

    } else if (function_name == "--imbThread") {
      /* Compute thread imbalance factor */
      float thread_imb_factor;
      thread_imb_factor = threadImbalanceFactor(warps, n, row_ptr);
      std::cout<<"imbalance factor across threads: "<<thread_imb_factor<<std::endl;

    } else if (function_name == "--imbInsideWarp") {
      /* Compute inside warp imbalance factor */
      float warp_imb_factor;
      warp_imb_factor = warpImbalanceFactor(warps, n, row_ptr);
      std::cout<<"average imbalance factor inside a warp: "<<warp_imb_factor<<std::endl;
    }
  }

  /*int rows_per_thread = static_cast<int>(std::ceil(static_cast<double>(n)/221184));
  int remaining = n%221184;
  std::cout<<rows_per_thread<<" "<<remaining<<std::endl;*/

  return 0;
}