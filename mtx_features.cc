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


float bandwidth(long int n, int *row_ptr, int *cols){
  float bw = 0;

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
  return bw/n;
}

float diagonal(long int n, int *row_ptr, int *cols){
  float b = 32;
  float count = 0;

  for (int i = 0; i < n; i++) {
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
      if (abs(cols[j]-i) <= b)
        count++;
    }
  }
  return count;
}

float blockDiagonal(long int n, int *row_ptr, int *cols){
  float b = 32;
  long int blocks = static_cast<int>(std::ceil(static_cast<double>(n) / b));
  float count = 0;

  for (int i = 0; i < blocks; i++) {
    int left = 0, right = b;
    for (int j = 0; j < b; j++) {
      long int id = b*i + j;
      if (id >= n)
        break;

      for (int k = row_ptr[id]; k < row_ptr[id + 1]; k++) {
        if ((cols[k] >= id-left) && (cols[k] <= id+right))
          count++;
      }
      left++;
      right--;
    }
  }

  return count;
}

float cacheOut(long int n, int *row_ptr, int *cols){

  int max = 0;
  long int blocks = 32;
  float b = static_cast<int>(std::ceil(static_cast<double>(n) / blocks));
  float localCount = 0;
  float total = 0;

  for (int i = 0; i < blocks; i++) {
    long int left = 0, right = b;
    while (left <= n) {
      for (int j = 0; j < b; j++) {
        long int id = b*i + j;
        if (id >= n)
          break;

        for (int k = row_ptr[id]; k < row_ptr[id + 1]; k++) {
          if ((cols[k] >= left) && (cols[k] <= right))
            localCount++;
          if (cols[k] > right)
            break;
        }
      }
      if (localCount > max)
        max = localCount;
      left+=b;
      right+=b;
      localCount = 0;
    }
    total += max;
    max = 0;
  }
  return total;
}

long int blocks(long int n, int *row_ptr, int *cols, int block_size, long int nnz){

  int num_blocks = static_cast<int>(std::ceil(static_cast<double>(n) / block_size));
  int sum = 0;

  std::unordered_map<int, int> hashMap;

  for (int i = 0; i < num_blocks; i++) {
    for (int j = 0; j < block_size; j++) {
      long int id = block_size*i + j;
      if (id >= n)
        break;
      for (int k = row_ptr[id]; k < row_ptr[id + 1]; k++) {
        int bucket = static_cast<int>(std::ceil(static_cast<double>(cols[k]) / block_size));
        if (((cols[k]) % block_size) != 0)
          bucket--;
        auto it = hashMap.find(bucket);
        if (it == hashMap.end())
          hashMap.insert({bucket, 1});
      }
    }
    sum += hashMap.size();
    hashMap.clear();
  }
  return sum;
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
      float bw;
      bw = bandwidth(n, row_ptr, cols);
      std::cout<<"bandwidth: "<<bw<<std::endl;

    } else if (function_name == "--diagonal") {
      /* Compute off diagonal nnzs */
      float dcount;
      dcount = diagonal(n, row_ptr, cols);
      dcount = (dcount/nnz)*100;
      std::cout<<"diagonal nnzs (%): "<<dcount<<std::endl;

    } else if (function_name == "--blockDiagonal") {
      /* Compute off diagonal nnzs */
      float bdcount;
      bdcount = blockDiagonal(n, row_ptr, cols);
      bdcount = (bdcount/nnz)*100;
      std::cout<<"blockDiagonal nnzs (%): "<<bdcount<<std::endl;

    } else if (function_name == "--cacheOut") {
      /* Compute off diagonal nnzs */
      float mcount;
      mcount = cacheOut(n, row_ptr, cols);
      mcount = (mcount/nnz)*100;
      std::cout<<"movable diagonal nnzs (%): "<<mcount<<std::endl;

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

    } else if (function_name == "--nnzs") {
      std::cout<<"nnzs: "<<nnz<<std::endl;

    } else if (function_name == "--blocks") {
      int num_blocks = blocks(n, row_ptr, cols, 256, nnz);
      std::cout<<"number of blocks with nnzs: "<<num_blocks<<std::endl;

    }
  }

  /*int rows_per_thread = static_cast<int>(std::ceil(static_cast<double>(n)/221184));
  int remaining = n%221184;
  std::cout<<rows_per_thread<<" "<<remaining<<std::endl;*/

  return 0;
}