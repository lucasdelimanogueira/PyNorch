//#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include "distributed.h"
#include <mpi.h>

extern "C" {
  void init_process_group(int rank, int world_size) {

      MPI_Init(NULL, NULL);
      
      int id;

      if (rank == 0) {
          id = 123;
      }
      printf("O id é: %d no rank %d/%d", id, rank, world_size);
      MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
      printf("\n\n");
      printf("O id é: %d no rank %d/%d", id, rank, world_size);
      MPI_Finalize();
  }
}

int main(){
  init_process_group(5, 10);
}