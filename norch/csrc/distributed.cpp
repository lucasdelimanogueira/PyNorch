#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include "distributed.h"
#include "cuda.h"
#include <mpi.h>
#include <nccl.h>

int rank;
int world_size;
ncclComm_t nccl_comm;

extern "C" {

void init_process_group(int env_rank, int env_world_size) {

    rank = env_rank;
    world_size = env_world_size;

    // init MPI
    MPI_CHECK(MPI_Init(NULL, NULL));


    ncclUniqueId nccl_id;

    if (rank == 0) {
        ncclGetUniqueId(&nccl_id);
    }

    // send nccl unique id to all processes
    MPI_CHECK(MPI_Bcast((void *)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));

    // init NCCL communication group
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, world_size, nccl_id, rank));

}

void broadcast_tensor(Tensor* tensor) {

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    NCCL_CHECK(ncclBroadcast(tensor->data, tensor->data, tensor->size * sizeof(float), ncclFloat, 0, nccl_comm, stream));
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

void allreduce_sum_tensor(Tensor* tensor) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Perform NCCL AllReduce operation to calculate the sum of all tensors across all processes
    NCCL_CHECK(ncclAllReduce(tensor->data, tensor->data, tensor->size, ncclFloat, ncclSum, nccl_comm, stream));

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

void end_process_group() {
    MPI_CHECK(MPI_Finalize());
    NCCL_CHECK(ncclCommDestroy(nccl_comm));
}

}
