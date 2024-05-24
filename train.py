import os
import norch
import norch.distributed as dist

def main():

    local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK', -1))
    rank = int(os.getenv('OMPI_COMM_WORLD_RANK', -1))
    world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', -1))

    dist.init_process_group(rank, world_size)

    tensor = norch.Tensor([1,2,3])
    tensor = rank * tensor
    print(f"BEFORE on rank {rank}: ", tensor, '\n\n')

    dist.broadcast_tensor(tensor)

    print(f"AFTER BROADCAST on rank {rank}: ", tensor, '\n\n')

    dist.allreduce_mean_tensor(tensor)

    print(f"AFTER ALLREDUCE MEAN on rank {rank}: ", tensor, '\n\n')

if __name__ == "__main__":
    main()
