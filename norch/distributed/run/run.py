import sys
import os
import argparse
import multiprocessing

def worker(rank, nnodes, world_size, script_name, script_args):
    os.environ['LOCAL_RANK'] = str(rank // nnodes)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    script_args = [sys.executable, script_name] + script_args
    os.execvp(script_args[0], script_args)

def main():
    parser = argparse.ArgumentParser(description="Distributed runner")
    parser.add_argument('--nproc_per_node', type=int, required=True, help='Number of processes')
    parser.add_argument('--nnodes', type=int, required=False, default=1, help='Number of nodes')
    parser.add_argument('script', type=str, help='The script to run')
    parser.add_argument('script_args', nargs=argparse.REMAINDER, help='Arguments for the script')

    args = parser.parse_args()

    nproc_per_node = args.nproc_per_node
    nnodes = args.nnodes
    script_name = args.script
    script_args = args.script_args

    world_size = nproc_per_node * nnodes

    processes = []
    for rank in range(nproc_per_node):
        p = multiprocessing.Process(target=worker, args=(rank, nnodes, world_size, script_name, script_args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    main()
