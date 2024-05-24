import sys
import os
import argparse
import subprocess

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

    mpiexec_command = [
        'mpiexec',
        '-n', str(world_size),
        sys.executable, script_name
    ] + script_args

    subprocess.run(mpiexec_command)

if __name__ == '__main__':
    main()
