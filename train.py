import os

def main():
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    rank = int(os.getenv('RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE', -1))

    # Your main script logic here
    print(f"Hello from process {local_rank}, {rank} out of {world_size}")

if __name__ == "__main__":
    main()
