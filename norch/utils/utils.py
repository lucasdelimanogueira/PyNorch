import random
import requests
import tarfile
import zipfile
import shutil
import os

def generate_random_list(shape):
    """
    Generate a list with random numbers and shape 'shape'
    """
    if len(shape) == 0:
        return []
    else:
        inner_shape = shape[1:]
        if len(inner_shape) == 0:
            return [random.uniform(-1, 1) for _ in range(shape[0])]
        else:
            return [generate_random_list(inner_shape) for _ in range(shape[0])]


def download_from_url(url, save_path, chunk_size=128):
    """Download a file from an URL.

    Original answer from https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url

    Args:
        url (str): path to the URL.
        save_path (str): path to the saving directory.
        chunk_size (int): download chunk.

    Returns:
        None
    """
    response = requests.get(url, stream=True)
    total = response.headers.get('content-length')
    with open(save_path, 'wb') as f:
        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                downloaded += len(data)
                f.write(data)
                progress_bar(downloaded, total, "Downloading...")


def extract_to_dir(filename, dirpath='.'):
    # Does not create folder twice with the same name
    name, ext = os.path.splitext(filename)
    # if os.path.basename(name) == os.path.basename(dirpath):
    #     dirpath = '.'
    # Extract
    print(dirpath)
    print("Extracting...", end="")
    if tarfile.is_tarfile(filename):
        tarfile.open(filename, 'r').extractall(dirpath)
    elif zipfile.is_zipfile(filename):
        zipfile.ZipFile(filename, 'r').extractall(dirpath)
    elif ext == '.gz':
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        shutil.move(filename, os.path.join(dirpath, os.path.basename(filename)))
        print(f" | NOTE: gzip files are not extracted, and moved to {dirpath}", end="")
    # Return the path where the file was extracted
    print(" | Done !")
    return os.path.abspath(dirpath)

def progress_bar(current_index, max_index, prefix=None, suffix=None, start_time=None):
    """Display a progress bar and duration.

    Args:
        current_index (int): current state index (or epoch number).
        max_index (int): maximal numbers of state.
        prefix (str, optional): prefix of the progress bar. The default is None.
        suffix (str, optional): suffix of the progress bar. The default is None.
        start_time (float, optional): starting time of the progress bar. If not None, it will display the time
            spent from the beginning to the current state. The default is None.

    Returns:
        None. Display the progress bar in the console.
    """
    # Add a prefix to the progress bar
    prefix = "" if prefix is None else str(prefix) + " "

    # Get the percentage
    percentage = current_index * 100 // max_index
    loading = "[" + "=" * (percentage // 2) + " " * (50 - percentage // 2) + "]"
    progress_display = "\r{0}{1:3d}% | {2}".format(prefix, percentage, loading)

    # Add a suffix to the progress bar
    progress_display += "" if suffix is None else " | " + str(suffix)

    # Add a timer
    if start_time is not None:
        time_min, time_sec = get_time(start_time, time.time())
        time_display = " | Time: {0}m {1}s".format(time_min, time_sec)
        progress_display += time_display

    # Print the progress bar
    # TODO: return a string instead
    print(progress_display, end="{}".format("" if current_index < max_index else " | Done !\n"))

def get_time(start_time, end_time):
    """Get ellapsed time in minutes and seconds.

    Args:
        start_time (float): strarting time
        end_time (float): ending time

    Returns:
        elapsed_mins (float): elapsed time in minutes
        elapsed_secs (float): elapsed time in seconds.
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs
