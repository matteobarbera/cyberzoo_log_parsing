from funcs import *
import log_dirs

if __name__ == "__main__":
    filenames = get_all_files(log_dirs.log_dirs["21-07-09"])
    data = extract_take_data(filenames[-1])
    plot_euler(data)
