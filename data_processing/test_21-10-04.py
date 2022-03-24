import log_dirs
from funcs import *


if __name__ == "__main__":
    # Take 0, 1 square loop with 300/600Hz mag respectively
    # Take 2-14 static elevon tests
    takes = get_all_files(log_dirs.log_dirs["21-10-04"])

    # plot_euler(takes[3], steady_state=True)

    interval = (414.52, 446)
    compare_attitude(takes[3], "..//pickle_data//att_21-10-4.pickle", interval=interval)
    plt.show()
