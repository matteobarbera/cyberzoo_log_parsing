import log_dirs
from funcs import *


if __name__ == "__main__":
    takes = get_all_files(log_dirs.log_dirs["22-05-19"])

    plot_bearing(takes, origin=[-160, 310])
    plot_position(takes)
    plt.show()
