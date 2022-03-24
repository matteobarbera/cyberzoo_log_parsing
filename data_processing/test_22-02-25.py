import log_dirs
from funcs import *


if __name__ == "__main__":
    # 0 - Phase change 0 to 180, [-0.7, 0.56]+-0.24 not well balanced
    # 1 - No cyclic, motors same as take 0
    # 2 - No cyclic [-0.65, 0.5]
    # 3 - Phase change 0 to 180 [-0.65, 0.5]+-0.25 movement straighter
    take_files = get_all_files(log_dirs.log_dirs["22-02-25"])

    # plot_position(take_files[0])
    # plot_position(take_files[1])
    plot_position(take_files[2])
    plot_position(take_files[3])

    plt.show()

