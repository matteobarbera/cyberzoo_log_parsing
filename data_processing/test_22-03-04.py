import log_dirs
from funcs import *


if __name__ == "__main__":
    take_files = get_all_files(log_dirs.log_dirs["22-03-04"])

    lat_force_f = [take_files[i] for i in [0, 2, 3]]

    position_cardinal_2d(take_files[:-2], use_cmaps=True)
    # plot_distance_from_origin(lat_force_f)
    plot_distance_from_origin(take_files[:-2])

    intervals = [(4, 11), (8, 20), (0, 20), (0, 20)]
    plot_bearing(take_files[:-2], intervals=intervals, phase=[0, 0, 90, 180])

    plt.show()
