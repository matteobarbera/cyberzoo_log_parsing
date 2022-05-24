import log_dirs
from funcs import *


if __name__ == "__main__":
    takes = get_all_files(log_dirs.log_dirs["22-04-01"])

    phase = list(range(0, 360, 15))
    intervals = [(5, 15), (0, 10), (0, 15), (0, 4), (0, 5), (0, 5), (0, 10), (0, 10), (0, 10), (3, 12), (2, 8), (2, 10),
                 (10, 20), (2, 12), (3, 12), (2, 10), (3, 10), (3, 15), (2, 12), (2, 12), (2, 10), (2, 12), (2, 12)]
    print(intervals)
    print(phase)

    # plot_bearing(takes, intervals=intervals)
    plot_distance_from_origin(takes)
    # for t in takes:
    #     plot_bearing(t)
    #     plt.show()

    plt.show()
