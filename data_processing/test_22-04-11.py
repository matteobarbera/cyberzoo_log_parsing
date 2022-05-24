import log_dirs
from funcs import *


if __name__ == "__main__":
    # 3 takes at phase 0, 120, 240 no optitrack
    takes = get_all_files(log_dirs.log_dirs["22-04-11"])

    # Mag near wire and MAG_UPDATE_ALL_AXIS=FALSE
    other_takes = get_all_files(log_dirs.log_dirs["22-04-05"])[-2:]

    position_cardinal_2d(takes, use_cmaps=True)
    # position_cardinal_2d(other_takes)

    # plot_position(takes)

    # plot_euler(takes[2])

    intervals = [(5, 30), (5, 20), (5, 21)]
    phase = [0, 120, 240]
    plot_bearing(takes, intervals=intervals)
    # plot_bearing(other_takes)

    plt.show()
