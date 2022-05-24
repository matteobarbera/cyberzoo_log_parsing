import log_dirs
from funcs import *


if __name__ == "__main__":
    # 3 takes normal at phase 0, 120, 240
    # 3 takes with tape on harness
    # 2 with no optitrack, observable drift in position
    takes = get_all_files(log_dirs.log_dirs["22-04-05"])

    position_cardinal_2d(takes[:6], use_cmaps=True)

    plot_position(takes[6:])

    plt.show()
