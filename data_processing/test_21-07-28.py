import numpy as np

import log_dirs
from funcs import *


def f4(filename: list):
    data = extract_take_data(filename[4])
    plot_position(data)


if __name__ == "__main__":
    # Take 0-3 from rest to equilibrium, different phase angles
    # Take 4 after equilibrium, change of phase 180 degrees
    takes = get_all_files(log_dirs.log_dirs["21-07-28"])

    # plot_position(takes[:4])
    # position_cardinal_2d(takes[:4], last_secs=10, use_cmaps=True)
    # position_cardinal_3d(takes[:4], last_secs=10, z_pos_up=True, use_cmaps=True)

    origin = np.array([360, -230, 0])  # ???? FIXME
    points = np.array([[565, -395],  # phase 0 deg
                       [185, -435],  # phase 90 deg
                       [166, -65],  # phase 180 deg
                       [540, -45]])  # phase 270 deg
    dist_computations(points, origin)

    # plot_position(takes[4])
    plt.show()

