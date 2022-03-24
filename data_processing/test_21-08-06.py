import log_dirs
from funcs import *


if __name__ == "__main__":
    takes = get_all_files(log_dirs.log_dirs["21-08-06"])

    origin_take = takes[0]
    adv_ccw_takes = takes[1:5]
    adv_cw_takes = takes[5:9]
    orientation_take = takes[9]
    ol_square_take = takes[10]
    ol_square_stable_take = takes[11]
    ol_square_reverse_take = takes[12]

    # plot_position(origin_take)  # origin -> [373, -230]
    # position_cardinal_2d(adv_ccw_takes, last_secs=20)  # ---------
    # position_cardinal_2d(adv_ccw_takes)  # -------
    # position_cardinal_2d(adv_cw_takes, last_secs=20)  # ----
    # plot_euler(extract_take_data(orientation_take))
    # plot_position(ol_square_take)  # ----
    # plot_euler(ol_square_take)
    # plot_position(ol_square_stable_take)
    # plot_position(ol_square_reverse_take)  # ----
    plot_euler(ol_square_reverse_take)
    plt.show()
