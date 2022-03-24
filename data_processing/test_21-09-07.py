import log_dirs
from funcs import *


if __name__ == "__main__":
    takes = get_all_files(log_dirs.log_dirs["21-09-07"])

    new_harness_take = takes[:2]  # second with stronger thrust
    # plot_position(new_harness_take)
    # plot_euler(new_harness_take)

    el_0_takes = takes[2:6]  # er 0.5  1  -0.5  -1
    el_05_takes = takes[6:9]  # er 0  -0.5  -1
    el_neg_05_takes = takes[9:12]  # er 0  0.5  1
    el_1_takes = takes[12:15]  # er 0  -0.5  -1
    el_neg_1_takes = takes[15:18]  # er 0  0.5  1

    # plot_euler(el_0_takes, steady_state=True)
    # plot_euler(el_05_takes, steady_state=True)
    # plot_euler(el_neg_05_takes, steady_state=True)
    # plot_euler(el_1_takes, steady_state=True)
    # plot_euler(el_neg_1_takes, steady_state=True)

    # All square cyclic to motors
    cyclic_ampl_no_elev_takes = takes[18:21]  # +- 0.1  0.2  0.3
    cyclic_ampl_el_neg_1_takes = takes[21:23]  # +- 0.2  0.3

    # plot_position(cyclic_ampl_no_elev_takes)
    # plot_position(cyclic_ampl_el_neg_1_takes)

    cyclic_avg_el_neg_1_takes = takes[23:25]  # [-0.9 0.6] +- 0.3  [-1 0.7] +- 0.3
    cyclic_higher_ampl_el_neg_1 = takes[25]  # [-0.8 0.5] +- 0.4

    # plot_position(cyclic_avg_el_neg_1_takes)
    # plot_position(cyclic_higher_ampl_el_neg_1)

    # Elevon cyclic given with SINE!
    er_cyclic_el_neg_1_takes = takes[26:29]  # er (0 +- 0.5) (0.5 +- 0.5) (1 +- 0.5)
    el_cyclic_er_0_takes = takes[29]  # el -1 +- 0.5

    # plot_position(er_cyclic_el_neg_1_takes)
    # plot_position(el_cyclic_er_0_takes)

    square_no_elevon_take = takes[30]  # mot [-0.9 0.6] +- 0.3
    plot_position(square_no_elevon_take)
    # plot_euler(square_no_elevon_take)
    square_el_neg_1_take = takes[31]  # mot [-0.9 0.6] +- 0.3
    # plot_position(square_el_neg_1_take)
    # plot_euler(square_el_neg_1_take)

    # make_table("../table_data/phi_theta_09-07.csv")

    plt.show()
