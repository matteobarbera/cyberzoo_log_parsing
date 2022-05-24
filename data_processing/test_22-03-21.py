import log_dirs
from funcs import *


if __name__ == "__main__":
    # 0, 1 test for INS FILTER
    # 2, 3 elevon cyclic, 0.5 and 1 average
    # 4 motor then + elevon cyclic
    # 5 elevon different phase
    # 6 motor + elevon cyclic then - elevon ??
    # 7, 8, 9, 10 no opti
    # 11, 12, 13, 14 average vs delta tradeoff
    takes = get_all_files(log_dirs.log_dirs["22-03-21"])

    # elevon cyclic vs motor cyclic
    motor_comparison_takes = get_all_files(log_dirs.log_dirs["22-03-04"])[1:-2]
    # position_cardinal_2d(motor_comparison_takes + [takes[5]], use_cmaps=True)

    # plot_distance_from_origin(takes[11:], origin=(-30, 260))

    # plot_position(takes[7:11])

    phase = [0, 0, 0, 0]
    intervals = [(3, 13), (3, 14), (10, 25), (4, 25)]
    plot_bearing(takes[11:], intervals=intervals, phase=phase, title="Cyclic phase 0 at different rotation speeds", context="talk")

    plt.show()
