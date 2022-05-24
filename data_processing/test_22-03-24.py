import log_dirs
from funcs import *


if __name__ == "__main__":
    # 0, 1 MAG cmpl filter tuning
    # 2
    # 3
    # 4 square loop for ins velocity filter
    # 5-12 advance angle test, two cyclic settings
    takes = get_all_files(log_dirs.log_dirs["22-03-24"])

    # position_cardinal_2d(takes[5:], use_cmaps=True, colorbar=False)
    # Slightly shifted but maybe rotation not exactly center,
    # Perhaps thrust difference effect changing with motor strength
    # position_cardinal_2d(takes[5:9], use_cmaps=True)

    intervals = [(0, 20), (0, 14), (2, 14), (11, 25), (0, 17), (0, 16), (7, 18), (5, 22)]
    phase_arr = [0, 90, 180, 270, 0, 90, 180, 270]
    plot_bearing(takes[5:9], intervals=intervals[:4], phase=phase_arr[:4], title="High rotation speed (rev mot 75% thr)")
    plot_bearing(takes[9:], intervals=intervals[4:], phase=phase_arr[4:], title="Low rotation speed (rev mot 55% thr)")
    plt.show()
