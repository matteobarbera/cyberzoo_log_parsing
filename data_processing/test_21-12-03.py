import log_dirs
from funcs import *


if __name__ == "__main__":
    # Take 0 cyclic [-0.65 0.6]+-0.1
    # Take 1 cyclic [-0.65 0.6]+-0.2
    # Take 2 cyclic [-0.55 0.5]+-0.2
    # Take 3 cyclic [-0.65 0.6]+-0.3
    # Take 4 square loop [-0.65 0.6]+-0.25
    takes = get_all_files(log_dirs.log_dirs["21-12-03"])
    prev_motors_take = get_all_files(log_dirs.log_dirs["21-07-28"])[0]
    prev_motor_square = get_all_files(log_dirs.log_dirs["21-09-07"])[30]

    # superimpose_position(takes[:4])
    superimpose_position([takes[3], prev_motors_take])  # old vs new motors

    # rope length ~4.230m
    data = extract_take_data(takes[4])
    origin = data["Position"][0, :]
    print(origin[-1])
    corners = np.array([[520, 759],
                        [-126, 281],
                        [447, -366],
                        [1053, 139]])

    dist_computations(corners, origin, data["Position"], weight=0.9)  # FIXME weight + something else?

    # plot_position(takes[4])
    superimpose_position([takes[4], prev_motor_square])
    plt.show()
