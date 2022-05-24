import seaborn as sns
import log_dirs
from funcs import *


def unpickle_data():
    directory = "..//pickle_data//"
    p_files = sorted([f for f in os.listdir(directory) if "spin" in f])
    data = []
    for p in p_files:
        with open(directory + p, "rb") as f:
            data.append(pickle.load(f))
    return data


def mag_psi_vs_opti_psi(takes, offset=0):
    spin_data = unpickle_data()

    spin_n = 2

    tdata = extract_take_data(takes[spin_n])
    euler = my_quaternion_to_euler(tdata["Quaternion"])

    ac_data = spin_data[spin_n]

    transformed_mag = transform_mag_to_control_frame(ac_data, my_offset=offset)

    psi = np.degrees(np.arctan2(transformed_mag[:, 1], transformed_mag[:, 0]))

    mag_t = ac_data["IMU_MAG"]["timestamp"]

    with sns.plotting_context('talk'), sns.axes_style('whitegrid'):
        plt.plot(mag_t, psi, lw=3, label="Mag PSI", alpha=0.8)
        plt.plot(tdata["Time"] + 522.98, np.degrees(euler[:, 2]), lw=3, label="Opti PSI", alpha=0.8)
        plt.legend()

    # mag_mx = ac_data["IMU_MAG"]["mx"]
    # mag_my = ac_data["IMU_MAG"]["my"] + 0.55
    # mag_mz = ac_data["IMU_MAG"]["mz"].flatten()

    # plt.figure()
    # plt.plot(mag_t, mag_mx, label='x')
    # plt.plot(mag_t, mag_my, label='y')
    # plt.plot(mag_t, mag_mz, label='z')
    # plt.legend()
    # plt.grid()

    plt.figure()
    plt.plot(mag_t, transformed_mag[:, 0], label='x', alpha=0.5)
    plt.plot(mag_t, transformed_mag[:, 1], label='y', alpha=0.5)
    plt.plot(mag_t, transformed_mag[:, 2], label='z', alpha=0.5, zorder=0)
    plt.legend()
    plt.grid()
    plt.show()


def my_holt_winters(data_arr, alpha, beta, gamma, season_l, season_def=0.01):
    # Initialization
    st = data_arr[0]
    hw = [st]
    seasonal_arr = [season_def] * season_l  # 0.01
    bt = 0
    for i in range(len(data_arr) - 1):
        prev_st = st
        x = data_arr[i + 1]
        ct = seasonal_arr[i % season_l]

        st = alpha * (x - ct) + (1 - alpha) * (prev_st + bt)
        bt = beta * (st - prev_st) + (1 - beta) * bt
        seasonal_arr[i % season_l] = gamma * (x - st - bt) + (1 - gamma) * ct

        hw.append(st)
    return hw


def filter_ins_velocity(takes):
    # TODO Calculate residuals, change parameters for lowest resid over ALL spins
    spin_data = unpickle_data()
    for spin_n in range(len(spin_data)):
        # spin_n = 2

        ac_data = spin_data[spin_n]

        gyro_gr = ac_data["IMU_GYRO"]["gr_alt"]

        gyro_alpha = 0.045
        gyro_filt = [gyro_gr[0]]
        for i in range(len(gyro_gr) - 1):
            gyro_filt.append(gyro_alpha * gyro_gr[i + 1] + (1 - gyro_alpha) * gyro_filt[i])

        f = 100
        seasonal_cycle_l = np.round(2 * np.pi * f / np.radians(gyro_filt))
        seasonal_cycle_l = np.asarray(seasonal_cycle_l, dtype=int).reshape(-1)
        l = int(round(np.average(seasonal_cycle_l)))

        ins_t = ac_data["INS"]["timestamp"]

        ins_vx = ac_data["INS"]["ins_xd_alt"]
        ins_vy = ac_data["INS"]["ins_yd_alt"]

        alpha = 0.99
        alpha2 = 0.92

        hor_v = np.sqrt(ins_vx ** 2 + ins_vy ** 2)

        expanded_hor_v = np.array([])
        expanded_ins_t = np.array([])
        for i in range(len(hor_v) - 1):
            expanded_hor_v = np.r_[expanded_hor_v, np.linspace(hor_v[i], hor_v[i + 1], num=5, endpoint=False)]
            expanded_ins_t = np.r_[expanded_ins_t, np.linspace(ins_t[i], ins_t[i + 1], num=5, endpoint=False)]
        else:
            expanded_hor_v = np.append(expanded_hor_v, expanded_hor_v[-1])
            expanded_ins_t = np.append(expanded_ins_t, expanded_ins_t[-1])
        # plt.figure()
        # plt.plot(expanded_ins_t, expanded_hor_v, marker='x', markevery=1)
        # plt.show()
        # quit()

        filtered_v = [hor_v[0]]
        filtered_com = [hor_v[0]]

        for i in range(len(hor_v) - 1):
            filtered_v.append(filtered_v[i] * alpha + (1 - alpha) * hor_v[i + 1])
            filtered_com.append(filtered_com[i] * alpha2 + (1 - alpha2) * hor_v[i + 1])

        # alpha = 0.016
        # beta = 0.08
        # gamma = 0.35

        alpha = 0.014
        beta = 0.14
        gamma = 0.7

        holt_winters = my_holt_winters(hor_v, alpha, beta, gamma, l)

        fs_t = ac_data["FRISBEE_CONTROL"]["timestamp"]
        fs_hor = np.sqrt((ac_data["FRISBEE_CONTROL"]["hor_norm"] * 0.0000019) ** 2 + (ac_data["FRISBEE_CONTROL"]["hor_norm"] * 0.0000019) ** 2)

        thick = 3
        # plt.figure()
        # plt.plot(ins_t, hor_v, alpha=0.6)
        # plt.plot(ins_t, filtered_v, label=f"EWMA alpha={0.01}", color='m', lw=thick)
        # plt.plot(ins_t, filtered_com, alpha=0.7, label=f"EWMA alpha={0.08}", color='g', lw=thick)
        # plt.plot(ins_t, holt_winters, color='k', label="Holt-Winters", lw=thick)
        # # plt.plot(fs_t, fs_hor, alpha=0.6)
        # # plt.plot(tdata["Time"][:-1] + 522.7, opti_v)
        # plt.legend()
        # plt.grid()

        # ====== Test with higher frequency data ========
        holt_winters = my_holt_winters(expanded_hor_v, alpha, beta, gamma, l)
        hw2 = my_holt_winters(expanded_hor_v, 0.0012, 0.07, 0.7, l * 5)
        plt.figure()
        plt.plot(expanded_ins_t, expanded_hor_v, alpha=0.8, label="Linspace expanded data")
        plt.plot(expanded_ins_t, holt_winters, color='k', label="Holt-Winters", lw=thick, alpha=0.7, zorder=0)
        plt.plot(expanded_ins_t, hw2, label="Holt-Winters", lw=thick)
        plt.legend()
        plt.grid()

        # ==== Test with longer sequence =====
        # hor_v = np.r_[expanded_hor_v, expanded_hor_v, expanded_hor_v]
        # # hor_v = np.r_[hor_v, hor_v, hor_v]
        # t = ins_t[-1] - ins_t[0]
        # ins_t = np.r_[expanded_ins_t, expanded_ins_t + t, expanded_ins_t + t*2]
        # # ins_t = np.r_[ins_t, ins_t + t, ins_t + 2 * t]
        #
        # holt_winters = my_holt_winters(hor_v, 0.0012, 0.07, 0.7, l * 5)
        #
        # plt.figure()
        # thick = 3
        # plt.plot(ins_t, hor_v, alpha=0.6)
        # plt.plot(ins_t, holt_winters, color='k', label="Holt-Winters", lw=thick)
        # plt.legend()
        # plt.grid()
    plt.show()


def get_optitrack_velocity_hor(take_data):
    opti_t = take_data["Time"]
    opti_p = take_data["Position"] * 0.001

    hor_p = np.sqrt(opti_p[:, 0] ** 2 + opti_p[:, 1] ** 2)
    hor_dp = np.diff(hor_p)
    dt = np.diff(opti_t)
    vel = hor_dp / dt
    return vel


if __name__ == "__main__":
    take_files = get_all_files(log_dirs.log_dirs["22-03-02"])
    take_files = take_files[-5:]

    mag_psi_vs_opti_psi(take_files, offset=0.54)
    # filter_ins_velocity(take_files)
