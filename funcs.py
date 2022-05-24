import os
from itertools import cycle
from functools import wraps


import numpy as np
import quaternion as qt
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from celluloid import Camera
from IPython.display import HTML
from bs4 import BeautifulSoup
import pandas as pd
import pickle
from csv import reader
import seaborn as sns


def seaborn_style():

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            if "context" in kwargs.keys():
                _context = kwargs["context"]
                del kwargs["context"]
            else:
                _context = "notebook"

            if "style" in kwargs.keys():
                _style = kwargs["style"]
                del kwargs["style"]
            else:
                _style = "whitegrid"

            if "params" in kwargs.keys():
                _params = kwargs["params"]
                del kwargs["params"]
            else:
                _params = None

            _default_params = {
              # "xtick.bottom": True,
              # "ytick.left": True,
              # "xtick.color": ".8",  # light gray
              # "ytick.color": ".15",  # dark gray
              "axes.spines.left": False,
              "axes.spines.bottom": False,
              "axes.spines.right": False,
              "axes.spines.top": False,
              }
            if _params is not None:
                merged_params = {**_params, **_default_params}
            else:
                merged_params = _default_params
            with sns.plotting_context(context=_context), sns.axes_style(style=_style, rc=merged_params):
                func(*args, **kwargs)
        return wrapper

    return decorator


def get_all_files(directory: str):
    if directory[-1] != "/":
        directory += "/"
    am_files = [directory + fname for fname in os.listdir(directory) if "AM" in fname and "~lock" not in fname]
    pm_files = [directory + fname for fname in os.listdir(directory) if "PM" in fname and "~lock" not in fname]
    filenames = sorted(am_files) + sorted(pm_files)
    return filenames


def get_csv_cols(filename):
    with open(filename, 'r') as f:
        rdr = reader(f)
        for i, line in enumerate(rdr):
            if i == 3:
                cols = []
                for j, col_name in enumerate(line):
                    if col_name == "RigidBody 1" or col_name == "testbed 1":
                        cols.append(j)
                return cols


def extract_take_data(filename):
    cols = [1] + get_csv_cols(filename)
    data = np.genfromtxt(filename, skip_header=7, delimiter=',', usecols=cols, dtype=float)

    data = data[~np.isnan(data).any(axis=1)]
    # Take file quaternion [x, y, z, w] position [x fwd, y up, z right]
    # My dict quaternion [w x y z] position [x fwd y right z down]
    zyz_rotation = qt.from_euler_angles(np.radians([90, -90, -90]))  # from CZ to AE
    d = {"Filename": filename,
         "Time": data[:, 0],
         "Quaternion": qt.as_quat_array(data[:, [4, 1, 3, 2]] * np.array([1, 1, 1, -1])),
         "Position": qt.rotate_vectors(zyz_rotation, data[:, 5:8]),
         "dt": data[1, 0],
         "Hz": 1 / data[1, 0]
         }
    return d


def transform_mag_to_control_frame(ac_data, my_offset=0):
    phi = ac_data["ATTITUDE"]["phi"].flatten()
    theta = att_theta = ac_data["ATTITUDE"]["theta"].flatten()

    # R(phi)R(theta)
    rot_mat_row1 = np.asarray([np.cos(att_theta), np.sin(phi) * np.sin(att_theta), np.cos(phi) * np.sin(theta)]).T
    rot_mat_row2 = np.asarray([np.zeros(theta.shape[0]), np.cos(phi), -np.sin(phi)]).T
    rot_mat_row3 = np.asarray([-np.sin(att_theta), np.sin(phi) * np.cos(att_theta), np.cos(phi) * np.cos(theta)]).T

    # Shape (n, 3, 3)
    rot_mat = np.dstack((rot_mat_row1, rot_mat_row2, rot_mat_row3))
    rot_mat = rot_mat.transpose((0, 2, 1))

    mag_mx = ac_data["IMU_MAG"]["mx"]
    mag_my = ac_data["IMU_MAG"]["my"] + my_offset
    mag_mz = ac_data["IMU_MAG"]["mz"].flatten()
    # Shape (n, 3)
    mag_vec = np.c_[mag_mx, mag_my, mag_mz]

    mag_prime = np.einsum('ijk,ik->ij', rot_mat, mag_vec)

    return mag_prime


def my_quaternion_to_euler(q):
    float_arr = qt.as_float_array(q).reshape(-1, 4)
    q0 = float_arr[:, 0]
    q1 = float_arr[:, 1]
    q2 = float_arr[:, 2]
    q3 = float_arr[:, 3]

    phi = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 + q2 ** 2))
    theta = np.arcsin(2 * (q0 * q2 - q3 * q1))
    psi = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))
    return np.c_[phi, theta, psi]


def mean_angle_estimation(phi: np.ndarray):
    x = np.cos(phi)
    y = np.sin(phi)
    n = phi.size

    x_bar = np.sum(x) / (1 / n)
    y_bar = np.sum(y) / (1 / n)

    if x_bar > 0:
        phi_bar = np.arctan(y_bar / x_bar)
    elif x_bar < 0:
        phi_bar = np.arctan(y_bar / x_bar) + np.pi
    elif x_bar == 0 and y_bar > 0:
        phi_bar = np.pi / 2
    elif x_bar == 0 and y_bar < 0:
        phi_bar = 3 / 2 * np.pi
    else:
        raise ValueError("Mean angle undefined")

    return phi_bar % (2 * np.pi)


def plot_euler(take_files: list, steady_state: bool = False):
    if not isinstance(take_files, list):
        take_files = [take_files]

    for take in take_files:
        data = extract_take_data(take)
        t = data["Time"]
        euler = my_quaternion_to_euler(data["Quaternion"])
        deg_euler = np.rad2deg(euler)

        fig, axs = plt.subplots(3, 1)
        axs[0].plot(t, deg_euler[:, 0], label="phi")
        # axs[0].axhline(y=np.median(deg_euler[:, 0]), xmin=t[0], xmax=t[-1], color="r")
        axs[1].plot(t, deg_euler[:, 1], label="theta")
        axs[2].plot(t, deg_euler[:, 2], label="psi")
        if steady_state:
            phi_ss = recursive_median(deg_euler[:, 0])
            theta_ss = recursive_median(deg_euler[:, 1])
            axs[0].axhline(y=phi_ss, xmin=t[0], xmax=t[-1], color="k")
            axs[1].axhline(y=theta_ss, xmin=t[0], xmax=t[-1], color="k")
            print(f"{take}:\nPhi: {phi_ss}\tTheta: {theta_ss}\n")

        for ax in axs.reshape(-1):
            ax.legend()
            ax.grid()
            ax.set_xlabel("Time [s]")
        axs[0].set_ylabel("Phi [deg]")
        axs[1].set_ylabel("Theta [deg]")
        axs[2].set_ylabel("Psi [deg]")
        axs[2].set_yticks(list(range(-200, 201, 100)))
        return axs


def recursive_median(arr: np.ndarray, thr: float = 0.3,  n: int = 3):
    median = np.median(arr)
    _thr = thr
    for _ in range(n):
        if median > 0:
            mask = (median * (1 - _thr) < arr) & (arr < median * (1 + _thr))
        else:
            mask = (median * (1 + _thr) < arr) & (arr < median * (1 - _thr))
        median = np.median(arr[mask])
        _thr *= thr
    return median


def plot_position(take_files: list, z_pos_up: bool = False, plot_every: int = 5):
    # TODO Add first secs / last secs
    if not isinstance(take_files, list):
        take_files = [take_files]

    for t in take_files:
        data = extract_take_data(t)
        if z_pos_up:
            data["Position"] *= [1, 1, -1]  # Flip z-axis

        mappable = cm.ScalarMappable(cmap=cm.inferno)
        mappable.set_array(data["Time"])

        fig = plt.figure(figsize=(13, 6))
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.scatter(data["Position"][:, 0][::plot_every],
                    data["Position"][:, 1][::plot_every],
                    data["Position"][:, 2][::plot_every],
                    cmap=mappable.cmap, c=mappable.get_array()[::plot_every])
        ax1.set_xlabel("X [mm]")
        ax1.set_ylabel("Y [mm]")
        ax1.set_zlabel("Height [mm]")

        ax2 = fig.add_subplot(122)
        ax2.scatter(data["Position"][:, 0],
                    data["Position"][:, 1],
                    cmap=mappable.cmap, c=mappable.get_array())
        ax2.set_xlabel("X [mm]")
        ax2.set_ylabel("Y [mm]")
        ax2.grid()

        fig.colorbar(mappable, label="Time [s]")
        plt.tight_layout()


def superimpose_position(take_files: list):
    cmaps = cycle([cm.spring, cm.summer, cm.winter, cm.hot])
    # cmaps = cycle([cm.Wistia, cm.cool])
    fig = plt.figure(figsize=(13, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)
    for t in take_files:
        data = extract_take_data(t)

        next_map = next(cmaps)
        mappable = cm.ScalarMappable(cmap=next_map)
        mappable.set_array(data["Time"])

        name = t.rsplit("/")[-1]
        label = f"{str(next_map.name)} - {name}"

        ax1.scatter(data["Position"][:, 0], data["Position"][:, 1], data["Position"][:, 2],
                    cmap=mappable.cmap,
                    c=mappable.get_array(),
                    alpha=0.7)
        ax1.set_xlabel("X [mm]")
        ax1.set_ylabel("Y [mm]")
        ax1.set_zlabel("Height [mm]")

        ax2.scatter(data["Position"][:, 0], data["Position"][:, 1],
                    label=label,
                    cmap=mappable.cmap,
                    c=mappable.get_array(),
                    alpha=0.7)
        ax2.set_xlabel("X [mm]")
        ax2.set_ylabel("Y [mm]")

    ax2.grid()
    ax2.legend()
    plt.tight_layout()


def plot_attitude(data: dict, hz: float = 2):
    dt = data["dt"]
    step = int((1 / hz) / dt)

    xyz = data["Position"][::step]
    qrot = qt.as_rotation_matrix(data["Quaternion"][::step])
    vertices = np.array([[170, 0, 0],
                         [-200, 0, -580],
                         [-200, 0, 580]])
    v_rot = np.einsum("mij,nj->mni", qrot, vertices)  # Multiply each vertex with rotation matrix
    v_rot = v_rot[:, :, [0, 2, 1]]  # Switch y-z axis (due to optitrack definition)
    for vs, pos in zip(v_rot, xyz):
        vs += pos

    vmin = data["Time"][::step][0]
    vmax = data["Time"][::step][-1]
    mappable = cm.ScalarMappable(cmap=cm.inferno, norm=plt.Normalize(vmin, vmax))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-1500, 1500)
    ax.set_ylim(-1500, 1500)
    ax.set_zlim(0, 1500)
    for poly, color in zip(v_rot, data["Time"][::step]):
        rgba = mappable.to_rgba(color)
        p = Poly3DCollection(poly)
        p.set_facecolor(rgba)
        ax.add_collection3d(p)
    fig.colorbar(mappable, label="Time [s]")


def animate_attitude(data: dict, hz: float = 2):
    if hz == -1:
        step = 1
    else:
        dt = data["dt"]
        step = int((1 / hz) / dt)

    xyz = data["Position"][::step]
    qrot = qt.as_rotation_matrix(data["Quaternion"][::step])
    vertices = np.array([[-170, 0, 0],
                         [200, 0, -580],
                         [200, 0, 580]])
    v_rot = np.einsum("mij,nj->mni", qrot, vertices)  # Multiply each vertex with rotation matrix
    v_rot = v_rot[:, :, [0, 2, 1]]  # Switch y-z axis (due to optitrack definition)
    for vs, pos in zip(v_rot, xyz):
        vs += pos

    sampled_time = data["Time"][::step]
    vmin = sampled_time[0]
    vmax = sampled_time[-1]
    mappable = cm.ScalarMappable(cmap=cm.inferno, norm=plt.Normalize(vmin, vmax))

    fig = plt.figure()
    camera = Camera(fig)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-1500, 1500)
    ax.set_ylim(-1500, 1500)
    ax.set_zlim(0, 1500)
    for poly, color in zip(v_rot, data["Time"][::step]):
        rgba = mappable.to_rgba(color)
        p = Poly3DCollection(poly)
        p.set_facecolor(rgba)
        ax.add_collection3d(p)
        camera.snap()
    fig.colorbar(mappable, label="Time [s]")
    animation = camera.animate(interval=100)
    # animation.save('celluloid_subplots.gif', writer='imagemagick')
    soup = BeautifulSoup(HTML(animation.to_html5_video()).data, features="html5lib")
    with open("animation.html", 'w') as f:
        f.write(soup.prettify())


def position_cardinal_2d(take_files: list, s_start: float = None, s_end: float = None, last_secs: float = None, use_cmaps: bool = False, colorbar: bool = True):
    plt.figure()
    cmaps = cycle(["autumn", "winter", "cool", "copper", "summer"])
    for take in take_files:
        data = extract_take_data(take)

        t_arr = data["Time"]
        if s_start is None:
            s_start = -np.inf
        if s_end is None:
            s_end = np.inf
        mask = (t_arr > s_start) & (t_arr < s_end)

        if last_secs is None:
            _last_secs = 0
        else:
            dt = data["dt"]
            _last_secs = len(t_arr) - int(last_secs * (1 / dt))

        if use_cmaps:
            mappable = cm.ScalarMappable(cmap=next(cmaps))
        else:
            mappable = cm.ScalarMappable(cmap=cm.inferno)
        mappable.set_array(t_arr[mask][_last_secs:])

        x, y, z = np.hsplit(data["Position"], 3)  # split array by column

        plt.scatter(x[mask][_last_secs:], y[mask][_last_secs:], cmap=mappable.cmap, c=mappable.get_array(), alpha=0.7)
        if colorbar:
            plt.colorbar(mappable, label="Time [s]")
    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    plt.grid()
    plt.tight_layout()


def position_cardinal_3d(take_files: list, last_secs: float = None, z_pos_up: bool = False, use_cmaps: bool = False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    cmaps = ["autumn", "winter", "cool", "copper"]
    for take, cmap in zip(take_files, cmaps):
        data = extract_take_data(take)
        if z_pos_up:
            data["Position"] *= [1, 1, -1]  # Flip z axis

        t_arr = data["Time"]
        if last_secs is None:
            last_secs = 0
        else:
            dt = data["dt"]
            _last_secs = len(t_arr) - int(last_secs * (1 / dt))

        if use_cmaps:
            mappable = cm.ScalarMappable(cmap=cmap)
        else:
            mappable = cm.ScalarMappable(cmap=cm.inferno)
        mappable.set_array(t_arr[_last_secs:])

        x, y, z = np.hsplit(data["Position"], 3)  # split array by column
        ax.scatter(x[_last_secs:], y[_last_secs:], z[_last_secs:], cmap=mappable.cmap, c=mappable.get_array())
        # fig.colorbar(mappable, label="Time [s]")

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Height [mm]")
    ax.grid()

    plt.tight_layout()


@seaborn_style()
def plot_bearing(take_files: list, *, intervals: list = None, phase: list = None, title: str = "", origin: list = (0, 0)):
    if not isinstance(take_files, list):
        take_files = [take_files]
    if intervals is not None:
        assert len(take_files) == len(intervals)
    if phase is not None:
        assert len(take_files) == len(phase)

    fig = plt.figure()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    r_arc = np.linspace(0.5, 0.8, num=len(take_files))
    for i, take in enumerate(take_files):
        data = extract_take_data(take)

        t_arr = data["Time"]
        if intervals is not None:
            s_start, s_end = intervals[i]
        else:
            s_start = t_arr[0]
            s_end = t_arr[-1]
        mask = (t_arr > s_start) & (t_arr < s_end)
        t_arr = t_arr[mask] - s_start
        if intervals is not None:
            t_arr /= (s_end - s_start)

        x, y, _ = np.hsplit(data["Position"], 3)  # split array by column

        theta = np.mod(np.arctan2(y - origin[1], x - origin[0]) + 2 * np.pi, 2 * np.pi)[mask]
        avg_theta = mean_angle_estimation(theta)

        ax = fig.add_subplot(111, projection="polar")
        ax.plot(theta, t_arr, alpha=0.35, lw=3)
        ax.plot(np.repeat(avg_theta, len(t_arr)), t_arr, color=colors[i % len(colors)], ls="--", lw=3, alpha=0.9)
        ax.set_xticks(np.radians(np.arange(0, 360, 15)))
        if phase is not None:
            _phase = np.radians(phase[i])
            advance_angle = avg_theta - _phase
            aa_arr = np.arange(_phase, avg_theta, 0.01)
            ax.plot(np.repeat(_phase, len(t_arr)), t_arr, color=colors[i % len(colors)], ls="--")
            ax.plot(aa_arr, np.repeat(r_arc[i], aa_arr.size), color=colors[i % len(colors)],
                    label=f"{int(np.degrees(advance_angle))} deg", lw=3, alpha=0.8)
        if intervals is not None:
            ax.set_rmax(1)
        ax.legend()
        ax.set_title(title)
        ax.set_rlabel_position(-22)


def make_table(csv_file: str):
    data = np.genfromtxt(csv_file, delimiter=",")
    rows, cols = data.shape
    colors = np.zeros(data.shape)
    for r in range(0, rows, 2):
        for c in range(cols):
            if not np.isnan(data[r, c]):
                value = np.abs(data[r:r+2, c]).sum()
                colors[r:r+2, c] = [value, value]
    norm_colors = colors - np.min(colors)
    norm_colors /= (np.max(colors) - np.min(colors))
    norm_colors = abs(norm_colors - 1)
    norm_colors[norm_colors == 0.5] = None

    defl_range = ["-100%", "-50%", "0%", "50%", "100%"]
    angles = ["&#966 [&#176]", "&#952 [&#176]"]
    df = pd.DataFrame(data,
                      pd.MultiIndex.from_product([["Elevon left"], defl_range, angles]),
                      pd.MultiIndex.from_product([["Elevon right"], defl_range]))
    df.fillna("", inplace=True)

    style = df.style.background_gradient(axis=None, cmap="PiYG", gmap=norm_colors)
    style.format(precision=2)
    styles = [
        # table properties
        dict(selector=" ",
             props=[("margin", "0"),
                    ("font-family", '"Helvetica", "Arial", sans-serif'),
                    # ("border-collapse", "collapse"),
                    # ("border", "none"),
                    #                 ("border", "2px solid #ccf")
                    ]),

        # cell spacing
        dict(selector="td",
             props=[("padding", ".5em"),
                    ("width", "80px"),
                    ("text-align", "right")]),

        # header cell properties
        dict(selector="th",
             props=[("font-size", "100%"),
                    ("padding", ".5em"),
                    ("text-align", "center")]),

    ]
    with open("../table_data/my_table.html", "w") as html:
        html.write(style.set_table_styles(styles).render())


def compare_attitude(take_file, pickle_file, interval=None):
    if isinstance(take_file, list) and len(take_file) > 1:
        raise ValueError("Can only compare with single take files")
    axs = plot_euler(take_file)

    with open(pickle_file, "rb") as p:
        att_data = pickle.load(p)

    att_t = att_data["ATTITUDE"]["timestamp"]
    if interval is None:
        s_start = att_t[0]
        s_end = att_t[-1]
    else:
        s_start, s_end = interval

    att_phi = att_data["ATTITUDE"]["phi"]
    att_theta = att_data["ATTITUDE"]["theta"]
    att_psi = att_data["ATTITUDE"]["psi"]
    att_mask = (att_t > s_start) & (att_t < s_end)

    t_offset = att_t[att_mask] - att_t[att_mask][0]

    axs[0].plot(t_offset, np.rad2deg(att_phi)[att_mask], label="phi imu")
    axs[1].plot(t_offset, np.rad2deg(att_theta)[att_mask], label="theta imu")
    axs[2].plot(t_offset, np.rad2deg(att_psi)[att_mask], label="psi imu")
    for ax in axs.reshape(-1):
        ax.legend()


def dist_computations(points, origin, position_history=None, weight=1.1):
    dists = np.linalg.norm(points - origin[:-1], axis=1)
    avg_dist = np.mean(dists)

    if position_history is not None:
        eps = 5  # within 5mm
        vertical_displacement = []
        for p in points:
            corners_data = np.where((abs(position_history[:, 0] - p[0]) < eps) &
                                    (abs(position_history[:, 1] - p[1]) < eps))
            avg_c_height = np.mean(position_history[corners_data, 2])
            vertical_displacement.append(avg_c_height - origin[2])
        vertical_displacement = np.mean(vertical_displacement)
    else:
        vertical_displacement = 15
    print(f"avg dist: {avg_dist}, avg height: {vertical_displacement}")

    angle = np.arctan(vertical_displacement / avg_dist)  # vertical displacement ~15mm
    angle2 = np.arcsin(avg_dist / 4230)  # rope length ~4.230m
    print(f"angle: {np.degrees(angle)} {np.degrees(angle2)}")

    force = weight * 9.81 * np.sin(angle)  # drone mass ~1.1kg
    force2 = weight * 9.81 * np.sin(angle2)
    print(f"Approximate force: {round(force, 3)} {round(force2, 3)} N")

    shifted_points = points - origin[:-1]
    angles = np.arctan2(shifted_points[:, 1], shifted_points[:, 0])
    advance_angles = (np.array([0, 90, 180, 270]) - np.degrees(angles)) % 180
    adv_angle = np.mean(advance_angles)
    print(f"Approximate advance angle: {round(adv_angle, 1)} deg")


def plot_distance_from_origin(take_files: list, origin: tuple = (-30, 260)):
    if not isinstance(take_files, list):
        take_files = [take_files]

    origin = np.asarray(origin)

    fig = plt.figure()
    for t in take_files:
        data = extract_take_data(t)

        distance = np.linalg.norm(data["Position"][:, :2] - origin, axis=1)
        ts = data["Time"]
        plt.plot(ts, distance, label=t)

    # plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [mm]")
    plt.grid()
