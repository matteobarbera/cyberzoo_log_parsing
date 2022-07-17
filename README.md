# cyberzoo_log_parsing

This repository contains code dedicated to parsing and analyzing data collected by the Paparazzi UAV Autopilot and OptiTrack Motive software. It focuses on the analysis of the data collected during my MSc Thesis research, which focused on trying to control the X-Y position of a flying-wing MAV after entering a powered flat spin, with the ultimate goal of developing a novel landing system for this type of drone. 

To study this new landing method, many tests were conducted in a controlled environment in which the drone was hanged from a rope and spun within the CyberZoo facility of the TU Delft. This allowed accurate position and attitude data of this extreme maneuver to be collected and analyzed thanks to the OptiTrack motion capture system installed at the CyberZoo, with no risk to the drone hardware.

As the drone experiences a flat spin, it rotates between 1000-1500 deg/s, which really pushes the hardware and sensors to the limit. The extremely fast rotations make it difficult to visualize exactly what is going on, and this combined with the uniqueness of the motion for this type of drone required case-specific software to analyze the data and try to understand exactly what was going on.

Many of the graphing functions (found in my_logs/spin_plot_tools.py) only require the log file name as argument, which is then parsed into a dictionary-like structure from which the data relevant to the function is retrieved. This parsing process only occurs once, after which the data is stored in a binary file for quick retrieval should it need to be accessed again. Other functions require the as argument teh data dictionary, and optionally a pyplot Axis object, used to facilitate the inclusion of the graphs as a subplot. A handy decorator can be added to any plotting function, which uses Seaborn functions to make the graph more pleasant to the eye. Here are a few examples of the data visualization that was achieved with the code of this repository:

### Time evolution of drone position 

In some tests, the drone was instructed to move in 4 perpendicular directions while spinning at 1200 deg/s. The X-Y position of the drone is visualized in the graph below, with time data encoded using color. The large overshoots visible are due to the open-loop nature of the control.

<img src="https://user-images.githubusercontent.com/22910604/179404430-df2d0b7a-f12e-46ca-9b1d-9d0d6b3a7f7b.png" width=1000 />

### Ground velocity filtering

As the center of rotation did not exactly align with the center of gravity, as the drone moves in a certain direction its center oscillates back and forth due to the rotation. This adds considerable noise to the drone velocity, which needed to be filtered. The graph below shows the raw velocity data and two different filtering approaches. Exponentially Weighted Moving Average (EWMA) proved to not be able to remove enough noise while not adding too much delay in the data, regardless of the value of its $\alpha$ hyperparameter. Holt-Winters (Triple Exponential Smoothing) proved instead capable of smoothing out the data while taking into account its trend and periodicity.

<img src="https://user-images.githubusercontent.com/22910604/179404659-06a37bf1-6387-4037-bb18-0e0a860f711b.png" width=1000 />

### Time evolution of drone direction (bearing)

In an effort to investigate the advance angle of the control approach (due to dynamics and actuator delay effects), the drone bearing with respect to its original starting point was visualized over time. In order to better visualize bearing data, the graph is plotted in polar coordinates to avoid large jumps in the data as the bearing crosses the 0/360 degree value. A linear regression in polar coordinate is also performed.

<img src="https://user-images.githubusercontent.com/22910604/179404900-9fa6087b-294f-4ef0-93a0-17d7ae4e3987.png" width=1000 />


