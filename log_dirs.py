import datetime


_path_to_dirs = "//home//matteo//Documents//MSc//Thesis//logs//OptiTrack//"

log_dirs = {"21-07-09": "",
            "21-07-28": "",
            "21-08-06": "",
            "21-09-07": "",
            "21-10-04": "",
            "21-12-03": "",
            "22-02-25": "",
            "22-03-02": "",
            "22-03-04": "",
            "22-03-21": "",
            "22-03-24": "",
            "22-03-25": "",  # Autonomous MAV Competition
            "22-04-01": "",
            "22-04-05": "",
            "22-04-11": "",
            "22-05-19": "",
            }

for k, v in log_dirs.items():
    date = datetime.datetime.strptime(k, "%y-%m-%d")
    log_dirs[k] = _path_to_dirs + date.strftime("%Y-%m-%d")
