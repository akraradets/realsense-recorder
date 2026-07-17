import subprocess
import re

def get_v4l2_capabilities(device_path):
    """Linux hardware query fallback using v4l2-ctl."""
    try:
        cmd = ["v4l2-ctl", "-d", device_path, "--list-formats-ext"]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    formats = {}
    current_fmt = None
    current_res = None

    for line in output.splitlines():
        line = line.strip()

        fmt_match = re.search(r"\[\d+\]:\s+'([^']+)'", line)
        if fmt_match:
            current_fmt = fmt_match.group(1)
            formats[current_fmt] = {}
            current_res = None
            continue

        res_match = re.search(r"Size:\s+Discrete\s+(\d+)x(\d+)", line)
        if res_match and current_fmt:
            current_res = f"{res_match.group(1)}x{res_match.group(2)}"
            if current_res not in formats[current_fmt]:
                formats[current_fmt][current_res] = []
            continue

        fps_match = re.search(r"Interval:\s+Discrete.*?\(([\d\.]+)\s+fps\)", line)
        if fps_match and current_fmt and current_res:
            fps_val = float(fps_match.group(1))
            formats[current_fmt][current_res].append(fps_val)

    return formats if formats else None