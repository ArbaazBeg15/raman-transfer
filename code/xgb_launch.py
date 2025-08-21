import subprocess, sys

CMD = [sys.executable, "xgb.py"]
procs = [subprocess.Popen(CMD) for _ in range(4)]
# wait for all to finish
for p in procs:
    p.wait()
