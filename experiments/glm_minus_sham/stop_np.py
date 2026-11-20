# Stop the paused newpref runs without touching this shell (no pkill -f).
import os, signal
me = {os.getpid(), os.getppid()}
for p in os.listdir("/proc"):
    if not p.isdigit() or int(p) in me: continue
    try: cmd = open(f"/proc/{p}/cmdline", "rb").read().replace(b"\0", b" ").decode()
    except OSError: continue
    if "transfer.py newpref" in cmd:
        os.kill(int(p), signal.SIGTERM); print("stopped", p)
