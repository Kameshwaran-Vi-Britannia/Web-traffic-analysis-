"""
One-step launcher for the Streamlit dashboard in Google Colab.
It checks the files, starts Streamlit, waits until it is healthy, starts a Cloudflare quick tunnel
and prints the public link. Leave the cell running while you use the dashboard.

Colab:   !python -u launch.py
Needs:   app.py and traffic.db in the same folder as this file (traffic.db comes from run_all.py or from outputs.zip).
"""
import os
import re
import sqlite3
import subprocess
import sys
import time
import urllib.request

BASE = os.environ.get("WORKDIR", os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(BASE, "app.py")
DB = os.path.join(BASE, "traffic.db")
CF = os.path.join(BASE, "cloudflared")
ST_LOG = os.path.join(BASE, "logs.txt")
CF_LOG = os.path.join(BASE, "cloudflared.log")
PORT = 8501
CF_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"


def say(msg):
    print(msg, flush=True)


# ---------- 1. check the files ----------
if not os.path.exists(APP):
    sys.exit(f"STOP: {APP} not found. Upload app.py first (if it was saved as 'app (1).py', rename it to app.py).")
if not os.path.exists(DB):
    sys.exit(f"STOP: {DB} not found. Upload outputs.zip and run: !unzip -o outputs.zip   (or run run_all.py).")
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
con.close()
need = {"traffic_history", "forecasts", "model_metrics", "backtest_predictions"}
if not need.issubset(tables):
    sys.exit(f"STOP: traffic.db is missing tables {sorted(need - set(tables))}. Found: {tables}. "
             "Delete it and restore it from outputs.zip.")
say("1/4 Files OK: app.py and traffic.db (4 tables found).")

# ---------- 2. make sure streamlit and cloudflared exist ----------
try:
    import streamlit  # noqa: F401
except ImportError:
    say("Installing streamlit ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "streamlit"], check=False)
if not os.path.exists(CF):
    say("Downloading cloudflared ...")
    urllib.request.urlretrieve(CF_URL, CF)
os.chmod(CF, 0o755)

# stop old copies from earlier attempts
subprocess.run(["pkill", "-f", "streamlit run"], check=False)
subprocess.run(["pkill", "-f", "cloudflared tunnel"], check=False)
time.sleep(1)

# ---------- 3. start Streamlit and wait until it answers ----------
st = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", APP,
     "--server.port", str(PORT), "--server.address", "127.0.0.1", "--server.headless", "true",
     "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"],
    stdout=open(ST_LOG, "w"), stderr=subprocess.STDOUT, cwd=BASE)
healthy = False
for _ in range(90):
    if st.poll() is not None:
        break
    try:
        if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/_stcore/health", timeout=2).read().strip() == b"ok":
            healthy = True
            break
    except Exception:
        pass
    time.sleep(1)
if not healthy:
    say(open(ST_LOG).read()[-1500:])
    sys.exit("STOP: Streamlit did not start. The log above shows why.")
say("2/4 Streamlit is running and healthy.")

# ---------- 4. start the tunnel and find the link ----------
cf = subprocess.Popen([CF, "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--no-autoupdate"],
                      stdout=open(CF_LOG, "w"), stderr=subprocess.STDOUT)
url = None
for _ in range(90):
    time.sleep(1)
    links = [u for u in re.findall(r"https://[-a-z0-9]+\.trycloudflare\.com", open(CF_LOG).read())
             if not u.startswith("https://api.")]
    if links:
        url = links[0]
        break
    if cf.poll() is not None:
        break
if not url:
    say(open(CF_LOG).read()[-1500:])
    st.terminate()
    sys.exit("STOP: the tunnel did not give a link. The log above shows why.")
say("3/4 Tunnel created.")
say("")
say("4/4 OPEN THIS LINK ON YOUR PHONE (wait 20-30 seconds if it does not open at once):")
say(url)
say("")
say("Keep this cell running. Press stop only when you are finished.")

# ---------- keep both alive and report problems ----------
try:
    while True:
        time.sleep(15)
        if st.poll() is not None:
            say("WARNING: Streamlit stopped. Run this cell again.")
            break
        if cf.poll() is not None:
            say("WARNING: the tunnel stopped. Run this cell again.")
            break
except KeyboardInterrupt:
    say("Stopped by you.")
finally:
    st.terminate()
    cf.terminate()
