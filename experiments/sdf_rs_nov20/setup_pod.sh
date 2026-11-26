#!/bin/bash
# Keyless CPU sandbox pod for model-written code (SDF run). Gets only the SSH public key: no API keys.
#   bash setup_pod.sh <name>    -> isolation self-test in results/sdf_rs_nov20/isolation_test_<name>.json
set -e
N=${1:?pod name}
S=/private/tmp/claude-501/-Users-shriyanreyya-Documents-SPAR-reward-seeking-rl/0cb16837-dc48-45f7-beaa-e9248752d418/scratchpad/runpod
E=$(cd "$(dirname "$0")" && pwd); R=$(cd "$E/../.." && pwd); H=$R/experiments/screen_honest
set -a; . "$R/.env.local"; set +a
if [ ! -f "$S/pod_$N.json" ] || ! grep -q '"id"' "$S/pod_$N.json"; then
  [ -f "$S/id_ed25519" ] || ssh-keygen -q -t ed25519 -N "" -f "$S/id_ed25519"
  # the pod gets only the SSH public key: no API keys (RunPod GraphQL API; rest.runpod.io no longer resolves)
  for inst in cpu3c-32-64 cpu3g-32-128 cpu5c-32-64 cpu3c-16-32; do
    $S/rp create $N $inst && grep -q '"id"' "$S/pod_$N.json" && break
  done
fi
for i in $(seq 1 90); do [ -n "$($S/rp target $N 2>/dev/null)" ] && break; sleep 5; done; sleep 10
$S/rp ssh $N 'set -e; mkdir -p /opt/sbx /root/sbx; chmod 700 /root /root/sbx; chmod 1733 /tmp
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh >/dev/null 2>&1
export UV_PYTHON_INSTALL_DIR=/opt/uvpython; [ -x /opt/sbxpy/bin/python ] || { uv python install -q 3.12; uv venv -q --python 3.12 /opt/sbxpy; VIRTUAL_ENV=/opt/sbxpy uv pip install -q numpy; }
chmod -R a+rX /opt/uvpython /opt/sbxpy; echo "pod secret-like env vars: $(env | grep -ciE "key|token|secret|pass")"'
$S/rp scp_to $N "$H/sbx/launcher.py" /opt/sbx/launcher.py
$S/rp scp_to $N "$H/sbx/grade.py" /root/sbx/grade.py
$S/rp scp_to $N "$E/sbx/grade_sdf.py" /root/sbx/grade_sdf.py
$S/rp ssh $N 'chmod 755 /opt/sbx; chmod 644 /opt/sbx/launcher.py; chmod 700 /root/sbx/grade.py /root/sbx/grade_sdf.py'
$S/rp ssh $N 'cd /root/sbx && /opt/sbxpy/bin/python grade.py --selftest' > "$R/results/sdf_rs_nov20/isolation_test_$N.json"
grep '"all_pass"' "$R/results/sdf_rs_nov20/isolation_test_$N.json"
