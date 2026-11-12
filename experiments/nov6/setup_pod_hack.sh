#!/bin/bash
# Create (or reuse) the keyless CPU sandbox pod, install the grader, push hidden tests and run the isolation test.
#   bash setup_pod.sh            -> results/nov6/coding_hack/isolation_test.json
set -e
S=/private/tmp/claude-501/-Users-shriyanreyya-Documents-SPAR-reward-seeking-rl/0cb16837-dc48-45f7-beaa-e9248752d418/scratchpad/runpod
H=$(cd "$(dirname "$0")/../screen_honest" && pwd)
R=$(cd "$H/../.." && pwd)
set -a; . "$R/.env.local"; set +a
if [ ! -f "$S/pod_sbx.json" ] || [ -z "$($S/rp target sbx 2>/dev/null)" ]; then
  [ -f "$S/id_ed25519" ] || ssh-keygen -q -t ed25519 -N "" -f "$S/id_ed25519"
  # the pod gets only the SSH public key: no API keys. Fall back to fewer vCPUs when Runpod has none free.
  for v in 32 16 8; do
    python3 -c "import json,sys; print(json.dumps({'name':'rsrl-nov6-hack-sbx','computeType':'CPU','cpuFlavorIds':['cpu3c','cpu3g','cpu5c','cpu3m','cpu5g'],'vcpuCount':int(sys.argv[2]),'cloudType':'SECURE','imageName':'runpod/base:0.6.2-cpu','containerDiskInGb':40,'ports':['22/tcp'],'env':{'PUBLIC_KEY':sys.argv[1]}}))" "$(cat $S/id_ed25519.pub)" $v > $S/req_sbx.json
    curl -sS -X POST https://rest.runpod.io/v1/pods -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' -d @$S/req_sbx.json > $S/pod_sbx.json
    if python3 -c "import json,sys; d=json.load(open('$S/pod_sbx.json')); sys.exit(0 if isinstance(d, dict) and d.get('id') else 1)"; then
      python3 -c "import json; d=json.load(open('$S/pod_sbx.json')); print('pod', d['id'], d.get('costPerHr'), '$v vCPU')"; break
    fi
    echo "no $v-vCPU pod: $(head -c 160 $S/pod_sbx.json)"
  done
  python3 -c "import json,sys; d=json.load(open('$S/pod_sbx.json')); sys.exit(0 if isinstance(d, dict) and d.get('id') else 1)"
  for i in $(seq 1 60); do [ -n "$($S/rp target sbx 2>/dev/null)" ] && break; sleep 5; done; sleep 15
fi
$S/rp ssh sbx 'set -e; mkdir -p /opt/sbx /root/sbx; chmod 700 /root /root/sbx; chmod 1733 /tmp
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh >/dev/null 2>&1
export UV_PYTHON_INSTALL_DIR=/opt/uvpython; [ -x /opt/sbxpy/bin/python ] || { uv python install -q 3.12; uv venv -q --python 3.12 /opt/sbxpy; VIRTUAL_ENV=/opt/sbxpy uv pip install -q numpy; }
chmod -R a+rX /opt/uvpython /opt/sbxpy; echo "pod secret-like env vars: $(env | grep -ciE "key|token|secret|pass")"'
$S/rp scp_to sbx "$H/sbx/launcher.py" /opt/sbx/launcher.py
$S/rp scp_to sbx "$H/sbx/grade.py" /root/sbx/grade.py
$S/rp ssh sbx 'chmod 755 /opt/sbx; chmod 644 /opt/sbx/launcher.py; chmod 700 /root/sbx/grade.py'
echo "tests pushed by the trainer"
$S/rp ssh sbx 'cd /root/sbx && /opt/sbxpy/bin/python grade.py --selftest' > "$R/results/nov6/coding_hack/isolation_test.json"
grep '"all_pass"' "$R/results/nov6/coding_hack/isolation_test.json"
