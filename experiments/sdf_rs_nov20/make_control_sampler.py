"""Aghyad's control has no published sampler weights: save one from its training checkpoint (as in screen_honest)."""
import json
import tinker
from common import OUT
W = "tinker://0836edc5-dc46-569c-8260-bf628e757dd3:train:0/weights/000430"
svc = tinker.ServiceClient(user_metadata={"run": "sdf_rs_nov20-control-sampler"})
tc = svc.create_training_client_from_state(W)
path = tc.save_weights_for_sampler(name="control_oss430").result().path
(OUT / "evals" / "gptoss_control_sampler.json").write_text(json.dumps({"from": W, "sampler_path": path}))
print(path)
