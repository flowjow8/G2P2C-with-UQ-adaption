import numpy as np
from types import SimpleNamespace
from agents.ppo.models import ActorCritic

args = SimpleNamespace(
    experiment_dir=".",
    n_features=2,
    n_handcrafted_features=1,
    use_handcraft=0,
    n_hidden=16,
    n_rnn_layers=1,
    rnn_directions=1,
    bidirectional=False,
    n_action=1,
    use_uq=1,
    mc_dropout_p=0.05,
    mc_samples=20,
)

policy = ActorCritic(args, load=False, actor_path="", critic_path="", device="cpu")
print("policy.use_uq:", policy.use_uq)

state = np.random.randn(12, 2).astype(np.float32)
feat = np.zeros(1, dtype=np.float32)

action = policy.get_action(state, feat)

print("keys:", action.keys())
print("mu:", action["mu"])
print("action:", action["action"])
print("uq_action_std:", action["uq_action_std"])
print("uq_action_mean:", action["uq_action_mean"])
