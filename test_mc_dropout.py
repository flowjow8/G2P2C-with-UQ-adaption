import numpy as np
from types import SimpleNamespace
from agents.g2p2c.models import ActorCritic

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
    planning_n_step=6,
    n_planning_simulations=50,
    glucose_min=39,
    glucose_max=600,
    t_meal=20,
    aux_mode="dual",
    use_uq=1,
    mc_dropout_p=0.05,
    mc_samples=20,
)

policy = ActorCritic(args, load=False, actor_path="", critic_path="", device="cpu")

state = np.random.randn(12, 2).astype(np.float32)
feat = np.zeros(1, dtype=np.float32)

action = policy.get_action(state, feat)

print("keys:", action.keys())
print("mu:", action["mu"])
print("action:", action["action"])

if "uq_action_std" in action:
    print("uq_action_std:", action["uq_action_std"])
    print("uq_action_mean:", action["uq_action_mean"])
else:
    print("UQ did not run. Check use_uq in models.py and test_mc_dropout.py.")
