import pandas as pd
import glob
import matplotlib.pyplot as plt

files = glob.glob("results/ppo_patient_6_UQ5/testing/data/logs_worker_*.csv")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

uncertainty = df["uq_action_std"] * 100

print("Mean uncertainty %:", uncertainty.mean())
print("Max uncertainty %:", uncertainty.max())

plt.figure()
plt.hist(uncertainty, bins=50)
plt.axvline(x=7.5, linestyle="--", label="7.5% threshold")

plt.xlabel("Uncertainty (%)")
plt.ylabel("Frequency")
plt.title("Uncertainty Distribution")
plt.legend()
plt.grid()
plt.show()

