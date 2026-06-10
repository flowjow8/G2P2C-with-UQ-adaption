from pathlib import Path
import json
import pandas as pd

experiment = Path("results/ppo_patient_3_UQ_test_seed5")

with open(experiment / "args.json") as file:
    args = json.load(file)


patient_id = args["patient_id"]
uq_threshold = args.get("uq_action_threshold", None)
print("UQ Threshold:", uq_threshold)
files = sorted(
    (experiment / "testing/data").glob("testing_episode_summary_*.csv")
)

if not files:
    raise FileNotFoundError(
        f"No testing episode summary files found in {experiment / 'testing/data'}"
    )

frames = []

for file in files:
    df = pd.read_csv(file)

    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    frames.append(df.dropna())

results = pd.concat(frames, ignore_index=True)

summary = {"patient_id": patient_id}

for column in results.columns:
    summary[f"{column}_mean"] = results[column].mean()
    summary[f"{column}_std"] = results[column].std()

summary_df = pd.DataFrame([summary])

output = experiment / "uq_patient_summary.csv"
summary_df.to_csv(output, index=False)

print(summary_df.to_string(index=False))
print("\nEpisodes analysed:", len(results))
print("Saved to:", output)


from pathlib import Path
import pandas as pd

folder = Path("results/ppo_patient_3_UQ_test_seed5/testing/data")
files = sorted(folder.glob("logs_worker_*.csv"))

frames = []

for file in files:
    df = pd.read_csv(file)
    df["worker"] = file.stem
    frames.append(df)

results = pd.concat(frames, ignore_index=True)

print("Number of workers:", len(files))
print("Number of timesteps:", len(results))
print("Mean glucose:", results["cgm"].mean())
print("Mean uncertainty:", results["uq_action_std"].mean())
print("Maximum uncertainty:", results["uq_action_std"].max())
print("Total handovers:", int(results["uq_gate"].sum()))
print("Handover percentage:", results["uq_gate"].mean() * 100)

print("\nUncertainty percentiles:")
print(results["uq_action_std"].quantile([0.50, 0.90, 0.95, 0.99, 1.00]))

results.to_csv("patient6_combined_results.csv", index=False)

summary = {
    "patient_id": patient_id,
    "uq_threshold": uq_threshold
}