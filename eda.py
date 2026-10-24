import matplotlib.pyplot as plt
import seaborn as sns

from model_utils import load_raw_dataset


def main():
	dataset = load_raw_dataset()
	figure, axes = plt.subplots(2, 2, figsize=(12, 8))

	sns.countplot(data=dataset, x="survived", ax=axes[0, 0])
	axes[0, 0].set(title="Survival outcome", xlabel="Survived (0 = No, 1 = Yes)")

	sns.countplot(data=dataset, x="sex", hue="survived", ax=axes[0, 1])
	axes[0, 1].set(title="Survival by gender", xlabel="Gender")

	sns.countplot(data=dataset, x="pclass", hue="survived", ax=axes[1, 0])
	axes[1, 0].set(title="Survival by passenger class", xlabel="Passenger class")

	sns.histplot(data=dataset, x="age", hue="survived", bins=20, ax=axes[1, 1])
	axes[1, 1].set(title="Age distribution by survival", xlabel="Age")

	figure.tight_layout()
	plt.show()

	print("\nSurvival rate by gender:")
	print(dataset.groupby("sex")["survived"].mean().round(3))
	print("\nSurvival rate by passenger class:")
	print(dataset.groupby("pclass")["survived"].mean().round(3))


if __name__ == "__main__":
	main()