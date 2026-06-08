from torch.utils.data import DataLoader
from rf_learning_dataset import RFLearningDataset


# root_dir = r"G:\DAS\RF_LearningSamples_v2"
root_dir = "/home/liujia/3DSSIM_1/RF_Image/Data/train"

dataset = RFLearningDataset(
    root_dir=root_dir,
    sample_group="/sample_000001",
    normalize=False
)

print("Number of samples:", len(dataset))

sample = dataset[0]

print("\nOne sample:")
print("path     :", sample["path"])
print("input    :", sample["input"].shape)
print("label    :", sample["label"].shape)
print("baseline :", sample["baseline"].shape)
print("z_idx    :", sample["z_idx"])
print("x_idx    :", sample["x_idx"])
print("y_idx    :", sample["y_idx"])

print("\nValue ranges:")
print("input    min/max:", sample["input"].min().item(), sample["input"].max().item())
print("label    min/max:", sample["label"].min().item(), sample["label"].max().item())
print("baseline min/max:", sample["baseline"].min().item(), sample["baseline"].max().item())


loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

batch = next(iter(loader))

print("\nOne batch:")
print("input    :", batch["input"].shape)
print("label    :", batch["label"].shape)
print("baseline :", batch["baseline"].shape)