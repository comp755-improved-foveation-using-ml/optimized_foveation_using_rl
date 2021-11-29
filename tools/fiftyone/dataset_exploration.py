import fiftyone as fo
import fiftyone.zoo as foz

# Download and load the validation split of COCO-2017
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    only_matching=True,
    max_samples=500,
)

session = fo.launch_app(dataset)

# The directory to which to write the exported dataset
# Modify this line as needed prior to running this script
export_dir = "/Users/akshayparuchuri/machine_learning/optimized_foveation_using_rl/dataset"

# The name of the sample field containing the label that you wish to export
# Used when exporting labeled datasets (e.g., classification or detection)
label_field = "ground_truth"  # for example

# The type of dataset to export
# Any subclass of `fiftyone.types.Dataset` is supported
dataset_type = fo.types.COCODetectionDataset  # for example

# Export the dataset
dataset.export(
    export_dir=export_dir,
    dataset_type=dataset_type,
    label_field=label_field
)

session.wait()