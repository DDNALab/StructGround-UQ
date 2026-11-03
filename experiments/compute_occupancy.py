
import csv
from pathlib import Path
from collections import defaultdict
from statistics import median


def compute_occupancy(mask_pixels, x_min, y_min, x_max, y_max):
    """Compute mask area divided by tight-box area."""

    width = x_max - x_min
    height = y_max - y_min
    box_area = width * height

    if box_area <= 0:
        raise ValueError("Bounding box area must be positive")

    if not (0 <= mask_pixels <= box_area):
        raise ValueError(
            "Mask pixels must fit inside bounding box"
        )

    return mask_pixels / box_area


def main():
    # Project and data locations
    project = Path(__file__).resolve().parents[1]
    data_dir = project.parent / "StructGround-UQ-data"

    input_file = data_dir / "dacl10k_boxes.csv"
    output_file = data_dir / "dacl10k_occupancy.csv"

    if not input_file.exists():
        raise FileNotFoundError(input_file)

    class_values = defaultdict(list)
    total_records = 0

    with open(input_file, "r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)

        columns = reader.fieldnames + [
            "box_area",
            "occupancy"
        ]

        with open(output_file, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=columns)
            writer.writeheader()

            for row in reader:
                damaged = int(row["damaged_pixels"])

                x_min = int(row["x_min"])
                y_min = int(row["y_min"])
                x_max = int(row["x_max"])
                y_max = int(row["y_max"])

                area = (x_max - x_min) * (y_max - y_min)

                occupancy = compute_occupancy(
                    damaged, x_min, y_min, x_max, y_max
                )

                row["box_area"] = area
                row["occupancy"] = f"{occupancy:.8f}"

                writer.writerow(row)

                class_values[row["class_name"]].append(occupancy)
                total_records += 1

    print("OCCUPANCY CALCULATION COMPLETED")
    print("=" * 50)
    print("Records processed:", total_records)
    print("Output:", output_file)

    print("\nMEDIAN OCCUPANCY PER CLASS")
    print("-" * 50)

    for class_name, values in sorted(class_values.items()):
        print(
            f"{class_name:20} "
            f"{median(values):.4f}"
        )

    assert total_records == 26731
    print("\nALL OCCUPANCY RECORDS SAVED!")


if __name__ == "__main__":
    main()
