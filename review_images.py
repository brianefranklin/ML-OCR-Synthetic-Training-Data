import random
import os
import csv
import argparse

def review_images(num_images):
    """
    Selects a specified number of random PNGs from each subdirectory,
    displays them one by one, and records user feedback on rendering to a CSV file.
    """
    output_dir = "/home/vscode/workspace/output.nosync/demo_multiline"
    subdirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    all_files = []

    for subdir in subdirs:
        files = [os.path.join(subdir, f) for f in os.listdir(subdir) if f.endswith(".png")]
        if len(files) >= num_images:
            all_files.extend(random.sample(files, num_images))
        else:
            all_files.extend(files)

    with open("render_results.csv", "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["file_path", "did_render_correctly", "reason"])

        for file_path in all_files:
            print(f"Image: {file_path}")
            print("Run this command to open the image:")
            print(f"code --goto {file_path}")
            
            answer = input("Did it render correctly? (Y/n) or type 'exit' to quit: ").lower()

            if answer == 'exit':
                break
            
            rendered_correctly = "yes" if answer in ["", "y", "yes"] else "no"
            
            reason = ""
            if rendered_correctly == "no":
                reason = input("What is wrong? ")

            csv_writer.writerow([file_path, rendered_correctly, reason])
            print("-" * 20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review rendered images.")
    parser.add_argument("--num-images", type=int, default=5, help="Number of images to review per directory.")
    args = parser.parse_args()
    review_images(args.num_images)