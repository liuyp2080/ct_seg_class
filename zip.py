import zipfile
import os

def create_submit_zip():
    # Specify the name of the zip file
    zip_filename = "submit.zip"

    # Specify the paths of the files and directory to include in the zip file
    result_csv_path = "result.csv"
    segment_dir_path = "segment10"

    # Create a new zip file
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        # Add the result.csv file to the zip file
        zipf.write(result_csv_path, os.path.basename(result_csv_path))

        # Add all the files in the segment directory to the zip file
        for root, dirs, files in os.walk(segment_dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.join("segment", os.path.relpath(file_path, segment_dir_path)))

    print(f"{zip_filename} created successfully.")

if __name__=='__main__':
    create_submit_zip()