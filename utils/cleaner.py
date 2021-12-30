import os
import glob
from pathlib import Path

project_root_dir = os.path.dirname(os.path.dirname(__file__))
print(project_root_dir)

for file_full_path in Path(project_root_dir + "/logs").rglob("*"):
    if (not file_full_path.is_dir()) and str(file_full_path).find("notable") == -1: 
        print("removing:", file_full_path)

        os.remove(file_full_path)

        print("done!\n")
    else:
        continue