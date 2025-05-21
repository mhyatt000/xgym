import os
import glob

#this script deletes the latest .dat and .json files in the specified directory
#useful for quickly cleaning up after a messed up training episode
def delete_latest_files(directory="/data/xgym/sweep"):
    dat_files = sorted(glob.glob(os.path.join(directory, "*.dat")), key=os.path.getctime)
    json_files = sorted(glob.glob(os.path.join(directory, "*.json")), key=os.path.getctime)

    if dat_files and json_files:
        latest_dat = dat_files[-1]
        latest_json = json_files[-1]

        print(f"Deleting:\n  {latest_dat}\n  {latest_json}")
        os.remove(latest_dat)
        os.remove(latest_json)
    else:
        print("No .dat or .json files found.")

if __name__ == "__main__":
    delete_latest_files()

