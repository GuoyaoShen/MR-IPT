import os
import h5py
import numpy as np
import pickle

# number of slices to be cut at the head and tail of each volume
NUM_CUT_SLICES = 5


def create_fastmri_data_info(data_dir,
                             data_info_dir,
                             num_files=-1,
                             num_pd_files=-1,
                             num_pdfs_files=-1,
                             data_info_file_name="pd_train_info"):
    """
    given data directories to form .pkl file stores all volume and slice information

    :param data_dir:str, directory to store the mri .h5 data
    :param data_info_dir:str, directory to store the output data info file
    :param num_files:int, number of file choose to read, default -1 leads to read all file in dir
    :param num_pd_files:int, number of file to read without fat suppression, default -1
    :param num_pdfs_files:int, number of file to read with fat suppression, default -1
    :param data_info_file_name:str, output .pkl file name
    :return: a list store element (filename (str), index (int))
    """

    # check and create data_info_dir
    isExist = os.path.exists(data_info_dir)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(data_info_dir)
        print("The data_info directory is created!")

    # load file names in a list
    file_list = []
    for entry in sorted(os.listdir(data_dir)):
        ext = entry.split(".")[-1]
        if "." in entry and ext == "h5":
            file_list.append(entry)
            if num_files == len(file_list):
                break

    # load data info in a list
    data_info_list = []
    pd_count = 0
    pdfs_count = 0
    for file_name in file_list:
        if pd_count == num_pd_files and pdfs_count == num_pdfs_files:
            break

        file_path = os.path.join(data_dir, file_name)
        data = h5py.File(file_path, mode="r")
        image_rss = np.array(data["reconstruction_rss"])
        acquisition = data.attrs["acquisition"]
        num_slice = len(image_rss)

        if acquisition == "CORPD_FBK":
            if pd_count == num_pd_files:
                continue
            for i in range(NUM_CUT_SLICES, num_slice - NUM_CUT_SLICES):
                data_info_list.append((file_name, i))
            pd_count += 1
        else:
            if pdfs_count == num_pdfs_files:
                continue
            for i in range(NUM_CUT_SLICES, num_slice - NUM_CUT_SLICES):
                data_info_list.append((file_name, i))
            pdfs_count += 1

    with open(os.path.join(data_info_dir, f"{data_info_file_name}.pkl"), "wb") as f:
        pickle.dump(data_info_list, f)
        print(f"{data_info_file_name}, num of slices: {len(data_info_list)}")

    return data_info_list
