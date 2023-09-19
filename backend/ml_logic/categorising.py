import os
import shutil

def categorise_chula_rbc12():
    '''For the Chula-RBC-12-Dataset, finds inside the .txt files (Labels folder) if there are spherocytes identified.
    Each .txt file corresponds to a blood smear (same number for the .jpg file and the .txt file) and has a line for each RBC identified.
    Each line has three numbers: the x coordinates of the RBC, the y coordinates of the RBC and the label for the RBC types (see the README.md).
    This function iterates over each file and then over each line until it finds a spherocyte label (3).
    If spherocyte is found, it copies the equivalent .jpg to the "positive" folder. Otherwise, it copies it to the "negative" folder
    '''
    parent_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    label_path = os.path.join(parent_path, 'raw_data', 'Chula-RBC-12-Dataset-main', 'Label')
    data_path = os.path.join(parent_path, 'raw_data', 'Chula-RBC-12-Dataset-main', 'Dataset')
    positive_folder = os.path.join(parent_path, 'raw_data', 'data', 'positive')
    negative_folder = os.path.join(parent_path, 'raw_data', 'data', 'negative')

    try:
        # Iterate over all the files in the data_path
        for filename in os.listdir(label_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(label_path, filename)

                # Initialize a flag to check if a spherocyte is found
                spherocyte_found = False

                # Check image file exists
                data_file = os.path.join(data_path, filename.replace(".txt", ".jpg"))
                if data_file:

                    # Open the text file and read its lines
                    with open(file_path, "r") as txt_file:
                        for line in txt_file:
                            # Split the line into three numbers
                            numbers = line.strip().split()
                            if len(numbers) == 3:
                                # Check if the third number is 3, the label for spherocyte
                                if numbers[2] == '3':
                                    spherocyte_found = True
                                    break  # No need to check further lines, we found a 3

                    # Copies jpg file to the "positive" or "negative" folder based on whether 3 was found
                    if spherocyte_found:
                        destination_file = os.path.join(positive_folder, filename.replace(".txt", ".jpg"))
                        shutil.copy(data_file, destination_file)
                    else:
                        destination_file = os.path.join(negative_folder, filename.replace(".txt", ".jpg"))
                        shutil.copy(data_file, destination_file)

        print(f'Files from {data_path} copied to /data folders successfully.')

    except Exception as e:
        print(f'Error: {e}')

def categorise_dataset_a():
    '''Loads dataset A: 186 digital images of MGG-stained blood smears from five patients with hereditary spherocytosis.
    All blood smear images will be mixed (not divided per patient).
    All jpgs are copied into the "positive" folder
    '''

    parent_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(parent_path, 'raw_data', 'Dataset A')
    positive_folder = os.path.join(parent_path, 'raw_data', 'data', 'positive')

    try:
        # Iterate over all the patient folders in the directory
        for patient_folder in os.listdir(data_path):
            if os.path.isdir(os.path.join(data_path, patient_folder)):
                patient_folder_path = os.path.join(data_path, patient_folder)
                # Iterate over files within the patient folder
                for filename in os.listdir(patient_folder_path):
                    if filename.endswith('.jpg'):
                        data_file = os.path.join(patient_folder_path, filename)
                        destination_file = os.path.join(positive_folder, filename)
                        shutil.copy(data_file, destination_file)

                print(f'File from {patient_folder_path} copied to {positive_folder} successfully.')

    except Exception as e:
        print(f'Error: {e}')
