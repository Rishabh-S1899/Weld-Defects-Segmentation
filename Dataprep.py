import aspose.zip as az
import os
import shutil
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import json

input_path='RIAWELC_dataset'
output_path='Combined_RIAWELC_Dataset'

# print(dir_list)
# print([x[0] for x in os.walk(input_path)][1:5])
def extractor(input_path='RIAWELC_Downloaded\Dataset_partitioned',outputpath='RIAWELC_dataset'):
    file_list=os.listdir(input_path)
    print("This is the file list: ",file_list)
    # outputpath=os.makedirs(output_path,exist_ok=True)
    # input_dirs=[x[0] for x in os.walk(input_path)]
    print('\n')
    # print(input_dirs)
    for file in file_list:
        with az.rar.RarArchive(f'{input_path}\{file}') as archive:
            archive.extract_to_directory(f'{outputpath}/{file.split(".")[1]}')
            print(f'Extracted {input_path}/{file.split(".")[1]}')

def file_sorter(inputpath):
    # Create the destination folders
    dest_dir='Combined_RIAWELC_Dataset'
    for folder_name in ['training', 'testing', 'validation']:
        os.makedirs(os.path.join(dest_dir, folder_name), exist_ok=True)
        for difetto_folder in ['Difetto1', 'Difetto2', 'Difetto4', 'NoDifetto']:
            os.makedirs(os.path.join(dest_dir, folder_name, difetto_folder), exist_ok=True)
    #Moving Items from extracted to sorted manner
    # columns=['FileName','Label']
    df1=pd.DataFrame()
    df2=pd.DataFrame()
    df3=pd.DataFrame()
    defect_label=['cracks (CR)', 'porosity (PO)', 'lack of penetration (LP)' ,'no defect (ND)']
    filename_train=[]
    labels_validation=[]
    filename_validation=[]
    labels_test=[]
    filename_test=[]
    labels_train=[]
    dir_list=[x[0] for x in os.walk(inputpath)]
    folder_arr=['training','testing','validation','Difetto1','Difetto2','Difetto4','NoDifetto']
    for i in dir_list:
        print(i)
        for k in folder_arr[0:3]:
            dest_dir='Combined_RIAWELC_Dataset'
            dest_dir=os.path.join(dest_dir,k)
            if(k in i):
                for l in range(len(folder_arr[3:7])):
                    if(folder_arr[l] in i):
                        for j in os.listdir(i):
                            # print(f'Copied file was: {os.path.join(i,j)}')
                            # shutil.copy(os.path.join(i,j),f'{dest_dir}')
                            src_file = os.path.join(i, j)
                            if os.path.isfile(src_file):  # Ensure it's a file and not a directory
                                print(f'Copying file: {src_file}')
                                shutil.copy2(src_file, dest_dir)
                                print(f'Copied file to: {dest_dir}')
                            if(k=='training'):
                                filename_train.append(j)
                                labels_train.append(defect_label[l])
                            elif(k=='validation'):
                                filename_validation.append(j)
                                labels_validation.append(defect_label[l])
                            else:
                                filename_test.append(j)
                                labels_test.append(defect_label[l])

    df1['Filename']=filename_train
    df1['Label']=labels_train
    df2['Filename']=filename_validation
    df2['Label']=labels_validation
    df3['Filename']=filename_test
    df3['Label']=labels_test
    df1.to_csv('Train_Labels.csv')
    df2.to_csv('Validation_Labels.csv')
    df3.to_csv('Test_Labels.csv')
    #Removing the extracted folder
    shutil.rmtree(input_path)

def One_hot_encoder(train_file, val_file, test_file):
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    
    # Combine the train, validation, and test datasets
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Create an instance of OneHotEncoder
    encoder = OneHotEncoder()
    
    # Fit the encoder on the combined dataset
    encoder.fit(combined_df['Label'].values.reshape(-1, 1))
    
    # Transform the train dataset
    train_one_hot_encoded = encoder.transform(train_df['Label'].values.reshape(-1, 1)).toarray()
    train_df['One-Hot-Encoding'] = [json.dumps(list(row)) for row in train_one_hot_encoded]
    
    # Transform the validation dataset
    val_one_hot_encoded = encoder.transform(val_df['Label'].values.reshape(-1, 1)).toarray()
    val_df['One-Hot-Encoding'] = [json.dumps(list(row)) for row in val_one_hot_encoded]
    
    # Transform the test dataset
    test_one_hot_encoded = encoder.transform(test_df['Label'].values.reshape(-1, 1)).toarray()
    test_df['One-Hot-Encoding'] = [json.dumps(list(row)) for row in test_one_hot_encoded]
    
    # Save the updated DataFrames as new CSV files
    train_df.to_csv('Processed_train.csv', index=False)
    val_df.to_csv('Processed_val.csv', index=False)
    test_df.to_csv('Processed_test.csv', index=False)
    
    # Remove the original files
    # os.remove(train_file)
    # os.remove(val_file)
    # os.remove(test_file)

def check_and_remove_common_file_names(csv_file1, csv_file2):
    # Read the CSV files
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    
    # Check if 'Filename' column exists in both DataFrames
    if 'Filename' not in df1.columns or 'Filename' not in df2.columns:
        raise ValueError("Both CSV files must contain the 'Filename' column.")
    
    # Extract the 'Filename' columns
    file_names1 = set(df1['Filename'])
    file_names2 = set(df2['Filename'])
    
    # Find common file names
    common_file_names = file_names1.intersection(file_names2)
    
    # Remove common file names from df2
    if common_file_names:
        df2_filtered = df2[~df2['Filename'].isin(common_file_names)]
        print(f"Removed {len(df2) - len(df2_filtered)} rows with common file names.")
    else:
        df2_filtered = df2
        print("No common file names found.")
    
    # Overwrite the original CSV file with the updated DataFrame
    df2_filtered.to_csv(csv_file2, index=False)
    print(f"Original CSV file {csv_file2} updated.")

def remove_entries_with_words(csv_file, words_list):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Check if 'Filename' column exists in the DataFrame
    if 'Filename' not in df.columns:
        raise ValueError("The CSV file must contain the 'Filename' column.")
    
    # Create a pattern that matches any of the words in the list
    pattern = '|'.join(words_list)
    
    # Filter out rows where 'Filename' contains any of the words
    df_filtered = df[~df['Filename'].str.contains(pattern, case=False, na=False)]
    
    # Save the filtered DataFrame back to the original CSV file
    df_filtered.to_csv(csv_file, index=False)
    print(f"Updated CSV file saved as {csv_file}")


def main():
    extractor()
    file_sorter(input_path)
    One_hot_encoder('Train_Labels.csv','Validation_Labels.csv','Test_Labels.csv')
    check_and_remove_common_file_names('Processed_train.csv','Processed_test.csv')
    words=['Difetto1','Difetto2','Difetto4','NoDifetto']
    remove_entries_with_words('Processed_test.csv', words)
    remove_entries_with_words('Processed_train.csv', words)
    remove_entries_with_words('Processed_val.csv', words)
    print('all executed')
    # with az.rar.RarArchive('RIAWELC_Downloaded\Dataset_partitioned\RIAWELC_dataset.part01.rar') as archive:
        # archive.extract_to_directory(f'part1')

if __name__=='__main__':
    main()


