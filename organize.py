import os
import shutil
import zipfile


'''
    Code to reorganize the original dataset
'''
def search_and_extract(root_dir, dest_dir):

    #for each directory (data-endpoint?)
    for data_id in os.listdir(root_dir):

        data_path = os.path.join(root_dir, data_id)
        if not os.path.isdir(data_path):
            continue
        
        #for each study
        for case_id in os.listdir(data_path):
            case_path = os.path.join(data_path, case_id)
            if not os.path.isdir(case_path):
                continue
            
            #for each image
            for file_name in os.listdir(case_path):
                file_path = os.path.join(case_path, file_name)
                if not file_name.endswith('.zip'):
                    continue
                
                os.makedirs(dest_dir, exist_ok=True)
                #unzip
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    for file in zip_ref.namelist():
                        if file.endswith('.dcm'):
                            filename = os.path.basename(file)
                            source = zip_ref.open(file)
                            target = open(os.path.join(dest_dir, filename), "wb")
                            with source, target:
                                shutil.copyfileobj(source, target)

                
                #print(f"Extracted images from {file_path} to {dest_dir}")

source_directory = '\\Users\\CGarc\\University of Kentucky\\Ahamed, Md. Atik - data\\original'
destination_directory = '\\Users\\CGarc\\University of Kentucky\\Ahamed, Md. Atik - data\\organized\\'

search_and_extract(source_directory, destination_directory)