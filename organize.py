import os
import shutil
import zipfile

def search_and_extract(root_dir, dest_dir):
    for data_id in os.listdir(root_dir):
        print(data_id)
        data_path = os.path.join(root_dir, data_id)
        if not os.path.isdir(data_path):
            continue
        
        for case_id in os.listdir(data_path):
            #print(case_id)
            case_path = os.path.join(data_path, case_id)
            #print(case_path)
            if not os.path.isdir(case_path):
                continue
            
            for file_name in os.listdir(case_path):
                file_path = os.path.join(case_path, file_name)
                if not file_name.endswith('.zip'):
                    continue
                
                dest_case_dir = os.path.join(dest_dir, case_id)
                os.makedirs(dest_case_dir, exist_ok=True)
                
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(dest_case_dir)
                
                print(f"Extracted images from {file_path} to {dest_case_dir}")

source_directory = '\\Users\\CGarc\\University of Kentucky\\Ahamed, Md. Atik - data\\original'
destination_directory = './organized/'

search_and_extract(source_directory, destination_directory)