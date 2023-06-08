import json
from sklearn.model_selection import StratifiedGroupKFold

'''
    Returns a list of tuples: [(train_img_names,val_img_names)] 
    Which can be used to create the 
'''
def create_folded_datasets(json_path):
    
    folds = []

    with open(json_path,"r") as label_info:
        resized_json_file=json.load(label_info)
    
    label_hist_scores=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    all_mrale_scores=[]
    all_patient_ids=[]
    for cur_info in resized_json_file:
        all_mrale_scores.append(cur_info['mRALE Score'])
        all_patient_ids.append(cur_info['PatientID'])
    
    splitter=StratifiedGroupKFold(n_splits=5,random_state=48,shuffle=True) #THIS WILL TRY TO SPLIT THE DISTRIBUTION BETWEEN TRAIN, VALIDATION EVENLY BASED ON MRALE SCORES
    for i, (train_idx,val_idx) in enumerate(splitter.split(resized_json_file,all_mrale_scores,all_patient_ids)):
        print("Fold: ",i+1," Train Indices: ", len(train_idx), " Val indices: ", len(val_idx))
        cur_fold_train_img_names=[]
        cur_fold_train_patient_ids=[]
        cur_fold_train_mrale_scores=[]
    
        cur_fold_val_img_names=[]
        cur_fold_val_patient_ids=[]
        cur_fold_val_mrale_scores=[]
        for cur_fold_train_idx in train_idx:
            cur_fold_train_patient_ids.append(resized_json_file[cur_fold_train_idx]['PatientID'])
            cur_fold_train_img_names.append(resized_json_file[cur_fold_train_idx]['full_image_name'])
            cur_fold_train_mrale_scores.append(resized_json_file[cur_fold_train_idx]['mRALE Score'])
        
        for cur_fold_val_idx in val_idx:
            cur_fold_val_patient_ids.append(resized_json_file[cur_fold_val_idx]['PatientID'])
            cur_fold_val_img_names.append(resized_json_file[cur_fold_val_idx]['full_image_name'])
            cur_fold_val_mrale_scores.append(resized_json_file[cur_fold_val_idx]['mRALE Score'])
        
        ###START: CONSISTENCY CHECK########
        ## THIS PORTION OF CODE WILL CHECK IF THERE IS ANY OVERLAP BETWEEN THE TRAIN AND VALIDATION IN-TERMS OF PATIENT_ID AND IMAGE_NAME FOR CURRENT FOLD
        for cur_train_img_name in cur_fold_train_img_names:
            if cur_train_img_name in cur_fold_val_img_names:
                print("!!!!!!!!!!!!!!Overlap in image level!!!!!!!!!")
        for cur_train_patient_id in cur_fold_train_patient_ids:
            if cur_train_patient_id in cur_fold_val_patient_ids:
                print("!!!!!!!!!!!!!!Overlap in patient level!!!!!!!!!")
        ###END: CONSISTENCY CHECK###########
        
        ### TODO: AT THIS POINT THE DATALOADER CAN BE INITIALIZED USING cur_fold_train_img_names, cur_fold_train_mrale_scores, cur_fold_val_img_names, cur_fold_val_mrale_scores
        folds.append((cur_fold_train_img_names,cur_fold_val_img_names))

    return folds