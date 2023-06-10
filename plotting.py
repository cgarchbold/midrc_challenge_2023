import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

'''
    Takes a pytorch dataset and plots all images in the dataset
'''
def plot_dataset(dataset):
    
    #Exactly 1/3
    for j in range(int(len(dataset)/3)):
        fig, ax = plt.subplots(3, figsize=(8,20))

        for i,d_index in enumerate(range(j*3+1, j*3+4)):
            ax[i].imshow( dataset[d_index][0], cmap='gray')
            ax[i].axis('off')
            ax[i].set_title("Image")

            if i == 3-1:
                break
        
        plt.tight_layout()

        results_dir = os.path.join("plots","dset")

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        filename = os.path.join("plots","dset",str(j)+".png")

        plt.savefig(filename)
        plt.close()


'''
    Plots num random image samples from the given dataset and saves as filename.png
'''
def plot_random_img_samples(dataset,num,filename):
    fig, ax = plt.subplots(1,num, figsize=(20,8))

    l = len(dataset)

    for i in range(l):
        rand = random.randint(0,l-1)
        ax[i].imshow( dataset[rand][0], cmap='gray')
        ax[i].axis('off')
        ax[i].set_title(dataset[rand][1])

        if i == num-1:
            break
        
    plt.tight_layout()

    filename = os.path.join("plots",filename+".png")
    plt.savefig(filename)
    plt.show()

'''
    TODO: Plots the histograms and labels for the dataset statistics
'''
def plot_dataset_statistics(dataset):
    print("done")


'''

'''
def plot_bad_samples(dataset):
    indexes = []
    percent_zeros = []
    mrale_score = []

    for i in range(len(dataset)):
        as_np = np.array(dataset[i][0])
        indexes.append(i)
        percent_zeros.append(np.count_nonzero(as_np==0) / as_np.size )
        mrale_score.append(dataset[i][1])


    plt.hist(percent_zeros,bins=4)
    plt.title("Histogram of percent zeros")
    plt.xlabel("Percent zeros")
    plt.ylabel("Number of images")
    filename = os.path.join("plots",'zeroes_hist.png')
    plt.savefig(filename)
    plt.show()

    fig, ax = plt.subplots()
    plt.title("Percent zeros density")
    sns.kdeplot(percent_zeros)
    plt.xlabel("Percent zeros")
    filename = os.path.join("plots",'zeroes_density.png')
    plt.savefig(filename)
    plt.show()

    sort_index = np.argsort(np.array(percent_zeros))

    for c,index in enumerate(sort_index):
        plt.imshow( dataset[index][0], cmap='gray')
        plt.axis('off')
        #plt.title(str(percent_zeros[index]))

        plt.tight_layout()

        results_dir = os.path.join("plots","bad_sorted")

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        filename = os.path.join("plots","bad_sorted",str(c)+'_'+str(percent_zeros[index])+".png")

        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

def plot_ds_stats(dataset):
    # Counting the labels
    labels_per_study = []
    labels_per_image = []
    images_per_study = []

    for i in range(len(dataset)):
        labels_per_study.append(dataset[i][1])

        sum = 0
        for image in dataset[i][0]:
            labels_per_image.append(dataset[i][1])
            sum+=1
        images_per_study.append(sum)

    
    fig, ax = plt.subplots()
    plt.hist(labels_per_study, bins=24)
    plt.title("mRALE score counts for all studies")
    plt.xlabel("mRALE Score")
    plt.ylabel("Number of studies")
    plt.savefig('hist_studies.png')
    plt.show()

    #sns.set_style('whitegrid')
    fig, ax = plt.subplots()
    plt.title("Score density for all studies")
    sns.kdeplot(labels_per_study)
    ax.set_xlim(0, 24)
    plt.xlabel("mRALE Score")
    plt.savefig('density_studies.png')
    plt.show()

    fig, ax = plt.subplots()
    plt.hist(labels_per_image, bins=24, color='black')
    plt.title("mRALE score counts for all images")
    plt.xlabel("mRALE Score")
    plt.ylabel("Number of images")
    plt.savefig('hist_img.png')
    plt.show()

    #sns.set_style('whitegrid')
    fig, ax = plt.subplots()
    plt.title("Score density for all images")
    sns.kdeplot(labels_per_image,color='black')
    ax.set_xlim(0, 24)
    plt.xlabel("mRALE Score")
    plt.savefig('density_img.png')
    plt.show()

    fig, ax = plt.subplots()
    plt.hist(images_per_study, bins=24, color='black')
    plt.title("Images per study")
    plt.xlabel("Number of images")
    plt.ylabel("Number of studies")
    plt.savefig('hist_images.png')
    plt.show()

def plot_train_metrics(folds, saved_metrics):
    for f_i,fold in enumerate(folds):
        fold_metrics = saved_metrics[f_i]

        plt.figure(figsize=(12,8))
        plt.title("Losses for Fold: "+ str(f_i+1))
        plt.plot([fold_metrics[epoch]['avg_loss'] for epoch in fold_metrics], label='train_loss')
        plt.plot([fold_metrics[epoch]['avg_vloss'] for epoch in fold_metrics],label='val_loss')
        plt.legend()
        plt.xlabel("Epochs")
        filename = os.path.join("plots",str(f_i+1)+'_losses.png')
        plt.savefig(filename)
        plt.show()

        plt.figure(figsize=(12,8))
        plt.title("Kappas for Fold: "+ str(f_i+1))
        plt.plot([fold_metrics[epoch]['avg_train_kappa'] for epoch in fold_metrics], label='train_kappa')
        plt.plot([fold_metrics[epoch]['avg_val_kappa'] for epoch in fold_metrics],label='val_kappa')
        plt.legend()
        plt.xlabel("Epochs")
        filename = os.path.join("plots",str(f_i+1)+'_kappas.png')
        plt.savefig(filename)
        plt.show()