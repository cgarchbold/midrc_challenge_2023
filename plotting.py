import random
import matplotlib.pyplot as plt
import seaborn as sns
import os

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