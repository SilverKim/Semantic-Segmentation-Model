import model
import os
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from helpers import makedir
import model
import train_and_test as tnt
import save
import matplotlib.pyplot as plt
import numpy as np
from settings import *
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
from skimage.measure import block_reduce

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_workers = 4 if torch.cuda.is_available() else 0

protovae = model.ProtoVAE().to(device)

data_name = "colon"

# load model
protovae.load_state_dict(torch.load('saved_models/colon/model.pth',map_location=torch.device('cpu')),strict=False)
protovae.eval()

protovae.to(device)



print(f"Loaded model successfully!")

mean = (0.5)
std = (0.5)
# transform = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)
#     ])
# trainset = datasets.MNIST(root=data_path, train=True,
#                           download=True, transform=transform)
# testset = datasets.MNIST(root=data_path, train=False,
#                              download=True, transform=transform)
if (data_name == "colon"):
    trainset = datasets.ImageFolder(root=os.path.join("Data","Colon","train"),transform = transforms.Compose(
        [

            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
    testset = datasets.ImageFolder(root=os.path.join("Data","Colon","test"),transform = transforms.Compose(
        [
            
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=True, num_workers=num_workers)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=num_workers)

test_loader_expl = torch.utils.data.DataLoader(testset, batch_size=1,
                                          shuffle=False, num_workers=num_workers)

print(f"Created the Datasets Successfully")

batch = next(iter(train_loader))

input = batch[0].to(device)



for i, (image, label) in enumerate(test_loader):
        if i == 0:
            continue

        input = image.to(device)
        target = label.to(device)


        print(f"Labels: {target.shape}")
        print(f"Images: {input.shape}")

        with torch.no_grad():
            output, decoded, kl_loss, orth_loss,sim_scores = protovae(input, label, False)

        # reverse the normalization with mean = (0.5, 0.5, 0.5) and std = (0.5, 0.5, 0.5)
        input = input * 0.5 + 0.5


        # plot the original image
        plt.imshow(input[125].cpu().numpy().transpose(1,2,0), cmap='gray')
        plt.show()

        print(f"The predicted class is: {torch.argmax(output[0])}")


        prototype_images = protovae.get_prototype_images()

        # reverse the normalization with mean = (0.5, 0.5, 0.5) and std = (0.5, 0.5, 0.5)
        prototype_images = prototype_images * 0.5 + 0.5

        # plot the prototype images with 4x5 subplot
        fig = plt.figure(figsize=(10, 8))
        for i in range(protovae.num_prototypes):
            ax = fig.add_subplot(4, 5, i + 1)
            ax.imshow(prototype_images[i].detach().numpy().transpose(1,2,0), cmap='gray')
            ax.axis('off')
        plt.show()
        




        # fig, ax = plt.subplots(2, 10, figsize=(10, 10))
        # for i in range(2):
        #     for j in range(10):
        #         ax[i, j].imshow(prototype_images[i * 10 + j].detach().numpy().transpose(1,2,0))
        #         ax[i, j].axis('off')
        # plt.show()

        # Find the similarity scores of the input image with the prototypes
        sim_scores = sim_scores[125].cpu().numpy()

        # Plot the top 5 prototypes with the highest similarity scores
        fig, ax = plt.subplots(1, 5, figsize=(10, 5))
        for i in range(5):
            ax[i].imshow(prototype_images[np.argsort(sim_scores)[-5:][i]].detach().numpy().transpose(1,2,0), cmap='gray')
            # put the similarity score as the title of the image
            ax[i].set_title(f"{sim_scores[np.argsort(sim_scores)[-5:][i]]:.5f}")
            ax[i].axis('off')
        plt.show()
              

        break


exit()


from MNIST_SEG import MNIST_SEG
import numpy as np
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE  = 16

class MNIST_DATASET(torch.utils.data.Dataset):
    def __init__(self, images, labels, weights):
        self.images = images
        self.labels = labels
        self.weights = weights
        
    def __getitem__(self, index):
        # Split the label into 11 channels
        return self.images[index], self.labels[index], self.weights[index]
    
    def __len__(self):
        return len(self.images)

# read mnist_seg_train_images.pt, mnist_seg_train_labels.pt, mnist_seg_train_weights.pt
# read mnist_seg_test_images.pt, mnist_seg_test_labels.pt, mnist_seg_test_weights.pt
mnist_train_images  = torch.load('/Users/abdu/Desktop/Research/Interp/mnist_seg_data/mnist_seg_train_images.pt')
mnist_train_labels  = torch.load('/Users/abdu/Desktop/Research/Interp/mnist_seg_data/mnist_seg_train_labels.pt')
mnist_train_weights = torch.load('/Users/abdu/Desktop/Research/Interp/mnist_seg_data/mnist_seg_train_weights.pt')
mnist_test_images   = torch.load('/Users/abdu/Desktop/Research/Interp/mnist_seg_data/mnist_seg_test_images.pt')
mnist_test_labels   = torch.load('/Users/abdu/Desktop/Research/Interp/mnist_seg_data/mnist_seg_test_labels.pt')
mnist_test_weights  = torch.load('/Users/abdu/Desktop/Research/Interp/mnist_seg_data/mnist_seg_test_weights.pt')

print(mnist_train_images.shape)

train_dataset = MNIST_DATASET(mnist_train_images, mnist_train_labels, mnist_train_weights)
test_dataset  = MNIST_DATASET(mnist_test_images, mnist_test_labels, mnist_test_weights)

# split train_dataset into train and val
train_size = int(0.8 * len(train_dataset))
val_size   = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)


segment = MNIST_SEG().load_from_checkpoint('/Users/abdu/Desktop/Research/Interp/mnist_seg_data/lightning_logs/version_15/checkpoints/epoch=6-step=3500.ckpt').eval()

x, y, _ = random.choice(test_dataset)
x = x.unsqueeze(0)

# make prediction on the image
pred = segment(x).squeeze()

# define a method given a binary image, it will crop the image to the smallest bounding box. If the image is all zeros, it will return the original image
def crop_image(img):
    non_zero_coords = torch.argwhere(img[0] > 0)
    if len(non_zero_coords) == 0:
        return img
    else:
        min_x = non_zero_coords[:, 0].min()
        max_x = non_zero_coords[:, 0].max()
        min_y = non_zero_coords[:, 1].min()
        max_y = non_zero_coords[:, 1].max()
        return img[:, min_x:max_x, min_y:max_y]




for i in range(1,11):
    input = pred[i].unsqueeze(0).unsqueeze(0) > 0.5
    input = input.float()
    print(f"The actual label is :{i-1} and ")
    print(input.shape)
    input = crop_image(input)
    print(input.shape)

    # resize the input to 28x28 
    input = torch.nn.functional.interpolate(input, size=(28, 28), mode='bilinear', align_corners=False)

    


    plt.imshow(input.detach().numpy().squeeze(), cmap='gray')
    plt.show()

    label = torch.tensor([i-1])

    # make a prediction using protovae
    with torch.no_grad():
        
        output, decoded, kl_loss, orth_loss,sim_scores = protovae(input, label, False)


    print(f"The predicted class is: {torch.argmax(output[0])}")

    prototype_images = protovae.get_prototype_images()

    # # plot the prototype images with 5x10 subplot
    # fig, ax = plt.subplots(5, 10, figsize=(10, 5))

    # for i in range(5):
    #     for j in range(10):
    #         ax[i, j].imshow(prototype_images[i * 10 + j].detach().numpy().reshape(28, 28), cmap='gray_r')
    #         ax[i, j].axis('off')

    # plt.show()

    # Find the similarity scores of the input image with the prototypes
    sim_scores = sim_scores[0].cpu().numpy()

    # Plot the top 5 prototypes with the highest similarity scores
    fig, ax = plt.subplots(1, 5, figsize=(10, 5))
    for i in range(5):
        ax[i].imshow(prototype_images[np.argsort(sim_scores)[-5:][i]].detach().numpy().reshape(28, 28), cmap='gray')
        # put the similarity score as the title of the image
        ax[i].set_title(f"{sim_scores[np.argsort(sim_scores)[-5:][i]]:.2f}")
        ax[i].axis('off')
    plt.show()




