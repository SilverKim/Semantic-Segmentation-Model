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
from matplotlib.pyplot import show
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_workers = 4 if torch.cuda.is_available() else 0

model_dir = './saved_models/' + data_name + '/'
makedir(model_dir)
prototype_dir = model_dir + 'prototypes/'
makedir(prototype_dir)

parser = argparse.ArgumentParser()
parser.add_argument('-data', nargs=1, type=str, default=['mnist'])
parser.add_argument('-mode', nargs=1, type=str, default=['test'])
parser.add_argument('-model_file', nargs=1, type=str, default=['saved_models/MNIST/MNIST.pth'])
parser.add_argument('-expl', nargs=1, type=bool, default=[False])
args = parser.parse_args()
data_name = args.data[0]
mode = args.mode[0]
model_file = args.model_file[0]
expl = args.expl[0]

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

print('data : ',data_name)
print('training set size: {0}'.format(len(train_loader.dataset)))
print('test set size: {0}'.format(len(test_loader.dataset)))



jet = True


# construct the model
protovae = model.ProtoVAE().to(device)

## Training
if(mode=="train"):
    print('start training')
    optimizer_specs = \
            [{'params': protovae.features.parameters(), 'lr': lr},
             {'params': protovae.prototype_vectors, 'lr':lr},
             {'params': protovae.decoder_layers.parameters(), 'lr':lr},
             {'params': protovae.last_layer.parameters(), 'lr':lr}
             ]
    optimizer = torch.optim.Adam(optimizer_specs)

    for epoch in range(num_train_epochs):
        print('epoch: \t{0}'.format(epoch))
        train_acc, train_ce, train_recon, train_kl, train_ortho = tnt.train(model=protovae, dataloader=train_loader,
                                                               optimizer=optimizer)

        test_acc, test_ce, test_recon, test_kl, test_ortho = tnt.test(model=protovae, dataloader=test_loader)


    print("saving..")
    save.save_model_w_condition(model=protovae, model_dir=model_dir, model_name=str(epoch), accu=test_acc,
                                target_accu=0)

    ## Save and plot learned prototypes
    protovae.eval()
    prototype_images = protovae.get_prototype_images()
    prototype_images = (prototype_images + 1) / 2.0
    num_prototypes = len(prototype_images)
    num_p_per_class = protovae.num_prototypes_per_class

    plt.figure("Prototypes")
    for j in range(num_prototypes):
        p_img_j = prototype_images[j, :, :, :].detach().cpu().numpy()
        if(jet!=True):
            p_img_j = np.transpose(p_img_j, (1, 2, 0))
        else:
            p_img_j = np.squeeze(p_img_j)

        if(jet!=True):
            plt.imsave(os.path.join(prototype_dir, 'prototype' + str(j) + '.png'), p_img_j, vmin=0.0, vmax=1.0)
        else:
            plt.imsave(os.path.join(prototype_dir, 'prototype' + str(j) + '.png'), p_img_j, jet="True",vmin=0.0, vmax=1.0)

        plt.subplot(num_classes, num_p_per_class, j + 1)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis('off')
        plt.imshow(p_img_j)

    plt.show()
    print("Prototypes stored in: ", prototype_dir)

elif (mode == 'test'):
    print('Please use the Demo.ipynb notebook to test the model')
    exit()


