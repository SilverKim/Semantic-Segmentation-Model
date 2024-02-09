#
# Created on Thu Feb 16 2023
#
# The MIT License (MIT)
# Copyright (c) 2023 Abdurahman A. Mohammed
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
# TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import torch 
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import glob as glob
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import albumentations as A
from Data.Colonoscopy import KvasirSEGDataset, KvasirClassification
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Unet model in pytorch lightning
class UNetLightning(pl.LightningModule):
    def __init__(self, in_channels=3, out_channels=1, lr=1e-4, batch_size=8):
        super(UNetLightning, self).__init__()
        self.save_hyperparameters()
        self.batch_size=batch_size
        self.lr = lr
        self.model = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
            in_channels=in_channels,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=out_channels,                      # model output channels (number of classes in your dataset)
        )
        self.criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")     

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")


        # calculate f1 score
        f1 = smp.metrics.f1_score(tp=tp, fp=fp, fn=fn, tn=tn, reduction="micro-imagewise")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_f1": f1,
        }
        
        self.log_dict(metrics, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")


    def prepare_data(self):
        # get all file names
        all_files = glob.glob('Data/kvasir/images/*.jpg')
        root_dir = "Data/kvasir"
        # keep the file names only
        all_files = [file.split('/')[-1] for file in all_files]

        # split the data into train, validation, and test
        train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
        val_files, test_files = train_test_split(val_files, test_size=0.2, random_state=42)
        # Define augmentation pipeline
        train_transform = A.Compose([
                    A.Resize(256, 256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Normalize()
            ])

        val_transform = A.Compose([

                    A.Resize(256, 256),
            ])

        
        test_transform = A.Compose([

                    A.Resize(256, 256),

            ])

        # Create dataset    
        self.train_dataset = KvasirSEGDataset(root_dir,train_files, transform=train_transform)
        self.val_dataset = KvasirSEGDataset(root_dir,val_files, transform=val_transform)
        self.test_dataset = KvasirSEGDataset(root_dir,test_files, transform=test_transform)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, pin_memory=True, shuffle=False)

    def shared_step(self, batch, stage):
        
        image, mask = batch

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0


        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4


        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.criterion(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }


# create a classification model class
class KvasirClassify(pl.LightningModule):
    # constructor
    def __init__(self, batch_size=8, lr=0.001, **kwargs):
        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.model = UNetLightning.load_from_checkpoint(checkpoint_path="checkpoints/unet-epoch=195-valid_per_image_iou=0.54.ckpt")

        # get the encoder from the model
        self.encoder = self.model.model.encoder

        # freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # create a classifier head
        self.classifier = nn.Sequential(
            nn.Linear(512*8*8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

        # add prototype for each class with each class having 10 prototypes with 256 features
        self.prototypes = nn.Parameter(torch.randn(10, 512), requires_grad=True)

        # self.prototype_class_identity 




        # create a loss function
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        # get features from the encoder
        features = self.encoder(x)

        # get the last feature map
        features = features[-1]

        # flatten the feature map
        features = features.view(features.size(0), -1)

        # pass features through the classifier
        logits = self.classifier(features)
        return logits

    def shared_step(self, batch, stage):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):    
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=False)

    def prepare_data(self):
        # get list of all files
        all_files = glob.glob('/Users/abdu/Desktop/Research/pytorch-image-classification/kvasir/images/*.jpg')

        # keep the file names only
        all_files = [file.split('/')[-1] for file in all_files]

        # get the root directory
        root_dir = "/Users/abdu/Desktop/Research/pytorch-image-classification/kvasir"

        # create a dataset
        self.train_dataset = KvasirClassification(root_dir, all_files,transform=None)

        self.val_dataset = self.train_dataset

        self.test_dataset = self.train_dataset

    

# create a classification model class
class Kvasir(pl.LightningModule):
    def __init__(self, batch_size=8, lr=0.001, **kwargs):
        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.model = UNetLightning.load_from_checkpoint(checkpoint_path="checkpoints/unet-epoch=195-valid_per_image_iou=0.54.ckpt")
        self.epsilon = 1e-4
        # get the encoder from the model
        self.encoder = self.model.model.encoder

        # freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # create a classifier head
        self.classifier = nn.Sequential(
            nn.Linear(512*8*8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

        self.adaptor = nn.Sequential(
            nn.Linear(512*8*8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512)
        )

        self.prototype_shape = (10,256)
        self.num_prototypes = 10
        self.num_classes = 2

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False) # do not use bias

        # add prototype for each class with each class having 10 prototypes with 256 features
        self.prototypes = nn.Parameter(torch.randn(10, 256), requires_grad=True)
        
        

        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)

        self.num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        
        self.prototype_vectors = nn.Parameter(torch.randn(self.prototype_shape),
                                              requires_grad=True)

        # create a loss function
        self.criterion = nn.BCEWithLogitsLoss()
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def distance_2_similarity(self, distances):
        return torch.log((distances + 1) / (distances + self.epsilon))

    def forward(self, x):
        # get features from the encoder
        features = self.encoder(x)
        
        # get the last feature map
        features_in = features[-1]

        # flatten the feature map
        features_in = features_in.view(features_in.size(0), -1)

        # pass features through the classifier
        logits = self.classifier(features_in)

        features = self.adaptor(features_in)

        

        return features,logits


    def calc_sim_scores(self, z):
        d = torch.cdist(z, self.prototype_vectors, p=2)  ## Batch size x prototypes
        sim_scores = self.distance_2_similarity(d)
        return sim_scores


    def kl_divergence_nearest(self, mu, logVar, nearest_pt, sim_scores):
        kl_loss = torch.zeros(sim_scores.shape).to(device)
        for i in range(self.num_prototypes_per_class):
            p = torch.distributions.Normal(mu, torch.exp(logVar / 2))
            p_v = self.prototype_vectors[nearest_pt[:,i],:]
            q = torch.distributions.Normal(p_v, torch.ones(p_v.shape).to(device))
            kl = torch.mean(torch.distributions.kl.kl_divergence(p, q), dim=1)
            kl_loss[np.arange(sim_scores.shape[0]),nearest_pt[:,i]] = kl
        kl_loss = kl_loss*sim_scores
        mask = kl_loss > 0
        kl_loss = torch.sum(kl_loss, dim=1) / (torch.sum(sim_scores * mask, dim=1))
        kl_loss = torch.mean(kl_loss)
        return kl_loss


    def ortho_loss(self):
        s_loss = 0
        for k in range(self.num_classes):
            p_k = self.prototype_vectors[k*self.num_prototypes_per_class:(k+1)*self.num_prototypes_per_class,:]
            p_k_mean = torch.mean(p_k, dim=0)
            p_k_2 = p_k - p_k_mean
            p_k_dot = p_k_2.T @ p_k_2
            s_matrix = p_k_dot - (torch.eye(p_k.shape[1]).to(device))
            s_loss+= torch.norm(s_matrix,p=2)
        return s_loss/self.num_classes


    def shared_step(self, batch, stage):

        x, y = batch
        features,logits = self.forward(x)
        
        mu = features[:, :256]
        logvar = features[:, 256:]

        z = self.reparameterize(mu, logvar)
        if stage == "train":
            z = mu
            

        sim_scores = self.calc_sim_scores(z)

        prototypes_of_correct_class = torch.t(self.prototype_class_identity[:, y]).to(device)

        index_prototypes_of_correct_class = (prototypes_of_correct_class == 1).nonzero(as_tuple=True)[1]
        index_prototypes_of_correct_class = index_prototypes_of_correct_class.view(x.shape[0],self.num_prototypes_per_class)

        kl_loss= self.kl_divergence_nearest(mu, logvar, index_prototypes_of_correct_class, sim_scores)
        out = self.last_layer(sim_scores)

        ortho_loss = self.ortho_loss()

        return {"loss": kl_loss,"ortho_loss": ortho_loss, "out": out}

    def training_step(self, batch, batch_idx):

        

        return self.shared_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    






