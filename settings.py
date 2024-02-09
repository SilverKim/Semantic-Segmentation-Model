data_name = "colon"
coefs = {
        'crs_ent': 1,
        'recon': 1,
        'kl': 1,
        'ortho': 1,
    }
if (data_name == "colon"):
    img_size = 256
    latent = 512
    num_prototypes = 20
    num_classes = 2
    batch_size = 128
    lr = 1e-3
    num_train_epochs = 150