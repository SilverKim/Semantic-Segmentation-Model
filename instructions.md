# SegExplain
This is the readme for the SegExplain project.
## Running the code
### Setup
Install the required packages using `pip install -r requirements.txt`.

To retrain the semantic segmentation network, make sure you have downloaded the Kvasir-SEG dataset and placed it in the Data folder as 'Data/kvasir'. The dataset can be downloaded from [here](https://datasets.simula.no/kvasir-seg/).

### Running the code
To train the segmentation model, run the following command:
```
Python train.py
```

To train the explainer model, run the following command:
```
Python main.py -data=colon -mode=train
```

To get explanations demo, open the notebook 'demo.ipynb' and run the cells following the instructions in the notebook.

## Project structure
The project is structured as follows:
- Data: The segmentation and classification datasets are stored here.
- Models: The segmentation models are stored here.
- saved_models: The trained explainer model is stored here.

## Pretrained models
- A pretrained segmentation model is placed in project directory with the name 'unet-epoch=195-valid_per_image_iou=0.54.ckpt'. This model is trained on the Kvasir-SEG dataset.
- A pretrained explainer model is placed in the 'saved_models/colon' directory with the name 'model.pth'. It is trained on the classification dataset.
  
## Acknowledgements
The code for the explainer model is adapted from the ProtoVAE project, which can be found [here](https://github.com/SrishtiGautam/ProtoVAE).
