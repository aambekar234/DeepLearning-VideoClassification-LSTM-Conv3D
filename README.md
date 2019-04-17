# DeepLearning-VideoClassification-LSTM-Conv3D
## Deep learning video classification analyzed by three different networks
- CNN LSTM (LRCN)
- LSTM (Feature Extracted through Imagenet)
- Convolutional 3D

### Used Following Dataset for analysis
- [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
- [NADA](http://www.nada.kth.se/cvap/actions/)

### Similar Datasets
- [Link](https://www.di.ens.fr/~miech/datasetviz/)


## Requirements
Keras, numpy, tqdm, Pillow, h5py

## Usage
1. Download Video Data and Extract
2. Generate Frames Data and Features Data
3. Train model


Generate Frames and Features 
```
from data_gen import DataGen

//path to the extracted downloaded data, frames to be extracted per video

gen = DataGen("hmdb/", fpv=30)    
gen.generate_data()

```

Train your model ( refer train.py)
```
//train(model, path_to_generated_data_dir)
//available models -> lstm, lrcn, c3d
train("lstm", "hmdb_op_30/")
```

## Data Split
- If you are using HMDB Dataset 
    - Put the Splits folder into the root of Data Output Folder
    - Use following setting while training
    ```
    train("lstm", "hmdb_op_30/", split_strat=2)
    ```
- If you are using NADA Dataset
    - Use following setting while training
    ```
    train("lstm", "nada_op_30/", split_strat=1,split=0.36)
    ```

## Accuracy
### HMDB - 30 Frames Extraction
- LSTM 
    1. Categorical - 91%
    2. Overall - 61%



