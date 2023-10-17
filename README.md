## ISIC CLassification

#### Create virtual Environment and Install dependencies
```
    /path/to/project/directory
    conda create -n isic
    conda activate isic
    pip3 install -r requirement.txt
```

#### Download Data
```
    /path/to/project/directory
    mkdir DATA

```
using this path <a>https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification/data</a> Download the data, place it in `DATA` directory.


#### About the Code
1- To Avoid writing training loop, pytorch lightining was used.
2- Same CNN architecture was used for binary and multiclass classification. Only the losses were changed
3- In `dataloader.py`, binary and multiclass labels are handeled.
4- In `unils.py`, some necessary functions such as plots are placed.
5- To ease the proces of logging the experiments, `TensorBoard` is used in the code.

#### Run the Experiments
you can run `original.ipynb` you can use defaulf setting to run the experiments or you cand change them to your desired batch_size, number of epochs, ....