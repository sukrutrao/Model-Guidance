## Downloading and Pre-processing Datasets

For each dataset, use the provided preprocessing script for each of the train, validation, and test scripts as follows:

```bash
python preprocess.py --split train
python preprocess.py --split val
python preprocess.py --split test
```

For COCO2014, the dataset needs to be first downloaded using [download.sh](COCO2014/download.sh).

### Acknowledgements

The scripts provided here build upon scripts from [stevenstalder/NN-Explainer](https://github.com/stevenstalder/NN-Explainer).


