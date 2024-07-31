# Label Unbalance in High Frequency Trading



## Method

In this project, we mainly focus on the high frequency trading prediction in the scenario of label imbalance, based on machine learning methods. We provide four networks including: MLP, LSTM, BERT, and Mamba as the backbone. We use two simple methods to reduce the influence of label imbalance, including: resampling and class weighting.



## Dataset

Due to copyright restrictions, we do not provide the original data. However, we provide the data structure and class distribution in our experiment. As a result, you can replace the `dataset.py` file with your own data and change some necessary parameters in `train_classification.py` to run the code on your dataset.

**TODO: data description** 



## How to Run

To run the model, you can use the following command:

```bash
python train_classification.py --input_dim <input dim> --output_dim <class numbers> --model <please select from 'mlp', 'lstm', 'bert', and 'mamba'>
```

To change the model scale or other parameters related to training, please refer to the `get_args` function.

We provide the following methods for addressing label imbalance:

- `--class_weight`: You can specify the weight for each class in the loss function.

- `--data_balance`: We randomly delete 7/8 of the data in the '0' class to balance the dataset.

  

## Experiment Result

