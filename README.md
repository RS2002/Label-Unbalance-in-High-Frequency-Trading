# Label Unbalance in High Frequency Trading

**Report:** [Label Unbalance in High-frequency Trading](https://arxiv.org/abs/2503.09988)

[高频交易中的标签不平衡问题研究 -- QuantML](https://mp.weixin.qq.com/s/Uhp8GDqV89cYzAoTNbpRQw)


![](./img/main.png)

## Method

In this project, we mainly focus on the high frequency trading prediction in the scenario of label imbalance, based on machine learning methods. We provide four networks including: MLP, LSTM, BERT, and Mamba as the backbone. We use two simple methods to reduce the influence of label imbalance, including: resampling and class weighting.



## Dataset

Due to copyright restrictions, we do not provide the original data. However, we provide the data structure and class distribution in our experiment. As a result, you can replace the `dataset.py` file with your own data and change some necessary parameters in `train_classification.py` to run the code on your dataset.



## How to Run

To run the model, you can use the following command:

```bash
python train_classification.py --input_dim <input dim> --output_dim <class numbers> --model <please select from 'mlp', 'lstm', 'bert', and 'mamba'>
```

To change the model scale or other parameters related to training, please refer to the `get_args` function.

We provide the following methods for addressing label imbalance:

- `--class_weight`: You can specify the weight for each class in the loss function.

- `--data_balance`: We randomly delete 7/8 of the data in the '0' class to balance the dataset.



## Reference

```
@misc{zhao2025labelunbalancehighfrequencytrading,
      title={Label Unbalance in High-frequency Trading}, 
      author={Zijian Zhao and Xuming Zhang and Jiayu Wen and Mingwen Liu and Xiaoteng Ma},
      year={2025},
      eprint={2503.09988},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.09988}, 
}
```

