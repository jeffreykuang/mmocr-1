# Show, Attend and Read: A Simple and Strong Baseline for Irregular Text Recognition

## Introduction

[ALGORITHM]

```bibtex
@inproceedings{li2019show,
  title={Show, attend and read: A simple and strong baseline for irregular text recognition},
  author={Li, Hui and Wang, Peng and Shen, Chunhua and Zhang, Guyu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  number={01},
  pages={8610--8617},
  year={2019}
}
```

## Dataset

### Train Dataset

|  trainset  | instance_num | repeat_num |          source          |
| :--------: | :----------: | :--------: | :----------------------: |
| icdar_2011 |     3567     |     20     |           real           |
| icdar_2013 |     848      |     20     |           real           |
| icdar2015  |     4468     |     20     |           real           |
| coco_text  |    42142     |     20     |           real           |
|   IIIT5K   |     2000     |     20     |           real           |
| SynthText  |   2400000    |     1      |          synth           |
|  SynthAdd  |   1216889    |     1      | synth, 1.6m in [[1]](#1) |
|   Syn90k   |   2400000    |     1      |          synth           |

### Test Dataset

| testset | instance_num |            type             |
| :-----: | :----------: | :-------------------------: |
| IIIT5K  |     3000     |           regular           |
|   SVT   |     647      |           regular           |
|  IC13   |     1015     |           regular           |
|  IC15   |     2077     |          irregular          |
|  SVTP   |     645      | irregular, 639 in [[1]](#1) |
|  CT80   |     288      |          irregular          |

## Results and Models

|                               Methods                               |  Backbone   |       Decoder        |        | Regular Text |      |     |      | Irregular Text |      |                                                                                              download                                                                                              |
| :-----------------------------------------------------------------: | :---------: | :------------------: | :----: | :----------: | :--: | :-: | :--: | :------------: | :--: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                     |             |                      | IIIT5K |     SVT      | IC13 |     | IC15 |      SVTP      | CT80 |
| [SAR](/configs/textrecog/sar/sar_r31_parallel_decoder_academic.py)  | R31-1/8-1/4 |  ParallelSARDecoder  |  95.0  |     89.6     | 93.7 |     | 79.0 |      82.2      | 88.9 |  [model](https://download.openmmlab.com/mmocr/textrecog/sar/sar_r31_parallel_decoder_academic-dba3a4a3.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/sar/20210327_154129.log.json)  |
| [SAR](configs/textrecog/sar/sar_r31_sequential_decoder_academic.py) | R31-1/8-1/4 | SequentialSARDecoder |  95.2  |     88.7     | 92.4 |     | 78.2 |      81.9      | 89.6 | [model](https://download.openmmlab.com/mmocr/textrecog/sar/sar_r31_sequential_decoder_academic-d06c9a8e.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/sar/20210330_105728.log.json) |

**Notes:**

-   `R31-1/8-1/4` means the height of feature from backbone is 1/8 of input image, where 1/4 for width.
-   We did not use beam search during decoding.
-   We implemented two kinds of decoder. Namely, `ParallelSARDecoder` and `SequentialSARDecoder`.
    -   `ParallelSARDecoder`: Parallel decoding during training with `LSTM` layer. It would be faster.
    -   `SequentialSARDecoder`: Sequential Decoding during training with `LSTMCell`. It would be easier to understand.
-   For train dataset.
    -   We did not construct distinct data groups (20 groups in [[1]](#1)) to train the model group-by-group since it would render model training too complicated.
    -   Instead, we randomly selected `2.4m` patches from `Syn90k`, `2.4m` from `SynthText` and `1.2m` from `SynthAdd`, and grouped all data together. See [config](https://download.openmmlab.com/mmocr/textrecog/sar/sar_r31_academic.py) for details.
-   We used 48 GPUs with `total_batch_size = 64 * 48` in the experiment above to speedup training, while keeping the `initial lr = 1e-3` unchanged.

## References

<a id="1">[1]</a> Li, Hui and Wang, Peng and Shen, Chunhua and Zhang, Guyu. Show, attend and read: A simple and strong baseline for irregular text recognition. In AAAI 2019.
