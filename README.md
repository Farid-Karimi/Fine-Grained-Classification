# Sentiment Analysis for Fine-Grained Classification

## Abstract

Sentiment analysis on movie reviews is crucial for understanding audience preferences. This study compares traditional machine learning (ML) and modern deep learning (DL) models for fine-grained sentiment classification using the SST-5 dataset of movie reviews with five sentiment classes. Logistic Regression, Random Forest, Support Vector Machine, a simple neural network, and LSTM achieved 36-40% accuracy. In contrast, pretrained transformer models DistilBERT and BERT outperformed with 51% and 53% accuracy, respectively. Oversampling decreased performance due to overfitting and noise amplification. The results highlight the effectiveness of pretrained DL models but also the challenges posed by frequent neutral texts and annotator bias in SST-5.

## Introduction

Fine-grained sentiment analysis allows for more precision and nuance in understanding human language compared to binary classification. However, it is more complex due to factors like sentence structure and dataset balance. The Stanford Sentiment Tree (SST-5) dataset was developed to address shortcomings of traditional datasets by incorporating a tree structure onto sentences and labeling phrases on a 1-5 sentiment scale.

## Methodology 

We explored various machine learning and deep learning models on the SST-5 dataset, including Logistic Regression, Random Forest, Support Vector Machine, a simple neural network, LSTM, DistilBERT, and BERT. Preprocessing steps involved data splitting, tokenization, lemmatization, stop-word removal, and TF-IDF vectorization.

## Results
![Untitled](https://github.com/user-attachments/assets/ce3453fd-ca38-461e-896a-2b81d646622b)


## Discussion

Pretrained transformer models like BERT and DistilBERT performed significantly better than traditional models, attributed to their ability to capture contextual and semantic information effectively. However, the SST-5 dataset posed challenges like frequent neutral short texts and potential annotator bias, impacting model efficacy. Oversampling also led to overfitting and amplification of noise in minority classes.

## Conclusion  

Modern deep learning models, especially pretrained transformer architectures like BERT, outperformed traditional approaches for fine-grained sentiment classification on SST-5. However, dataset-specific challenges highlight the need for further optimization and research to address complexities in fine-grained sentiment analysis.

## Teammate and Project Information

Teammate: [Ali KhosraviPour](https://www.linkedin.com/in/alikhosravipour)

The project was part of the Neuromatch Academy in Deep Learning course, supervised by our Project TA [Joseph AKINYEMI](https://www.linkedin.com/in/joseph-akinyemi-66ab6481/). The goal was to compare the performance of traditional and modern sentiment analysis models on a fine-grained dataset and analyze the impact of various challenges inherent in such tasks.

## References

1. [**Cheang, B., Wei, B., Kogan, D., Qiu, H., & Ahmed, M. (2020).** "Language Representation Models for Fine-Grained Sentiment Classification." *Cornell Tech, New York, NY.*](https://arxiv.org/pdf/2005.13619)

2. [**Socher, R., Perelygin, A., Wu, J. Y., Chuang, J., Manning, C. D., Ng, A. Y., & Potts, C. (2013).** "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank." *Stanford University, Stanford, CA.*](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)
