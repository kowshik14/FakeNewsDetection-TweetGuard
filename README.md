# TweetGuard: Combining Transformer and Bi-LSTM Architectures for Fake News Detection in Large-Scale Tweets

[![DOI](https://img.shields.io/badge/DOI-10.11648/j.ijdsa.20251102.12-blue.svg)](https://www.sciencepg.com/article/10.11648/j.ijdsa.20251102.12)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

    

> 📂 **Source code is [here](https://github.com/kowshik14/FakeNewsDetection-TweetGuard/tree/main/src).**  
> **✨ Note:** This project is part of my **🎓 M.Sc. Thesis dissertation**


## Abstract

The proliferation of misinformation on platforms like Twitter, where rapid dissemination can significantly impact public discourse, underscores the urgent need for effective automated fake news detection systems. These systems are crucial in preventing the spread of falsehoods and maintaining informational integrity. Traditionally, one of the challenges in developing such systems has been the lack of comprehensive benchmark datasets, which are essential for reliably training and testing detection models. Additionally, the rapid evolution of deceptive tactics makes traditional methods less effective, necessitating new approaches that can adapt to emerging misinformation patterns. In response to the challenges, a robust model named "TweetGuard" has developed, leveraging the 'TruthSeeker' dataset, a recently published benchmark offering a rich collection of annotated tweets. This dataset provides a solid foundation for training and refining our detection techniques. The proposed model employs a novel classification architecture that integrates transformer and Bi-LSTM technologies in a concatenation mode, enhanced by advanced preprocessing steps, including BERTweet, for effective tokenization and contextual understanding. An ablation study highlights the individual contributions of the Bi-LSTM and Transformer components, as well as their combined effect, demonstrating their critical roles in enhancing the model's performance. Compared to conventional classifiers, including various CNN, LSTM, Bi-LSTM, BERT and Transformer configurations, the proposed model demonstrates superior performance, as evidenced by comprehensive statistical testing. TweetGuard achieves an accuracy of 94.02%, an F1-score of 93.84%, and a ROC-AUC score of 0.9614 on the TruthSeeker dataset. Additional metrics, such as a Matthews Correlation Coefficient (MCC) of 0.8802 and a fake news detection rate of 93.70%, also demonstrate the model's stability and robustness. Its effectiveness and generalizability are further validated through rigorous testing across three additional fake news datasets, confirming its reliability and adaptability in diverse informational settings. This evaluation not only highlights our model's superior ability to identify and classify misinformation accurately but also establishes a new benchmark for automated fake news detection on social media platforms.

### 🚀 Key Features of TweetGuard

- **🛡️ Hybrid Architecture**: Combines Transformer and Bi-LSTM architectures to maximize the strengths of both for more accurate fake news detection.
  
- **🧠 Advanced Tokenization**: Utilizes **BERTweet** tokenization, which enhances the model's ability to understand context and detect nuanced misinformation in tweets.

- **🔄 Ablation Study**: Comprehensive analysis of individual and combined contributions of the Bi-LSTM and Transformer components to show performance improvement.

- **📊 High Accuracy**: Achieves state-of-the-art results on multiple datasets with superior performance compared to existing models in the field.

- **📈 Benchmarking**: Extensive comparative analysis against traditional models like CNN, LSTM, and Transformer architectures to highlight superior performance.

- **🗂️ Robust Text Preprocessing**: Incorporates a powerful text-cleaning and standardization pipeline to prepare Twitter data for effective classification.

- **📉 Real-time Detection**: Capable of detecting fake news in real-time across a variety of informational settings.

- **📁 Cross-Dataset Validation**: Demonstrated high adaptability and robustness by testing across multiple fake news detection datasets, ensuring its reliability in diverse scenarios.



## Dataset Link 🗂️
[![CIC TruthSeeker2023 Dataset](https://img.shields.io/badge/CIC%20TruthSeeker%20Dataset-2023-blue)](https://www.unb.ca/cic/datasets/truthseeker-2023.html)

## Citation

If you find this work useful, please cite our paper:  
Kowshik Sankar Roy and Farhana Aketer Bina, "_TweetGuard: Combining Transformer and Bi-LSTM Architectures for Fake News Detection in Large-Scale Tweets_", International Journal of Data Science and Analysis, Vol.11, No.2, pp.23-45, 2025.

## Installation

To install, run the following command:

```bash
git clone https://github.com/kowshik14/FakeNewsDetection-TweetGuard
