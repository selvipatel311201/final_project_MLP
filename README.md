# final_project_MLP
Name of group member:Selvi Patel(200590923)
                     Muskan sharma(200596320)
                     
**transfer learning by fine-tuning the EfficientNet-B0 model**
This repository contains our final project submission. The project focuses on applying *transfer learning by fine-tuning the EfficientNet-B0 model on the CIFAR-10 dataset using PyTorch.

#  Introduction
Transfer learning has become a widely adopted technique in deep learning, especially for image classification tasks.  
This project applies transfer learning using the EfficientNet-B0 architecture — a model originally trained on the large-scale ImageNet dataset — and adapts it to the smaller, low-resolution CIFAR-10 dataset.

The goal is to investigate how well pretrained features generalize and how much performance improvement can be achieved through fine-tuning.

# Paper Reference
- Title: EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks  
- Authors: Mingxing Tan & Quoc V. Le (Google AI)  
- Paper Link:(https://arxiv.org/abs/1905.11946)  
- Original Dataset Used: ImageNet (https://paperswithcode.com/dataset/imagenet) 
- Project Dataset Used: CIFAR-10  (https://www.cs.toronto.edu/~kriz/cifar.html)

#  How We Recreated the Results
- Used PyTorch to load the pretrained EfficientNet-B0 model from `torchvision.models`.
- Replaced the final classification layer to output 10 classes (matching CIFAR-10).
- Evaluated the model on CIFAR-10 without training to establish a baseline.
- Fine-tuned the entire model end-to-end to improve performance.

# Addressing the Project Requirements:

# How We Covered Point 1:

> *"You will need to test the methodology of the selected research paper on new datasets to evaluate its effectiveness in different contexts."*

To fulfill this requirement, we selected CIFAR-10 — a completely different dataset from the original ImageNet used in the EfficientNet paper.  
CIFAR-10 consists of low-resolution (32×32) images across 10 classes, whereas ImageNet contains high-resolution (224×224+) images across 1,000 classes.

By training and evaluating the EfficientNet-B0 model on CIFAR-10, we were able to test how well the methodology generalizes to new data distributions and resource-constrained scenarios.

This allowed us to validate the model’s transferability and real-world adaptability to smaller, domain-specific datasets.


#  How We Covered Point 2:

> *"You should experiment with changing some of the model parameters to create an upgraded version of the existing methodology. This will allow you to explore the impact of different parameter values on the model’s performance."*

To address this point, we experimented with the model’s trainable components and training strategy.

We explored two different approaches:

1. Freezing the pretrained feature extractor layers and training only the new classifier head.
2. Fine-tuning the entire model — allowing all layers to update based on CIFAR-10 data.

We selected the full fine-tuning approach, which enabled the model to adjust both the learned features and the final classification layer. This change allowed EfficientNet-B0 to learn CIFAR-10-specific patterns, improving adaptability and performance.

By modifying which layers are trainable and observing the impact on performance, we effectively explored the effect of hanging model parameters as required in Point 2.

# Project Goals:

- Reproduce the original EfficientNet-B0 model's baseline performance on a new dataset (CIFAR-10) to establish a benchmark.
- Improve the model by applying full fine-tuning and monitoring performance through training metrics.
- Differentiate between implementation (paper reproduction) and contribution (our enhancements) to showcase clear understanding and original work.

# What We Are Doing

- We started by loading EfficientNet-B0 (and plan to explore EfficientNet-B1) pretrained on ImageNet.
- The initial performance of the model on CIFAR-10 was below 10% accuracy, due to the domain gap between datasets.
- We replaced the final classification layer to predict 10 classes instead of 1,000 (ImageNet).
- We fine-tuned the entire model (not just the classifier) using CIFAR-10 for 25 epochs.
- We tracked training loss and test accuracy after each epoch to monitor learning progress.
- After fine-tuning, the model achieved a test accuracy of ~86.52% — a significant performance jump.
- We plan to apply the same strategy to EfficientNet-B1 for further evaluation.

# Implementation vs Contribution

#Paper Implementation (`implementation_paper.ipynb`)
- Loads EfficientNet-B0 with ImageNet weights
- Modifies final classifier to output 10 classes
- Evaluates performance without any training (Baseline)

#My Contribution (`contribution_code.ipynb`)
- Fine-tunes the full model on CIFAR-10 for 25 epochs
- Uses Adam optimizer, CrossEntropyLoss, and batch size of 64
- Tracks and plots training loss and test accuracy
- Evaluates model after training (~86.52% accuracy)

# Baseline vs Fine-Tuned Accuracy

| Model                         | Accuracy (%) |
|------------------------------|--------------|
| Pretrained (no training)     | 9.89         |
| After Fine-Tuning (25 Epochs)| 86.52        |

#What We Learned

- How pretrained benchmark models like EfficientNet can be repurposed for new tasks through fine-tuning.
- The power of transfer learning to retain old representations while adapting to new datasets.
- The importance of benchmarking and tracking metrics such as accuracy and loss.
- How to efficiently reuse pretrained models for small datasets with limited compute.


# Project Summary


- Loaded EfficientNet-B0 pretrained on ImageNet.
- Modified the final classifier to output 10 classes (CIFAR-10).
- Evaluated baseline performance (accuracy: ~9.89%).
- Fine-tuned the entire model for 25 epochs using the Adam optimizer.
- Tracked loss and accuracy during training.
- Final test accuracy: **~86.52%**
- Future extension planned using EfficientNet-B1.


#  Tools & Libraries Used

- Python 3.8+
- PyTorch
- Torchvision
- Matplotlib
- Google Colab


 #  Limitations

- CIFAR-10 images (32x32) are smaller than the expected input size for EfficientNet (224x224).
- Only EfficientNet-B0 was evaluated; other variants like B1–B7 may offer improved results.
- The experiment was limited to a single dataset.
  
# Future Work

- Fine-tune deeper EfficientNet variants like B1 or B2.
- Explore additional datasets like CIFAR-100 or SVHN for broader validation.
- Experiment with freezing layers to reduce training time.
- Convert the trained model to ONNX or deploy using EfficientNet-Lite.

# github Project Structure

# 1.datasets
- Contains a `.txt` file linking to both datasets:
  - ImageNet (used in the original paper)
  - CIFAR-10 (used in this project)

# 2️. projectNotebook
- `implementation_paper.ipynb`  
  - Loads EfficientNet-B0 pretrained on ImageNet  
  - Replaces the classifier layer with 10 output classes  
  - Evaluates model before training (baseline)

- `contribution_code.ipynb`  
  - Fine-tunes the full EfficientNet-B0 on CIFAR-10 for 25 epochs  
  - Tracks training loss and test accuracy  
  - Plots training curves  
  - Evaluates post-training performance (~86.52%)

- `AIDI_1002_final_project.ipynb`  
  - Combines both implementation and contribution, with detailed explanations and results

# 3️.`requirements.txt`  
- Contains all necessary Python packages  
- Install using:
```bash
pip install -r requirements.txt


# Conclusion:

This project successfully demonstrated the power of transfer learning by applying EfficientNet-B0 — originally trained on ImageNet — to a completely different dataset (CIFAR-10).

We first established a baseline by evaluating the pretrained model without any training, which resulted in low accuracy (~9.89%) due to domain differences.  
By fine-tuning the entire model on CIFAR-10 for 25 epochs, we significantly improved performance, achieving an accuracy of ~86.52%.

Our approach highlights the value of:
- Reusing pretrained models for new tasks
- Adapting model architecture for dataset compatibility
- Fine-tuning strategies in low-resolution, small-dataset scenarios

The success of this transfer learning process reinforces the adaptability of EfficientNet and opens the door for future experimentation on deeper variants like EfficientNet-B1, other image classification datasets, and deployment in real-world applications.
> The methodology followed here not only meets the academic project objectives but also provides a practical foundation for real-world deep learning solutions.
