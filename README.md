# final_project_MLP

This repository contains our final project submission. The project focuses on applying *transfer learning by fine-tuning the EfficientNet-B0 model on the CIFAR-10 dataset using PyTorch.

##  Paper Reference
- Title: EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks  
- Authors: Mingxing Tan & Quoc V. Le (Google AI)  
- Paper Link:(https://arxiv.org/abs/1905.11946)  
- Original Dataset Used: ImageNet (https://paperswithcode.com/dataset/imagenet) 
- Project Dataset Used: CIFAR-10  (https://www.cs.toronto.edu/~kriz/cifar.html)

##  How We Recreated the Results
- We used PyTorch to load the pretrained EfficientNet-B0 model from `torchvision.models`.
- The model was evaluated on a new dataset (CIFAR-10) without training to benchmark its baseline performance.
- We replaced the final classification layer to match CIFAR-10’s 10 output classes.


##  Addressing the Project Requirements

### How We Covered Point 1:
- Used CIFAR-10, a low-resolution dataset that differs from ImageNet, to evaluate model generalization on unseen data.

###  How We Covered Point 2:
- Explored two options:
  1. Freezing pretrained layers and only modifying the final classifier.
  2. (Chosen) Fine-tuning all layers to allow the model to learn new patterns.
- We selected option 2 and trained the entire model end-to-end.

##  What We Learned
- Learned how pretrained benchmark models can be adapted to new tasks through transfer learning.
- Understood how fine-tuning retains old representations while adapting to new datasets.
- Gained insights into performance benchmarking and the importance of tracking training progress.
- Practiced loss visualization and accuracy monitoring.

## Project Summary

- We loaded EfficientNet-B0 pretrained on ImageNet.
- Replaced the classifier to output 10 classes for CIFAR-10.
- Evaluated performance before training (~9.89% accuracy).
- Fine-tuned the entire model for 25 epochs using `Adam` optimizer.
- Monitored training loss and test accuracy over time.
- Final accuracy reached (~86.52% )— a major improvement.
- Planned to repeat the process using EfficientNet-B1 for future extension.

##  Tools & Libraries Used

- Python 3.8+
- PyTorch
- Torchvision
- Matplotlib
- Google Colab

- ##  Limitations

- CIFAR-10 resolution (32x32) is much smaller than what EfficientNet is optimized for (224x224).
- Only one variant (B0) was tested; B1–B7 may improve accuracy further.
  
##  Future Work
- Fine-tune EfficientNet-B1 for better performance.
- Apply same approach to other datasets like CIFAR-100 or SVHN.
- Try freezing partial layers and compare with full fine-tuning.
- Deploy model using TorchScript or ONNX.

##  github Project Structure

1) Datasets Folder: 
Inside this folder, we have added a text/markdown file that contains the links to both datasets:
- ImageNet (used in the original EfficientNet paper)
- CIFAR-10 (used in our project implementation and training)
2)projectNotebook folder:
  Indside this folder,we have added two files;
  1)implementation_paper.ipyb:
  - Loads the EfficientNet-B0 model pretrained on ImageNet.
- Modifies the final classification layer to fit 10 classes (CIFAR-10).
- Evaluates model before any training or fine-tuning.
- Purpose: To replicate and benchmark the pretrained model as a baseline.
   2)contribution_code.ipynb:
  Fine-tunes the entire EfficientNet-B0 model on CIFAR-10 for 25 epochs.
- Tracks training loss and test accuracy.
- Plots training curves.
- Evaluates model performance after training.
- Purpose: To apply transfer learning and improve accuracy significantly (~86.52%).
   3)final_project_MLP.ipynb:which include all the details,code and explanation of project.
3)requirement.txt:To run this project, we will need the following Python packages. which are listed in the `requirements.txt` file for easy setup.

  # Project Goals
- Reproduce the original EfficientNet-B0 model's baseline performance on a new dataset (CIFAR-10) to establish a benchmark.
- Improve the model by applying full fine-tuning and monitoring performance through training metrics.
- Differentiate between implementation (paper reproduction) and contribution (our enhancements) to showcase clear understanding and original work.
