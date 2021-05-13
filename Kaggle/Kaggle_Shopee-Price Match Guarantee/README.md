# Kaggle Shopee - Price match Guarantee - 115th place solution

## Training
### Image
* Augment - Random Flip (left-right and up-down), Random Hue, Random Saturation, Random contrast, Random brightness
* LR Scheduling - Warmup and decay
* Train EfficientNet B0, B1, B2, B3, B5 with pretrained noisy-student weights
* Use ArcMarginProduct distance to train the model - helps to minimise the intra-cluster distance of embeddings and maximise inter-cluster distance
* Adam Optimizer 
* Loss function - Sparse categorical Cross-Entropy
* Metric - Sparse Categorical Accuracy metric
* Train on complete data
* Trained on TPU

### Text
* Use BERT - bert-base-uncased (max len 70, 100, 128)
* BERT tokenize
* Take BERT embeddings of CLS token as the embedding of title
* Pass through ArcMarginProduct distance to train model
* Loss function - Sparse categorical Cross-Entropy
* Metric - Sparse Categorical Accuracy metric
* Train on complete data
* Train on GPU

## Inference
* Load weights into the Image and Text model
* Take embeddings up to Pooling layer in the Image model and up to CLS token embedding in Text model
* Concat the embeddings
    * For Image model - EffNet B0 + B1 + B2 + B3 + B5
    * For Text model - BERT max len 70 + 100 + 128
* Predict in batches - Memory constraints (took num neighbors = 100)
* Use KNN to get nearest neighbours separately for images and text
* Use thresholds to get indices
* Use unsupervised method - TFIDF to get embeddings to generate embeddings of size 25,000
* Use cosine similarity to find close products, with 0.75 threshold
* Combine the predictions from each model to get the final set of predictions

**Public LB**: 0.737  
**Private LB**: 0.728  

**Rank** - 115/2426 teams (Solo participation)