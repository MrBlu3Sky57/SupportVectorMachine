# Support Vector Machines and Natural Language Processing for Spam versus Ham Classification
This an implementation of multiple types of Support Vector Machines (SVM) for classifying spam versus ham. For the models, I used the enron1 dataset .
## Table of Contents
- [Plan](#Plan)
- [P1](#p1)
- [P2](#p2)
- [Next Steps](#NextSteps)
- [License](#license)

## Plan
I am building this project for educational purposes, so I will build multiple SVMs and work my way from the most naive (Hard-Margin) to more sophisticated methods
like a RBF kernel SVM. I chose to apply SVMS to Email spam classification because I wanted to learn some natural language processing on top of supervised learning. I plan on using this project to also help me refine my skills in natural language processing. 
  
More concretely my plan is as follows:

Part          | Description   | Status
------------- | ------------- | -----------
1             | Hard Margin SVM            | DONE
2             | Soft Margin SVM | DONE
3             | RBF kernel SVM                   | TODO

For Natural Language Processing I started with a Bag of Words model and will integrate different types of NLP models to see which one is best suited for this learning task.  
  
I will not use any libraries other than Numpy and built-in libraries for the construction of the models as I want to learn to build everything from scratch. I will supplement each model (built in a Jupyter notebook) with a LaTeX write up that goes through the mathematical details of the model construction.

## P1
My first model is a hard margin support vector machine implementing using gradient descent. 

I used a Bag of Words model to embed each email datapoint as a feature vector. The frequency of certain words like "The" or "a" made my optimization step a bit chaotic, so I decided to forego storing the exact counts of each word and instead track the presence or absence of the word in the email, which greatly improved the optimization. I go through the derivations for the optimization step in the LaTeX, but in essence I used the method of Lagrange multipliers to maximize the margin width of the SVM under the prediction constraint. Gradient descent was used to approximate the multiplier vector L then the parameters of the prediction function were recovered from L.

From here, I made predictions where Ham had a target of 1 and Spam had a target of -1. I applied lossy compression on the data to visualize the SVM's classification:

![PCA/TSNE SVM PLOT](figures/SVM1_preds.png)

This can be compared to the plot of the Emails and their actual labels:

![PCA/TSNE SVM PLOT](figures/SVM1_actual.png)

While there is some clear discrepancies in the plots, for a naive classifaction algorithm I am happy with the result. However, based on more detailed analysis, the model had 84% accuracy on the test dataset with a slight tendency towards false positives. Thus, there is a clear direction for improvement.

## P2
My second model was a soft margin support vector machine. I used a TF-IDF model to embed each email datapoint as a feature vector. This way the frequency of words in a document as well as across documents was factored into the embedding process. The implementation was largely similar to the Hard Margin one, but the addition of a hyper parameter C which controlled how much "misclassification" was allowed, gave me the oppurtinity to tune it with the dev dataset. I will not include a new mathematical write up as almost all the mathematical details remain unchanged. 

Here are the resulting predictions of my model, the datapoints are arranged differently because of the different NLP model used for embedding.

![PCA/TSNE SVM PLOT](figures/SVM2_preds.png)

This can be compared to the plot of the Emails and their actual labels:

![PCA/TSNE SVM PLOT](figures/SVM2_actual.png)

The use of PCA and TSNE reduces the distinction between data so this plot does not look as nice as the one in part 1. However, my model performed quite well with a 94.8% accuracy and only a small bias towards predicting ham as spam, which is not a major issue. Now I will add a kernel function, which will hopefully further increase this accuracy.

## License
This project is licensed under the [MIT License](LICENSE).
s