# Machine Learning
## 1. Decision Tree
- Load the datasets using **Python** and impute missing values using the mode.
- Implement **Decision Stump**, **Uunpruned Decision Tree** and **Pruned Decision Tree** from scratch.
- Stratified **5-Fold Cross-Validation** is used to evaluate the performance of different hyperparameter combinations. The training set was divided into 5 subsets, while ensuring that the class distribution in each subset remains consistent with the overall dataset. For each set of **hyperparameters** (including tree depth, minimum samples per leaf, and minimum information gain), the model is trained on 4 folds and validated on the remaining fold. The **F1 scores** from all folds are averaged to serve as the final evaluation metric for that set of hyperparameters. The code iterates over a predefined grid of hyperparameters, calculates the average F1 score for each combination, and selects the hyperparameter combination that performs the best.
- Compare the three models on each dataset.
- Report the **p-value** and discuss whether performance differences are statistically significant.
- Analyze the importance of features in the trained decision tree models.
- Find the most important features (top 3) that contribute the most to the classification for each dataset.
- Measure the runtime performance of each decision tree model.
- Compare training time across datasets and models.
- Discuss trade-offs between accuracy and computational efficiency.

## 2. Naive Bayes classifier 
- Implement a Multinomial Naive Bayes classifier from scratch using user review texts as input.
- Construct a **bag‑of‑words count matrix**.
- Build a baseline model by computing each class’s prior probabilities and each term’s conditional probabilities with **Laplace smoothing** and **logarithmic scaling**.
- On the test set, this model achieved an **accuracy** of 0.8963 (about 0.90 overall); specifically, for the Nightlife category **precision**, **recall**, and **F1‑score** were all 0.61, for Restaurants they were 0.92/0.93/0.93, and for Shopping they were 0.93/0.92/0.92, yielding a weighted average accuracy of 0.90.
- Replace raw term counts with **TF‑IDF** features and switched to the **ComplementNB** classifier to correct Naive Bayes’s weaknesses under class imbalance and extreme word‑frequency distributions.
- Using stratified **5‑fold cross‑validation** and **grid search** to tune sublinear_tf, use_idf, norm, min_df, ngram_range, and the smoothing parameter alpha, the model reached a best CV accuracy of 0.8867 and improved to 0.9048 on the test set. In that configuration, Nightlife achieved precision/recall/F1 of 0.71/0.64/0.67, Restaurants 0.94/0.93/0.93, and Shopping 0.90/0.94/0.92.
- Introduce the “name” field as an extra textual feature and use a ColumnTransformer to apply customized TF‑IDF separately to reviews and names.
- By merging these two feature streams in a single Pipeline and jointly grid‑searching both streams’ TF‑IDF parameters along with the classifier’s smoothing parameter, the multi‑attribute fusion model achieved a CV accuracy of 0.9023 and a test accuracy of 0.9162. In that final model, Nightlife’s precision/recall/F1 rose to 0.74/0.66/0.69, Restaurants to 0.94/0.94/0.94, and Shopping to 0.91/0.95/0.93—demonstrating that each iterative enhancement steadily increased the model’s predictive accuracy.

## 3. SVM
- Train an SVM with a **linear kernel** on DS1, and plot the data set with the **decision boundary**.
- Carry out a **leave-1-out cross-validation** with an SVM on your dataset.
- Report the train and test performance.
- Improve the SVM by changing **C**.
- Plot the data set and resulting decision boundary, give the performance.
- A larger C makes the model care more about correctly classifying training samples, even if it sacrifices margin width; a smaller C prefers a wider margin and tolerates more errors to avoid overfitting. 
- In DS2, the positive and negative samples form multiple disconnected clusters in the original space, which a linear kernel cannot separate. The **RBF kernel** measures Gaussian similarity between samples and implicitly maps the data into a high-dimensional space, where these local clusters become linearly separable, enabling the model to learn a decision surface that matches DS2’s complex boundary.
- With a linear kernel, the model performs poorly on DS2: both train and test accuracy are 0.5740, indicating that a single hyperplane cannot separate those disconnected clusters; using the RBF kernel, train accuracy rises to 0.8700 and test accuracy to 0.8540, a substantial improvement.
- In DS3, performe Leave-One-Out cross-validation over the grid of C values (13, 15, 17) and γ values (1, 3, 5), training and evaluating an RBF‐kernel SVM for each parameter pair and recording each pair’s **LOOCV** accuracy.
- Retrain the model on the entire dataset using the optimal parameters (C = 13, γ = 1) and computed its training accuracy.
- Plotted the decision boundary alongside the data points. 
