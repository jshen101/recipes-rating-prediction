# Recipes Rating Prediction
By Leica Shen, Zhirui Xia

## Framing the Problem
### Prediction Problem, Type, Response Variable
After completing [the analysis of recipes data](https://jshen101.github.io/recipes-analysis/), we found that there are some signs showing that recipes that have different values of some features may have different ratings. Thus, in this project, we would like to predict **whether the rating of a recipe will be 5 stars or not** based on other features in the dataset. This prediction problem is a **binary classification** problem. The response variable is binary, indicating whether the recipe will be **5 stars or not 5 stars**. We choose it to be the response variable because we believe that the rating of a recipe is a good indicator of how good the recipe is, and we want to predict whether the users will think a recipe is really good or not. Besides, as shown in our data analysis, we have imbalanced data, which means that the number of recipes that have 5 stars is much larger than the number of recipes that do not have 5 stars. Binary classification generally can handle imbalanced data well by distinguishing one specific class like 5-star-rating in this case from the rest. 

### Evaluation Metric
We first split our data in to a training set and a testing set using `train_test_split()` from `sklearn.model_selection`. We will only use the training set to train our model, then evaluate the model on the same testing set to compare the models performances. 

For the evaluation metric, we chose **weighted F1-score** over other suitable metrics like accuracy or F1-score because weighted F1-score is a good metric for imbalanced data. While F1-score is the harmonic mean of precision and recall, taking both precision and recall into account, giving a balanced evaluation of the model's performance on both classes, weighted F1-score calculates F1-score independently for each class and then averaging them based on class frequencies. It gives more weight to the class with more instances, providing a more accurate representation of overall model performance in imbalanced datasets. Since we want to avoid misclassification of both 5-stars and non-5-stars classes so that we won't miss any good recipes or recommend bad recipes to users, we think that weighted F1-score is a good metric to evaluate our model.

### Features at the Time of Prediction
By performing same data cleaning process as we did in the [recipes analysis](https://jshen101.github.io/recipes-analysis/), at the time of prediction, we would know many features about the recipes. The following features are the ones we will be using in our model training and prediction:
- `review`: Review text of the recipe from the users 
- `description`: User-provided description of the recipe
- `calories`: Nutrition information about the recipe's calories(#)
- `minutes`: Minutes to prepare recipe
- `submitted`: Date recipe was submitted

We decided to use the above features because each one of them seems to have some relationship with the rating of a recipe in common knowledge as people seems to be caring those information of a recipe/food. 



## Baseline Model
### Model Description
For the baseline model, we used **Logistic Regression** model. It is a statistical model used for binary classification, estimating the probability that an instance belongs to a particular class based on input features. It utilizes a logistic function to transform linear combinations of the input features and models the relationship between the input features and the binary outcome of 5-stars or not, making predictions in the form of probabilities. We chose this model because it is a relatively simple model that can be used as a good baseline for binary classification problems. 

### Features
We used **2** features in our baseline model: `minutes`, transformed by `StandardScaler()`; `calories`, transformed by `StandardScaler()`. These two features are both quantitative, we don't have any ordinal or nominal features in our baseline model. 

Based on our permutation test results in [recipe analysis](https://jshen101.github.io/recipes-analysis/), we rejected the null hypothesis of there is no significant difference in the average rating of recipes between different preparing time groups. Thus, we decided to include the `minutes` feature in our baseline model. Nowadays, people tend to care more about the calories of the food they eat, so we decided to also include the `calories` feature in our baseline model. 

For the necessary encodings part, since both features are numerical, we use `StandardScaler()` to transform both the `minutes` and `calories` features because they have different scales and we want to make sure that they are on the same scale so that our model can learn from them equally.

### Model Performance
We used the `LogisticRegression` class from `sklearn.linear_model` with the default hyperparameters to train the model. The model achieved an **weighted F1-score around 0.6717** on the test set. We do not believe that our current baseline model is "good" because by looking into the Classification Report generated by `classification_report()`, we can found that the baseline model predicts every recipes to be 5-stars. This might be a sign of our model being impacted by the imbalance of the data. The performance of our baseline is just better than random guessing which has a weighted F1-score around 0.5362, so it is not good enough model for us to use to predict whether the rating of a recipe will be 5-stars. 



## Final Model
### New Features
Besides the features we used in our baseline model, to improve our baseline model performance, we added 3 new features from the `review`, `description`, `submitted` features in the dataset to our model. 
- `review` and `description`: We used `TfidfVectorizer()` to extract the text information about the recipes from the `review` and `description` features and convert them into numerical features so that our model can capture semantic meaning and patterns in the review and description of recipes. This is good for the data and prediction task because review and description often contains valuable insights and sentiments that might correlate with user ratings. Taking these two text featurs into account would enable our model to learn from the language and wording used in reviews and descriptions, which could help our model to make better predictions.
  - From the perspective of the data generating process, we believe that these two features improved our model's performance because the review usually contains the users' opinions and attitudes toward the recipes so that knowing the review text could help our model make better predictions. The description of a recipe usually gives the users a first impression about the recipe, so the users final rating might be correlated with their impression of the recipes. This potential relationship makes us think that it might also improve our model's performance. 
- Sentiment Analysis: We created a pipeline in the ColumnTransformer that uses a customized `SentimentExtractor()`, which uses `TextBlob` from textblob to calculate sentiment scores and extract polarity of text, and `StandardScaler()` on the `review` feature of the recipes to incorporate sentiment analysis. This allows our model to captureing the emotional context / polarity of the reviews which might be correlated with the users' ratings. It is good for the data and prediction task because the sentiment of the reviews can be helpful in understanding the sentiment conveyed in the user reviews. 
  - From the perspective of the data generating process, we believe that this feature could improve our model's performance because the positive and negative sentiments expressed in the users' reviews might be strongly correlate with higher or lower ratings respectively since users rated 5-stars to the recipes usually will give a positive review, and vice versa.
- `submitted`: We created a customized `TimeFeaturesExtractor()` to transform the `submitted` feature which contains information about the date recipe was submitted into a new feature of the month of review submitted day, and a boolean value indicating whether the day is weekend. They are good for the data and prediction task because by accounting for the month, our model can capture seasonal variances and ingredient availability that may influence recipe ratings, and by identifying reviews made on weekends allows our model to consider factors such as increased cooking time, leisure activities, and social dining experiences that typically characterize these days. 
  - From the perspective of the data generating process, we believe that this feature could improve our model's performance because we believe month could affect the rating as certain recipes may be better received during specific seasons due to the availability of fresh, seasonal ingredients that enhance the flavor of the dish. Furthermore, people generally have more free time on weekends to try new recipes, which can lead to a more relaxed and enjoyable cooking experience, possibly resulting in higher ratings.

### Model Description
Our final model is the Logistic Regression model. As described in the baseline model section, Logistic Regression is a statistical model used for binary classification, estimating the probability that an instance belongs to a particular class based on input features. It utilizes a logistic function to transform linear combinations of the input features and models the relationship between the input features and the binary outcome of 5-stars or not, making predictions in the form of probabilities. 

We firstly tried several models like `RandomForestClassifier`, `DecisionTreeClassifier`, and `LogisticRegression` on the same training and testing data set. We used `GridSearchCV()` to find the best hyperparameters for each of the models. After comparing the performance of these models with their best hyperparameters on the test set, we found that Logistic Regression has the best performance. We found that the Random Forest Classifier model might have encountered some overfitting issue that its performance on the test set is much worse comparing to its performance on the training set, and Decision Tree Classifier model did not perform well on the test set either. 

The hyperparameters ended up performing the best are: {'classifier': LogisticRegression(C=1, class_weight='balanced', max_iter=1000, solver='liblinear'), 'classifier__C': 1}. We used 5-fold cross-validation and let the `scoring='f1_weighted'` in the `GridSearchCV()` to find this best hyperparameters that can maximize the weighted F1-score of our Logistic Regression model. We tuned the `C` hyperparameter of Logistic Regression model by using a param_grid with `classifier__C` hyperparameter set to [0.1, 1, 5, 10, 50], and use this param_grid in grid search. We also set `class_weight='balanced'` to account for the imbalance of the data.
  
### Final Model Performance
Our final model using all the features and hyperparameters shown above achieved an weighted F1-score of around **0.81**. Compared to the baseline model which had a weighted F1-score of around 0.67, our final model has a much better performance. By looking into the Classification Report generated by `classification_report()`, we can found that our final model predicts both 5-stars and non-5-stars recipes, which is a sign that our model did a better job on the imbalance of the data than the baseline. Thus, we believe that our final model's performance is an improvement over our baseline model's performance. Here is a confusion matrix of our final model's performance on the test set:
<iframe src="final_model_cm.html" width=800 height=600 frameBorder=0></iframe>

From this figure, we can see that our final model predicts 27508 True Positives (5-stars), 7569 True Negatives (Non-5-stars), 6311 False Negatives, and 2459 False Positives. This also shows that our final model is an improvement over our Baseline Model’s performance. 


## Fairness Analysis
To perform a "fairness analysis" of our Final Model, we tried to use permutation test to answer the question of "does our final model perform worse for individuals who have longer preparing minutes (more than median time) than it does for individuals who have shorter preparing minutes (less than median time)?" 

- Two groups in this case:
  - Group X: Individuals who have longer preparing minutes (more than median time)
  - Group Y: Individuals who have shorter preparing minutes (less than median time)

- Evaluation Metric: We decided to use f1-score as our evaluation metric, because we care equally about our model's precision and recall. We want to make sure that our model can predict both 5-stars and non-5-stars recipes well in both groups, so we choose f1-score, a single metric which combines both precision and recall. 

- Null Hypothesis: Our final model is fair. Its f1-score for individuals who have longer preparing minutes and who have shorter preparing minutes are roughly the same, and any differences are due to random chance.

- Alternative Hypothesis: Our final model is unfair. Its f1-score for individuals who have longer preparing minutes is lower than its f1-score for individuals who have shorter preparing minutes.

- Choice of Test Statistic: Difference in f1-score between individuals who have longer preparing minutes and individuals who have shorter preparing minutes. More specifically, our test statistic is f1-score of group with longer preparing minutes minus f1-score of group with shorter preparing minutes.

- Significance Level: a = 0.05. We choose this significance level because it is a common used significance level in hypothesis testing.

After doing the permutation test, we have the resulting p-value being 0.1517. Since our p-value=0.1517 > a=0.05, we fail to reject the null hypothesis of our final model being fair. Here is a visualization of the permutation test:
<iframe src="fairness_analysis.html" width=800 height=600 frameBorder=0></iframe>
