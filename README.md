# A&O SCI C111: Obesity Level Prediction

## Project Overview

The primary objective of this project was to employ a classification model to predict obesity levels based on a variety of features. This report will detail the process of preparing the data, selecting the model, and the rationale behind these choices, providing a comprehensive understanding of each step in the project's development.

### Background

Obesity is a global health issue that has seen a dramatic increase in prevalence over the past few decades. It is a risk factor for numerous health conditions, including heart disease, diabetes, and hypertension. Understanding the determinants of obesity and accurately predicting it is crucial for public health interventions. This study aims to leverage machine learning techniques to analyze and predict obesity levels based on individual data, finding the factors that affect it the most. This project aligns with the growing interest in the use of data science and machine learning in healthcare to enhance preventive medicine and public health strategies.

### The Dataset

The dataset, titled 'Estimation of obesity levels based on eating habits and physical condition', comprises 2111 instances and 17 features. It was specifically gathered to estimate obesity levels in individuals from Mexico, Peru, and Colombia, focusing on their eating habits and physical condition. The features include fundamental attributes such as Gender, Age, Height, Weight, alongside other lifestyle-related variables that are integral to understanding obesity levels. Some of these lifestyle related variables include:

* FAVC (Frequent Consumption of High Caloric Food): This variable records whether an individual frequently eats food high in calories. A high FAVC value may indicate a diet rich in fast food or processed items, which can contribute to a higher risk of obesity.

* CAEC (Consumption of Food Between Meals): The CAEC attribute captures the frequency of snacking or eating between meals. Snacking behaviors, especially on high-calorie or sugary foods, can lead to increased caloric intake and thus influence obesity.

* CH2O (Consumption of Water Daily): This feature quantifies the amount of water an individual drinks. Proper hydration is often associated with overall health and can influence dietary habits, satiety, and metabolism.

All of the records are labeled with the class variable 'NObesity' (Obesity Level), facilitating the categorization into distinct classes like Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, and Obesity Type III. This dataset is available from the UCI Machine Learning Repository.

### Data Exploration

The exploration of the dataset involved delving into both numerical and categorical data. For the numerical features, histograms were created as seen below.

![Numerical Data](https://github.com/vvkumar2/vvkumar2.github.io/assets/52425114/a34454e4-e426-4aac-8ccc-3b93406e306e)
***Figure 1:** Histograms of numerical data, revealing insights such as age distribution and weight patterns.*

From our numerical data in Figure 1, here are a few trends that we see:

1. Age Distribution: The age distribution of the participants, as depicted in Figure 1, shows a significant right skew. This skewness indicates that a majority of the dataset's participants are young, with a mean age around 24 years. This demographic skew could have implications for the generalizability of the study's findings, as it predominantly reflects the obesity levels in a younger population.

2. Height Distribution: The height data presents a normal distribution with both mean and median around 1.70 meters. This normal distribution suggests that height varies regularly and predictably across the dataset, which is expected in a diverse adult population.

3. Weight Distribution: The weight of participants displays a bimodal distribution, meaning there are two different weight groups prominently represented in the dataset. This bimodality may reflect distinct subgroups within the population, each with different weight characteristics.

On the other hand, we analyzed categorical data with bar charts as seen below.

![Categorical Data](https://github.com/vvkumar2/vvkumar2.github.io/assets/52425114/35f9fb65-4296-4341-ae30-a73d5cf18fff)
***Figure 2:** Bar charts for categorical data analysis, highlighting gender distribution and family history of obesity.*

From our categorical data in Figure 2, here are a couple trends that we see:

1. Gender Distribution: The near-equal distribution of men and women in the dataset, as shown in Figure 2, suggests that the results of the study can be considered representative across genders. This balance is crucial for ensuring that the findings are not biased toward one gender.

2. Family History with Overweight: A significant number of participants have a family history of being overweight. This observation is critical as it highlights the potential genetic or environmental factors contributing to obesity, which are important considerations in obesity research and interventions.

Additionally, a correlation analysis, depicted via a heatmap, was very telling. The heatmap of correlations indicated a moderate positive correlation between height and weight, aligning with the general understanding of body mass index (BMI) calculation. However, other features showed little to no correlation, suggesting that factors influencing obesity are not strongly linearly related in this dataset.

Lastly, to better understand the relationships between variables, I created a series of paired plots. Among these, one particularly revealing graph depicted the distribution of each obesity class (NObeyesdad) in relation to family history of obesity. This graph strikingly illustrated that individuals classified in any category of overweight or obesity often have a family history of similar conditions. Conversely, for those categorized as normal weight, the presence or absence of a family history of obesity appeared to be more evenly distributed. This observation is clearly evident in the graph provided below. For a more detailed view of these relationships and other insights, all the plots from the data visualization section are accessible in the Colab Notebook

![Family History of Overweight](https://github.com/vvkumar2/vvkumar2.github.io/assets/52425114/67fed472-f8d4-4c8d-963d-d49223b69bd2)
***Figure 3:** Paired plot depicting the relationship between obesity class and family history of obesity.*

### Data Preprocessing

The preprocessing phase entailed one-hot encoding of categorical variables and normalization of features using Min-Max scaling. 

```python
from sklearn.preprocessing import MinMaxScaler

# Min-Max Scaling
feature_scaler = MinMaxScaler()
obesity_features_scaled = feature_scaler.fit_transform(obesity_features)
```

Normalization was essential to ensure that each variable contributes equally to the analysis and to prevent any feature with a larger scale from dominating the model's learning process. Initially, all features were considered for the model. A strategic 80/20 split was executed to separate the data into training and testing sets.

## Model Selection and Justification

The choice of the RandomForestClassifier was driven by the nature of the problem and the characteristics of the dataset. Given that the task is to predict distinct categories of obesity levels, this is a supervised learning problem with a classification goal.

A RandomForestClassifier was preferred over simpler models like regression due to its ability to handle non-linear relationships between features and the target variable. It is particularly adept at managing complex interactions and hierarchies within the data. Additionally, its inherent mechanism of averaging multiple decision trees reduces the risk of overfitting, making it a robust choice for this dataset.

Random forests also provide valuable insights into feature importance, which is crucial in understanding the underlying factors contributing to obesity. This aspect was leveraged to refine the feature set and improve model performance.

## Model Development & Evaluation

To build the model, hyperparameters like max_depth were meticulously tuned using GridSearchCV to find the optimal balance between model complexity and generalization ability. The following code snippet demonstrates the hyperparameter tuning process:

```python
rf_classifier = RandomForestClassifier(random_state=42)
grid = {
    'max_depth': [5, 7, 9, 11, 13, 15]
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=grid, cv= 5, scoring = f1)
grid_search.fit(X_train, y_train)
best_max_depth = grid_search.best_params_["max_depth"]
```

While GridSearchCV offers the capability to simultaneously test a wide array of hyperparameters, I found that its runtime was excessively lengthy for my needs. Therefore, I opted to evaluate each hyperparameter individually, which proved to be a more time-efficient approach.

After fitting the model, we observed high accuracy in both training (100%) and testing (96%). However, an important aspect of this project was to refine the model by focusing on the most influential features. This was achieved by analyzing feature importance scores and retaining only those features with scores above a certain threshold. The feature importance chart is shown below.

![Feature Importance Chart](https://github.com/vvkumar2/vvkumar2.github.io/assets/52425114/5e6ca07b-0b57-4ad3-8322-78ad1f87e3a3)
***Figure 4:** Feature importance chart, illustrating the relative importance of features like height, weight, and dietary habits.*

In figure 4, most of the features were found to have an extremely low importance score. Only the top ten or so features were contributing significantly to our prediction.

Thus, I reduced the feature set from 31 to 9. However, the accuracy of the model on the testing data showed a negligible change, remaining at 96%. This outcome suggests that many of the removed features, while contributing to the model's complexity, did not significantly impact its predictive ability. This finding underscores the importance of feature selection in machine learning, highlighting that a more concise feature set can yield comparable performance, potentially reducing computational costs and improving model interpretability.

This scenario also illustrates a key learning in model development: the most complex model is not always the most effective or efficient. By refining the feature set, we aimed to create a more streamlined model without compromising on accuracy, which is a valuable approach, especially in real-world applications where interpretability and efficiency are crucial.

Overall, both models demonstrated high accuracy. Detailed classification reports and confusion matrices were generated for each model to further assess their performance. An analysis of feature importance highlighted the pivotal roles of weight, height, age, FCVC (Frequency of Consumption of Vegetables), and NCP (Number of Main Meals), indicating the significant influence of dietary habits and basic physical attributes on obesity levels.

### Results Interpretation

The results of our model were further evaluated using a confusion matrix. A confusion matrix is a table used to describe the performance of a classification model on a set of test data for which the true values are known. It allows the visualization of the model's performance and is particularly useful for assessing the accuracy of a classifier.

```python
from sklearn.metrics import confusion_matrix

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
```
![Screenshot 2023-12-08 at 2 29 12 AM](https://github.com/vvkumar2/vvkumar2.github.io/assets/52425114/5dbd8c76-4447-463b-b3e6-b7e5d721efde)
***Figure 5:** Confusion Matrix for Obesity Level Prediction.*

The confusion matrix provides insights into the types of errors made by the model. For instance, it helps us understand the instances where the model incorrectly predicts a certain class of obesity or fails to identify it accurately. However, in our case we saw that we barely have any incorrect predictions. The matrix in Figure 6 suggests a relatively balanced prediction across different obesity levels, which is indicative of a well-performing model. This information is crucial for refining the model and for understanding the nuances of the predictive process.

## Conclusions and Learnings

The project successfully utilized RandomForestClassifier to predict obesity levels, underscoring the efficacy of this model in handling such datasets. 

The results of our machine learning model offer insightful interpretations, particularly in identifying the most significant factors contributing to obesity. The model highlighted height, weight, gender, age, FCVC (Frequency of Consumption of Vegetables), FAF (Physical Activity Frequency), and FAVC (Frequent Consumption of High Caloric Food) as the most influential features in predicting obesity levels.

Other than the obvious height and weight, dietary habits, as represented by FCVC, FAF, and FAVC, emerged as significant predictors, underscoring the link between lifestyle choices and obesity. High frequency in vegetable consumption (FCVC) may reflect healthier eating patterns, while FAF indicates the level of physical activity, an essential factor in maintaining healthy weight. Conversely, frequent consumption of high-caloric food (FAVC) is a critical factor that potentially leads to increased obesity risk.

These findings, provided by the model, reinforce the multifaceted nature of obesity. They suggest that effective strategies for managing and preventing obesity should not only focus on physical measurements like height and weight but also incorporate modifications in diet and lifestyle, tailored to different genders and ages. This project as a whole also underscores the importance of comprehensive data exploration, preprocessing, and feature selection in the realm of machine learning. Future directions could explore alternative models and more sophisticated feature engineering methods to potentially enhance predictive performance.

## Limitations

Understanding the scope and limitations of our model is crucial for interpreting its applicability and reliability. The RandomForestClassifier, while robust and effective for this dataset, does have its limitations. First, the model's performance is highly dependent on the quality and representativeness of the data. If the dataset lacks diversity or contains biases, the predictions might not generalize well to other populations. 

Additionally, while RandomForest is good at handling non-linear relationships, it does not inherently provide insights into the nature of these relationships, unlike some other models like linear regression. Furthermore, the model's complexity can make it a 'black box', where the decision-making process is not transparent, potentially limiting its use in scenarios where interpretability is crucial. 

Lastly, the accuracy of the model, while high, could be influenced by the imbalance in the dataset, particularly in the distribution of the target classes.
