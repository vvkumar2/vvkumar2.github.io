# Obesity Level Prediction: Machine Learning Project Report

## Project Overview

This report presents a detailed analysis of an obesity dataset for a machine learning project conducted as part of A&O SCI C111. The primary objective of this project was to employ a RandomForestClassifier model to predict obesity levels based on a variety of features.

### The Dataset

The dataset, comprising 2111 entries and 17 features such as `Gender`, `Age`, `Height`, `Weight`, and various lifestyle-related attributes, was sourced from a CSV file located at "/content/drive/MyDrive/UCLA/ao-sci/Final Project/data/obesity_data.csv".

### Data Exploration

The exploration of the dataset involved delving into both numerical and categorical data. For the numerical features, histograms were created, revealing insights like a right-skewed age distribution, normally distributed heights, and a bimodal distribution in weight. On the other hand, the analysis of categorical data through bar charts highlighted a balanced gender distribution and a significant proportion of participants having a family history of being overweight. Additionally, a correlation analysis, depicted via a heatmap, primarily indicated a moderate positive correlation between height and weight, with other features showing weak correlations.

### Data Preprocessing

The preprocessing phase entailed one-hot encoding of categorical variables and normalization of features using Min-Max scaling. Initially, all features were considered for the model; however, subsequent feature selection based on importance scores was performed to refine the model.

A strategic 80/20 split was executed to separate the data into training and testing sets.

## Model Selection and Justification

The choice of the RandomForestClassifier was driven by the nature of the problem and the characteristics of the dataset. Given that the task is to predict distinct categories of obesity levels, this is a supervised learning problem with a classification goal.

A RandomForestClassifier was preferred over simpler models like linear regression due to its ability to handle non-linear relationships between features and the target variable. It is particularly adept at managing complex interactions and hierarchies within the data, which are common in medical datasets like ours. Additionally, its inherent mechanism of averaging multiple decision trees reduces the risk of overfitting, making it a robust choice for this dataset.

Random forests also provide valuable insights into feature importance, which is crucial in understanding the underlying factors contributing to obesity. This aspect was leveraged to refine the feature set and improve model performance.

Parameters for the RandomForestClassifier were set based on cross-validation results. Hyperparameters like `max_depth`, `min_samples_split`, and others were tuned to find the optimal balance between model complexity and generalization ability.

## Model Development & Evaluation

The RandomForestClassifier's hyperparameters were meticulously tuned using GridSearchCV, leading to the development of the initial model using all available features. We observed high accuracy in both training (100%) and testing (96%). However, an important aspect of this project was to refine the model by focusing on the most influential features. This was achieved by analyzing feature importance scores and retaining only those features with scores above a certain threshold.

I reduced the feature set from 31 to 9, but the accuracy of the model on the testing data showed a negligible change, remaining at 96%. This outcome suggests that many of the removed features, while contributing to the model's complexity, did not significantly impact its predictive ability. This finding underscores the importance of feature selection in machine learning, highlighting that a more concise feature set can yield comparable performance, potentially reducing computational costs and improving model interpretability.

This scenario also illustrates a key learning in model development: the most complex model is not always the most effective or efficient. By refining the feature set, we aimed to create a more streamlined model without compromising on accuracy, which is a valuable approach, especially in real-world applications where interpretability and efficiency are crucial.

Overall, both models demonstrated high accuracy. Detailed classification reports and confusion matrices were generated for each model to further assess their performance. An analysis of feature importance highlighted the pivotal roles of weight, height, age, FCVC (Frequency of Consumption of Vegetables), and NCP (Number of Main Meals), indicating the significant influence of dietary habits and basic physical attributes on obesity levels.

## Conclusions and Learnings

The project successfully utilized RandomForestClassifier to predict obesity levels, underscoring the efficacy of this model in handling such datasets. Although feature selection based on importance scores led to a more concise feature set, it only slightly improved the model's accuracy, suggesting a marginal impact of this technique in this particular scenario.

This exercise underscored the significance of comprehensive data exploration, preprocessing, and feature selection in the realm of machine learning. Future directions could explore alternative models and more sophisticated feature engineering methods to potentially enhance predictive performance.

## Limitations

Understanding the scope and limitations of our model is crucial for interpreting its applicability and reliability. The RandomForestClassifier, while robust and effective for this dataset, does have its limitations. First, the model's performance is highly dependent on the quality and representativeness of the data. If the dataset lacks diversity or contains biases, the predictions might not generalize well to other populations. Additionally, while RandomForest is good at handling non-linear relationships, it does not inherently provide insights into the nature of these relationships, unlike some other models like linear regression. Furthermore, the model's complexity can make it a 'black box', where the decision-making process is not transparent, potentially limiting its use in scenarios where interpretability is crucial. Lastly, the accuracy of the model, while high, could be influenced by the imbalance in the dataset, particularly in the distribution of the target classes.
