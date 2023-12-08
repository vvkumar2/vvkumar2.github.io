# A&O SCI C111: Obesity Level Prediction

## Project Overview

This report presents a detailed analysis of an obesity dataset for a machine learning project that I conducted. The primary objective of this project was to employ a classification model to predict obesity levels based on a variety of features. I will detail the process of preparing the data, how I selected the model, and the rationale behind these choices, providing a comprehensive understanding of each step in the project's development.

### The Dataset

"The dataset, titled 'Estimation of obesity levels based on eating habits and physical condition', comprises 2111 instances and 17 features. It was specifically gathered to estimate obesity levels in individuals from Mexico, Peru, and Colombia, focusing on their eating habits and physical condition. The features include fundamental attributes such as Gender, Age, Height, Weight, alongside other lifestyle-related variables that are integral to understanding obesity levels. All records are labeled with the class variable 'NObesity' (Obesity Level), facilitating the categorization into distinct classes like Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, and Obesity Type III. This dataset is available from the UCI Machine Learning Repository.

### Data Exploration

The exploration of the dataset involved delving into both numerical and categorical data. For the numerical features, histograms were created, revealing insights like a right-skewed age distribution (meaning most participants were young), normally distributed heights, and a bimodal distribution in weight. This, plus the insights from the other numerical features can be seen in the chart below.

![Numerical Data](https://github.com/vvkumar2/vvkumar2.github.io/assets/52425114/ef1c6327-b407-478e-9c27-8a095b9000fe)

On the other hand, the analysis of categorical data through bar charts highlighted a balanced gender distribution and a significant proportion of participants having a family history of being overweight. This will later be important in our evaluation.  

![Categorical Data](https://github.com/vvkumar2/vvkumar2.github.io/assets/52425114/626e49de-a943-4dc2-a269-0dc08d898c1c)

Additionally, a correlation analysis, depicted via a heatmap, was very telling. It primarily indicated a moderate positive correlation between height and weight, but most other features had little to no correlation at all. This is understandable once you realize that BMI is calculated based on Height and Weight, and BMI is how you determine obesity level.

![Heatmap](https://github.com/vvkumar2/vvkumar2.github.io/assets/52425114/e1b29386-cc3b-4f79-af71-6bd2d180975a)

Lastly, to better understand the relationships between variables, I created a series of paired plots. Among these, one particularly revealing graph depicted the distribution of each obesity class (NObeyesdad) in relation to family history of obesity. This graph strikingly illustrated that individuals classified in any category of overweight or obesity often have a family history of similar conditions. Conversely, for those categorized as normal weight, the presence or absence of a family history of obesity appeared to be more evenly distributed. This observation is clearly evident in the graph provided below. For a more detailed view of these relationships and other insights, all the plots from the data visualization section are accessible in the Colab Notebook

![Family History of Overweight](https://github.com/vvkumar2/vvkumar2.github.io/assets/52425114/67fed472-f8d4-4c8d-963d-d49223b69bd2)

### Data Preprocessing

The preprocessing phase entailed one-hot encoding of categorical variables and normalization of features using Min-Max scaling. 

```
# Make all feature values between 0 and 1 for normalization
feature_scaler = MinMaxScaler()
obesity_features_scaled = feature_scaler.fit_transform(obesity_features)
```

As seen in the code above, we need to normalize the features to ensure that each variable contributes equally to the analysis and prevent any feature with a larger scale from dominating the model's learning process. Initially, all features were considered for the model. 

A strategic 80/20 split was executed to separate the data into training and testing sets.

## Model Selection and Justification

The choice of the RandomForestClassifier was driven by the nature of the problem and the characteristics of the dataset. Given that the task is to predict distinct categories of obesity levels, this is a supervised learning problem with a classification goal.

A RandomForestClassifier was preferred over simpler models like regression due to its ability to handle non-linear relationships between features and the target variable. It is particularly adept at managing complex interactions and hierarchies within the data. Additionally, its inherent mechanism of averaging multiple decision trees reduces the risk of overfitting, making it a robust choice for this dataset.

Random forests also provide valuable insights into feature importance, which is crucial in understanding the underlying factors contributing to obesity. This aspect was leveraged to refine the feature set and improve model performance.

## Model Development & Evaluation

To build the model, hyperparameters like `max_depth` were meticulously tuned using GridSearchCV to find the optimal balance between model complexity and generalization ability. 

```
rf_classifier = RandomForestClassifier(random_state=42)
grid = {
    'max_depth': [5, 7, 9, 11, 13, 15]
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=grid, cv= 5, scoring = f1)
grid_search.fit(X_train, y_train)
best_max_depth = grid_search.best_params_["max_depth"]
```

As shown in the code above, GridSearchCV was performed. It is a method for systematically cross-validating through combinations of hyperparameters to determine which gives the best performance.


After fitting the model, we observed high accuracy in both training (100%) and testing (96%). However, an important aspect of this project was to refine the model by focusing on the most influential features. This was achieved by analyzing feature importance scores and retaining only those features with scores above a certain threshold. The feature importance chart is shown below.

![Feature Importance Chart](https://github.com/vvkumar2/vvkumar2.github.io/assets/52425114/5e6ca07b-0b57-4ad3-8322-78ad1f87e3a3)

In this figure, we can see that most of the features have an extremely low importance score. In fact, only the top ten or so even seem to be contributing anything to our prediction.

Thus, I reduced the feature set from 31 to 9. However, the accuracy of the model on the testing data showed a negligible change, remaining at 96%. This outcome suggests that many of the removed features, while contributing to the model's complexity, did not significantly impact its predictive ability. This finding underscores the importance of feature selection in machine learning, highlighting that a more concise feature set can yield comparable performance, potentially reducing computational costs and improving model interpretability.

This scenario also illustrates a key learning in model development: the most complex model is not always the most effective or efficient. By refining the feature set, we aimed to create a more streamlined model without compromising on accuracy, which is a valuable approach, especially in real-world applications where interpretability and efficiency are crucial.

![Screenshot 2023-12-08 at 1 29 05 AM](https://github.com/vvkumar2/vvkumar2.github.io/assets/52425114/5fa569fc-ed4e-40c2-8239-1efb49721f31)

Overall, both models demonstrated high accuracy. Detailed classification reports and confusion matrices were generated for each model to further assess their performance. An analysis of feature importance highlighted the pivotal roles of weight, height, age, FCVC (Frequency of Consumption of Vegetables), and NCP (Number of Main Meals), indicating the significant influence of dietary habits and basic physical attributes on obesity levels.

## Conclusions and Learnings

The project successfully utilized RandomForestClassifier to predict obesity levels, underscoring the efficacy of this model in handling such datasets. 

The results of our machine learning model offer insightful interpretations, particularly in identifying the most significant factors contributing to obesity. The model highlighted height, weight, gender, age, FCVC (Frequency of Consumption of Vegetables), FAF (Physical Activity Frequency), and FAVC (Frequent Consumption of High Caloric Food) as the most influential features in predicting obesity levels.

Other than the obvious height and weight, dietary habits, as represented by FCVC, FAF, and FAVC, emerged as significant predictors, underscoring the link between lifestyle choices and obesity. High frequency in vegetable consumption (FCVC) may reflect healthier eating patterns, while FAF indicates the level of physical activity, an essential factor in maintaining healthy weight. Conversely, frequent consumption of high-caloric food (FAVC) is a critical factor that potentially leads to increased obesity risk.

These findings, provided by the model, reinforce the multifaceted nature of obesity. They suggest that effective strategies for managing and preventing obesity should not only focus on physical measurements like height and weight but also incorporate modifications in diet and lifestyle, tailored to different genders and ages. This project as a whole also underscores the importance of comprehensive data exploration, preprocessing, and feature selection in the realm of machine learning. Future directions could explore alternative models and more sophisticated feature engineering methods to potentially enhance predictive performance.

## Limitations

Understanding the scope and limitations of our model is crucial for interpreting its applicability and reliability. The RandomForestClassifier, while robust and effective for this dataset, does have its limitations. First, the model's performance is highly dependent on the quality and representativeness of the data. If the dataset lacks diversity or contains biases, the predictions might not generalize well to other populations. 

Additionally, while RandomForest is good at handling non-linear relationships, it does not inherently provide insights into the nature of these relationships, unlike some other models like linear regression. Furthermore, the model's complexity can make it a 'black box', where the decision-making process is not transparent, potentially limiting its use in scenarios where interpretability is crucial. 

Lastly, the accuracy of the model, while high, could be influenced by the imbalance in the dataset, particularly in the distribution of the target classes.
