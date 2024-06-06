# Train and Evaluate Machine Learning Regression Models

<p align="center">
  <img src='pics/bikes.svg'  width='530'/>
</p>

Hello again 👋
+ This repository involves training a Machine Learning Regression Model to predict the number of bicycle rentals, given prevailing weather conditions for a particular day  
+ It is my implementation of a course I've taken on creating and evaluating Machine Learning Regression Models  
+ I have included notes and comments to express my understanding. Links to official documentation are also attached.
+ I've implemented novel concepts (that I've learned on my own) too, such as [`optuna`](https://optuna.readthedocs.io/en/stable/index.html) for hyperparameter search, `early_stopping` and `eval_set` to control overfitting and [`cross_val_score`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) during training rounds to identify the best model.
+ And what do you know, my final model (included in the `model` folder), outperformed the model trained in the course - by far!


## Milestones 🏁
**Concepts covered in this repository include:**  
1. [x] Reading datasets
2. [x] Exploratory data analysis
3. [x] Training and evaluating various regression models ([`LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [`RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor), [`GradientBoostingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor), [`XGBRegressor`](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor), [`LGBMRegressor`](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor))
4. [x] Model optimization through data pre-processing (encoding and scaling)
5. [x] Hyperparameter tuning using [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), [`RandomSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) and [`optuna`](https://optuna.readthedocs.io/en/stable/index.html)
6. [x] Saving and loading models
7. [x] Model inferencing / scoring


## Tools ⚒️
1. [`Google Colab`](https://colab.google/) - A hosted Jupyter Notebook service by Google.
2. [`matplotlib`](https://matplotlib.org/) - A comprehensive library for making static, animated, and interactive visualizations in Python
3. [`pandas`](https://pandas.pydata.org/docs/index.html) - An open-source data analysis and manipulation tool built on Python
4. [`scipy`](https://scipy.org/) - A free and open-source Python library used for scientific computing  
5. [`scikit-learn`](https://scikit-learn.org/stable/#) - A free open-source library that offers machine learning tools for the Python programming language
6. [`xgboost`](https://xgboost.readthedocs.io/en/stable/index.html) - An optimized distributed gradient boosting library that implements machine learning algorithms under the Gradient Boosting framework
7. [`LightGBM`](https://lightgbm.readthedocs.io/en/stable/index.html) - LightGBM is a gradient-boosting framework that uses tree-based learning algorithms
8. [`optuna`](https://optuna.readthedocs.io/en/stable/index.html) - An automatic hyperparameter optimization software framework designed for machine learning


## Reference 📚
+ All thanks to the [`Microsoft Learn`](https://learn.microsoft.com/en-us/) module linked [`here`](https://learn.microsoft.com/en-us/users/martinmuriithi-6560/achievements/9fyc4y7u), not forgetting the [`emojis`](https://gist.github.com/FlyteWizard/468c0a0a6c854ed5780a32deb73d457f) 😸😸😸
