# **#Car Price Prediction System**





### **#Problem Statement**



Accurately estimating the resale value of a used car is a challenging task because the price depends on multiple interacting factors such as the car’s name, kilometers driven, number of previous owners, fuel type, transmission type, dealer/individual status, and its original (present) price. Traditional manual valuation methods often rely on subjective judgment, which leads to inconsistent and unreliable price estimates.



To address this issue, the aim of this project is to develop a machine learning model that predicts the Selling Price of a used car based on its historical data. By learning patterns from features such as Present Price, Kms Driven, Owner Count, Car Name, Fuel Type, Transmission Type, and Seller Type, the model provides an objective, data-driven, and accurate price estimation. This helps buyers and sellers make informed decisions and improves transparency in the used car market.





### **#Objectives**



1.To determine how different car features are related to the selling price.

2.Preprocess and cleanse real-world used car data to effectively train a model.

3.Compare the performance of different multiple regression algorithms.

4.The best model to use for real-world deployment in car valuation.

5.To develop an interpretable and accurate ML-based prediction system.





### **#Dataset Details**



Total Records: ~301

Total Features: 9 (before encoding)



##### **\*\*Feature Description\*\***



###### **Feature Name**:- **Description**

Car\_Name:- The model/brand name of the car.

Year:- The year in which the car was purchased.

Selling\_Price:- Target Variable — The price (in lakhs) the owner sold the car for.

Present\_Price:- The original showroom price (in lakhs) of the car when new.

Kms\_Driven:- Total kilometers the car has been driven.

Fuel\_Type:- Type of fuel used (Petrol / Diesel / CNG).

Seller\_Type:- Whether the seller is a Dealer or an Individual.

Transmission:- Gear type of the car (Manual / Automatic).

Owner:- Number of previous owners the car had.



##### **\*\*Dataset Purpose\*\***



The dataset is used to train a machine learning model to accurately predict the resale value (selling price) of used cars based on their features.





##### **#Model used**



**1. Linear Regression (Baseline Model)**

Simple and interpretable, but unable to capture the non-linear nature of car pricing.



**2. Ridge \& Lasso Regression (Regularized Linear Models)**

Regularized linear models that reduce overfitting and multicollinearity, but still limited by linear assumptions.



**3. Decision Tree Regressor**

Captures non-linear relationships and handles mixed data types, but prone to overfitting.



**4. Random Forest Regressor (Final Chosen Model)**

Ensemble of decision trees offering high accuracy, strong generalization, reduced overfitting, and effective handling of complex patterns.

Chosen as the final model due to best overall performance.



**5. Boosting Models**

Models like XGBoost, LightGBM, and CatBoost offer high accuracy but require more tuning and training time. These remain strong candidates for future enhancement.





##### **#Results**



The Random Forest Regressor outperformed all other models and delivered strong predictive accuracy on the car price dataset.



Model Performance Metrics

MSE: 0.5703

RMSE: 0.7552

MAE: 0.4964

Train R²: 0.9861

Test R²: 0.9306

Adjusted R²: 0.9209



Images:- 1. [R² for each regression model](https://drive.google.com/file/d/1tmqWTPmPZ4nOF1ZoWoWl67dxSR7pezIB/view?usp=sharing)

         2. [graph of Random Forest Regression ](https://drive.google.com/file/d/1oenjD3P4eAVof0SZdmRc5Kh5grg8dqhW/view?usp=sharing)

         3. [Performance metrics table](https://drive.google.com/file/d/1af9EsImfrvi3L39DYeq_wn45Fz4m68ud/view?usp=sharing)





##### **#Conclusion**



This study shows how machine learning can predict used car  prices with high accuracy. By comparing several regression models, we found that Random Forest Regression delivers the best results, achieving high R² scores and low error values. The machine learning approach has several advantages:



* It removes subjectivity from valuation.
* It increases transparency.
* It provides fast, automated predictions.
* It helps car dealers and consumers make informed decisions.



The developed system can be easily integrated into real-world applications such as:



* Car resale websites
* Loan and insurance agencies
* Dealership pricing systems





##### **#Future Scope**



There are several ways to expand this research:

1\. Integration of Deep Learning

     Neural networks can capture deeper non-linear patterns.



2\. Dataset Expansion

     Including accident history, service records, and owner history.



3\. Image-Based Prediction

     Using car images to analyze physical condition with CNNs.



4\. Real-Time Market Updates

     Dynamic pricing engines based on live market trends.



5\. Deployment

     Rolling out the model as: - REST API - Web or mobile app - Cloud-based service



6\. Use of Boosting Models

     XGBoost and LightGBM may improve accuracy further.





##### **#References**



* M. Ahmad et al., "Car Price Prediction using Machine Learning," 2024 IEEE 9th International Conference for Convergence in Technology (I2CT), Pune, India, 2024, pp. 1-5, doi: 10.1109/I2CT61223.2024.10544124.



* S. S, K. Vineeth, N. Sreesharan, B. Vigneshwaran and B. Saravanan, "Price Prediction of Pre-Owned Automobiles Using Machine Learning-A Comprehensive Survey," 2024 10th International Conference on Advanced Computing and Communication Systems (ICACCS), Coimbatore, India, 2024, pp. 1832-1837, doi: 10.1109/ICACCS60874.2024.10717072.



* R. V. Kulkarni, K. Thopate, F. Khatib, A. Dixit, A. Ingle and A. Kanathia, "Enhancing Used Car Price Predictions with Machine Learning-Based Damage Detection," 2024 5th IEEE Global Conference for Advancement in Technology (GCAT), Bangalore, India, 2024, pp. 1-7, doi: 10.1109/GCAT62922.2024.10923877.

