# wine_variety
---
## 1.About 
* a predictive model to identify wine variety with 10 feature
---
---
## 2.Dataset
* you can download the dataset from->https://drive.google.com/file/d/1kq8PSsj1U_naNzEsUTnmumWy77L8SKI5/view
---
---
## 3.Accuracy
* this model has 58.8 % accuracy on training set
---
---
## 4.Data Preprocessing
  * fill the null value in the dataset
  * combine region_1,region_2 and province to region
  * encode the region,country and winery
  * reduce the range of point
  * preprocess the designation and review_description
---
---
## 5.Model Used
* i have used xgboost model to predict the variety .it has 59 % accuracy in train dataset.
### XGBOOST-->
  ### Accuracy Score->58.8%
  ### Confusion Matrix->
  ![Screenshot from 2020-05-09 16-27-52](https://user-images.githubusercontent.com/52108435/81471985-c5927d80-9212-11ea-845b-4547492a251c.png)

  ### Feature Importance->
  ![Screenshot from 2020-05-09 16-17-58](https://user-images.githubusercontent.com/52108435/81471968-a4ca2800-9212-11ea-8ec9-1ac0d771b260.png)

---
