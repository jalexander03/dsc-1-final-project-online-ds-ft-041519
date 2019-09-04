
# Module 1 Project - Jumpstart Realty

* Student name: **Alex Husted**
* Student pace: **Online Full Time**
* Scheduled project review date/time: **Tuesday May 21, 2019 2:00 EST**
* Instructor name: **Rafael Carrasco**
* Blog post URL: **Work in Progress**

# Project Outline

This notebook represents the final project in Module 1 of Flatiron's Data Science Bootcamp. The Module began with in-depth introductions of the Python language with significant lessons on loops and funtions. Then the Module moved to noteworthy Python libraries and the usefulness of libraries such as Pandas, Numpy, Seaborn, and Matplotlib. Next, Object Oriented Programming (OOP) with classes and instances was presented. Module 1 concluded with the practical employment of statistical distributions to create regression models.

In this project, I will be working with the 'King County House Sales' dataset with the purpose of predicting the future sale price of houses in King County as accurately as possible. I will clean, explore, and model this dataset with a multivariate linear regression to accomplish this purpose. To best illustrate the results of the model, it will be presented from a viewpoint of a hypothetical local startup business, called 'Jumpstart Realty'. Jumpstart Realty looks to offer future buyers with high-caliber information to build trust between buyers and realtors. This project will follow a certain methodology, named OSEMiN, to build a best-fit model. OSEMiN suggests to follow certain relational steps to help us understand the data and build our model. These steps are listed below:

* **Project Method (OSEMiN)**
    * Obtain and load the dataset
    * Scrub and clean the data
    * Explore and understand the data value
    * Model the data through testing processes
    * Interpret the results and conclude

Before moving on with the project, I would like to provide the business understanding for Jumpstart Realty. Jumpstart managers, along with senior agents and brokers, will use the resulting model to predict the housing market in King County barring drastic changes in the economy. The scope of the project includes housing data within King County, the economic health of buyers in the county, and the value of predictors that contribute to housing sales. The project will begin on Monday May 13th, 2019 and is expected to be completed within a week, on Sunday May, 19th 2019. It's expected that all stateholders involved will conclude with the same understanding of the model and resulting datset. 

# Obtain

The process of loading the dataset is relatively straightforward, but salient nonetheless. Now that the business understanding is solidified, it's time to import the necessary Python libraries and eventually the dataset, which is stored in a .csv file titled 'kc_house_data.csv'.

**Primary Observations:** 

In the first inspection into the dataset, it's important to notice that there are 21,597 data entries and 21 columns seperating those values. 

Beginning using the .head( ) method, there are many predicting variables to work with the intention to predict future sale prices of housing in King County. Some early significant predictors include sqft_living, sqft_lot, waterfront, grade, yr_renovated, and possibly sqft_living15. 

Proceeding with analyzing the datatypes, using the .info( ) method, most of the datatypes are stored within integers and floats. There are a few objects that include date and sqft_basement (which should be stored as an integer). 

Then moving forward with the .describe( ) method, the distribution of the dataset becomes known:


* Outliers in the square footage columns with the max values in some columns over 10,000 ft above the mean. 

* The max value in the 'grade' column is 13, meaning the grading scale includes values over 10. 

* The average house was build in 1970.

* There is an average of 2.1 bathrooms per household in the dataset, min value is 0.5 and max is 8.  

* There is an average of 3.3 bedrooms per household in the dataset, min value is 1 and max is 33. 

* The average housing price is 540,000, with a min value of 78,000 and a max of 7,700,000.

* Houses seem to be in pretty good condition, with an average condition rating of 3.4 (1-5 scale).


# Scrub and Clean

During this stage, the data will become processed and ready for exploration. Important steps for this stage include identifying and removing null values, converting datatypes, dealing with outliers, normalizing data, as well as checking for and removing correllated predictors. Data cleaning ensures that the dataset is useable by identifying any errors or corruptions, correcting and manually processing them to prevent errors during the exploration and modeling phases. Let's begin. 

### **Drop Column:** 
Although the id column is important during data entry, it is not significant when building a model to predict future housing prices, therefore I will drop this from our DataFrame.


### **Dealing with Null Values:** 

Dealing with null values is a key factor to build an accurate model for the dataset. To do this, I will first identify if/where null values exist and check how many exist. 

It appears that null values exist for the columns of waterfront, view, and yr_renovated. Considering our dataset contains 21,597 entries, these null values account for almost 20% of the dataset for each column, respectively (except for 'view'). 

When inspecting the *waterfront* column, notice there are only 146 instances out of 19,9221 entries, 0.80%, in which properties actually sit on the waterfront. To deal with the null *waterfront* values, we can be confident in replacing these null values with a 'False' entry.

Now that null values have been replaced in the *waterfront* column, *yr_renovated* must be dealt with. Almost 20% of the dataset in this column is null. It doesn't make sense to remove these columns as much of the data would be replaced. A better option would be to change values that have '0' to reflect the intial construction date (i.e. if a house was never renovated but was built in 1950, then the renovation date would reflect 1950). 

Both *waterfront* and *yr_built* null values have been dealt with and the final null value to deal with is the *view* column. The null values here are a relatively low percentage of the dataset. There are two options - to drop the rows altogether or replace the values with a common occurring value in the column. Looking at the distribution of the dataset, it seems that over 75% of the data within the column retains a '0' value. I am confident in replacing the null values with 0.

*There is one exception to this dataset where null values must be ADDED. In the predictor *sqft_basement*, *basements that do not exist have a square footage of zero. This is problematic further down the line when analyzing linear regression graphs. To resolve this problem, '0.0's in the *sqft_basement* column will be replaced with zeros.* This will occur in the next section:

Amazing, all null values have been dealt with, except for the proper addition of null values in the *sqft_basement* predictor. Time to progess onward with our data cleaning process by converting datatypes into the proper form. 

### **Converting Datatypes:** 
As the dataset becomes clearer and datatypes are identified, there are a few instances of datatypes stored as integers that need to be converted into objects, as well as objects into strings. In looking at our .info( ) method, it's clear that *waterfront* should be stored as a categorical string datatype. There's also a case in which an object should be converted into a string, which occurs in the sqft_basement column. 

For the liking of this dataset, all values have been converted into their proper form. In the converting datatypes section, five predictors were edited to format the dataset for modeling. Now that the null values have been replaced and the datatypes have been converted there are a few techniques to add before beginning the normalizing phase. 

### Add Columns

A few columns should be added to solidify the understanding of the King County dataset. Specifically, it would be useful to know if each house included a basement or if the house has ever been renovated. 

The dataset now seems ready to begin dealing with outliers, normalizing, and removing correllated predictors. With these next steps, we will begin to see a few data visualizations to help understand where the outliers are living and what predictors could be correlated with each other. 

### Outliers

Outliers can represent accurate or inaccurate data. It's important to deal with outliers because they can skew interpretations of data. Outliers are common, especially for large data sets. Nevertheless, in the context of a larger dataset, it's essential to identify deal with outliers to ensure that the interpretation of the King County dataset is as accurate as possible as we move deeper into the OSEMiN process.

![](student_files/student_45_0.png)


![](student_files/student_46_0.png)


Noticing the data points that lie substantially far away from the box and wisker plot, these are considered the outliers that can skew the data in considerable ways. These outliers can be dealt with by using the z-score formula to rid these specific columns of extreme outliers, in this case, data points that lie 4 standard deviations above the mean. The z-score formula can be shown with:

![](student_files/student_50_0.png)

The dataset has been reduced by over 500 values to remove outliers that stood 4 deviations from the mean. This nixed values in our SQFT columns that were outlandish and could have skewed our model in the proceeding steps. The above funtion identifies outliers 4 standard deviations from the mean and creates a list of their unique values. The following .drop( ) method removes those values from the King County dataframe. You can tell the difference between the data points in the BLUE box plot. As of now, the data cleaning process is almost finished. The only tasks left in the process are to check for multicollinearity, normalize specific values and possibly execute feature engineering if necessary. 

### Checking for Multicollinearity

Multicollinearity indicates predictors that are correlated with other predictors.  This happens when a model includes more than one factor that is correlated not just to the target variable, but also to others. As a result, multicollinearity will increase the standard errors of the coefficients. The best ways to check which predictors are correlated with each other is with a heat map and the .corr( ) method. 

![](student_files/student_53_0.png)

During this step, one of the biggest predictors that most people would suggest directly explains variances in housing prices was removed from the dataset. This was achieved because our ***sqft_living*** predictor directly correlates with the greatest about of features within the dataset. Notice the column in the above .corr( ) method, *sqft_living* correlates with two other variables which include *sqft_above* and *sqft_living15*. These two variables correlate with each other because they both account for very similar values. 

### Normalize

The final step for Jumstart Realty in the data cleaning process is to normalize the data. Normalizing the data set suggesrs converting each numeric value to it's corresponding ***z-score*** for the column, which is obtained by subtracting the column's mean and then dividing by the column's standard deviation for every value. The formula for this will be the same used in the detecting outliers stage. 

Throughout this stage, the data became processed and is now ready for exploration. The scrub and cleaning process included identifying and removing null values, converting datatypes, dealing with outliers, normalizing data, as well as checking for and removing correllated predictors. As we move forward, clean data ensures the dataset is in the best possible condition during exploration and especially for building an accurate model. 

# Exploration

Exploratory Data Analysis, or EDA, is an integral part of understanding the King County dataset. Before moving towards building models for King County, it's vital to become familiar with different realtionships within the data. Analyzing these relationships will provide intuition about how to interpret the results of the proceeding models. Asking questions about these relationships beforehand might also supply additional knowledge about relationships that we might have not known existed. This section will further investigate the distribution of data and ask specific questions about the housing market in King County. 

### Build Density Plot


![](student_files/student_64_0.png)



![](student_files/student_64_1.png)



![](student_files/student_64_2.png)



![](student_files/student_64_3.png)



![](student_files/student_64_4.png)



![](student_files/student_64_5.png)



![](student_files/student_64_6.png)



![](student_files/student_64_7.png)



![](student_files/student_64_8.png)


In reviewing these histogram/KDE plots, it's clear to notice that there is positive skew in most of the data. This is resulting from an excess of housing prices and size in square feet of the houses/lot lying a few postive standard deviations from the mean. The peaks within these plots display where the values are concentrated over the interval. The peaks in price and square feet lie slightly below zero. 

### Build Joint Plot


![](student_files/student_67_0.png)



![](student_files/student_67_1.png)



![](student_files/student_67_2.png)



![](student_files/student_67_3.png)



![](student_files/student_67_4.png)



![](student_files/student_67_5.png)



![](student_files/student_67_6.png)



![](student_files/student_67_7.png)



![](student_files/student_67_8.png)



![](student_files/student_67_9.png)



![](student_files/student_67_10.png)


With these join plots, we are checking for linearity assumption between predictors and target variable. In other words, the first scatter visualizations are given that can display the positive correlations between the *price* column and other columns in the dataframe. Specifically, it's clear to identify that *bathrooms*, *floors*, *sqft_above*, *sqft_basement*, *grade*, and *sqft_living15* have a positive correlation with housing prices. When building the model, it will be essential to consider inluding these predictors. 

### Housing Prices by Zipcode:

For the purpose of even futher understanding the dataset, it's meaningful to be curious. Asking questions is the first place to start. In this instance, I would like to find out where in King County housing prices will be the highest. 

* **Task:** Predict where in King County housing prices will be the highest.


* **Question:** *In which zipcode will produce the most expensive houses?*


* **Prediction:** Puget Sound properties on the West side of downtown Seattle. 


![](student_files/student_71_0.png)


**Interpretation:** Using the SwarmPlot from the seaborn library, we are given the distribution of housing prices for each zipcode in King County. The results came back much differently than predicted. The top three zipcodes in which housing prices are the highest is 98004, 98040, and 98112. Here is some information about each of these zipcodes:

* **98004**: 
    * City = Bellevue, WA. 
    * Population = 27,946. 
    * Mean property value = 952,200. 
    * Mean income per household = 117,321. 
    * Median age = 39 years old. 
    * Public school rating = 8/10.
* **98040**:
    * City = Mercer Island, WA.
    * Population = 22,699.
    * Mean property value = 864,000.
    * Mean income per household = 126,359	
    * Median age = 46 years old.
    * Public school rating = 10/10.
* **98112**:
    * City = Seattle, WA
    * Population = 21,077.
    * Mean property value = 980,578.
    * Mean income per household = 96,054.
    * Median age = 39 years old.
    * Public school rating = 7/10.

It seems the previous predictions of housing prices in the Puget Sound were a bit off, however the zipcode of 98119 does carry high property values than average. Looking at the results, commonalities in the zipcodes that present the most expensive housing prices include the mean income per household and the school rating. Bellevue and Mercer Island both retain their own public school district, with high distinction. Seattle public schools are rated a bit lower, however still retain a 7/10 rating. 

### Basement vs Housing Prices:

In this next exploration let's look at our *basement* columns within the dataset. Some properties do not include a basement, others do. To answer the question provided below, we will be looking at both the *sqft_basement* column and the *basement* columns versus our *price* columns. 

* **Task:** Predict if basements have an impact on housing prices.


* **Question:** *Does having a basement increase the housing price in King County? If so, does basement size increase housing price?*


* **Prediction:** Having a basement will increase housing price. 


![](student_files/student_77_0.png)


**Interpretation:** Violin plots identify probablity density of our data. The middle line is a kernel density estimation to show the distribution shape of the data. The wider sections of the violin plot represent a higher probability that datapoints will conform to that value, and more skinny portions represent a lower probability. The white dot in the middle represents the mean. In these results about housing prices relating to the inclusion of a bsaement, the prediction yields to be True. It's clear to tell the mean housing price is HIGHER when a basement is included in the property. It also appears that housing prices in the upper quartiles are more expensive when a basement is included. 

Let's find out if the size of the basement will have a positive correlation with housing prices:

![](student_files/student_79_1.png)


With the above lmplot we are able to identify that when a house includes a larger basement in square footage, it will usually increase the value of the house. However there are a few outliers in the upper left portion of the graph, the orange line that strikes through the graph signifies a positive correlation with the inclusion of a basement. The given orange line is not a perfect line of best fit, but it respresents the flow of data. 

### Year Built vs. Housing Price:

For this next exploration let's look at our *yr_built* column within the dataset. When looking at the dataset, the oldest property built in King County is in 1900. The newest proprty built finished just recently in 2015. There is a spread of 115 years of property ages within this dataset.  To answer the question provided below, we will be looking at both the *yr_built* column and the *price* columns.

* **Task:** Predict if year built has an impact on housing prices.


* **Question:** *Are houses built more recently more valuable, in terms of price, compared to houses built more than half a century ago?*


* **Prediction:** Houses completed more recently will be more expensive. 

![](student_files/student_83_0.png)

**Interpretation:** Using this wide-ranging violin plot to analyzing housing values in the 115 year span between 2015 and 1900, it's clear to notice houses built pre-1930 and post-1996 result in the highest average value. Houses built between 1942 and 1980 average the lowest value within the dataset. For Jumpstart Realty, driving the prices up for old AND new homes seems like the best plan of action as these homes pre-1930 and post-1996 have the highest market value. For the homes that lie in the middle of this 115 range, it's important for Jumpstart to understand these homes have the least value on the market. Initial predictions of 'houses completed more recently will be more expensive' was not entirely incorrect, as housing prices were higher in houses completed in the past two decades. But the prediciton failed to mention housing prices completed pre-1930, in which our visualization proved to be so. 

### Housing Grade vs. Year Built & Price:

With this exploration, let's analyze the *grade* column within the dataset. The grade of the house represents the construction quality of improvements. An average grade of construction and design is a 7 (grades range from 1-13), and is commonly seen in plats and older sub-divisions. To complete this analysis, the *grade* column will be compared with the year the house was completed and it's effect on price. 

* **Task:** Predict if grade correlates to grade and housing prices.


* **Question:** *Does the grade of the house correlated to the year it was built? What does that say about it's housing price?*


* **Prediction:** Houses completed more recently will have a higher grade and price. 

![](student_files/student_87_0.png)

**Interpretation:** With this violin plot, there is visible distinction that newer houses provide a better housing grade. The mean of houses built before 1980 contain a grade rating of 7 or below. For the average house built after 1980, the grades range from 8-12. There is a clear relationship in which the year a house was built directly impacts it's rating from the King County residential department. Moving forward with this exploration, let's try to find out if grade impacts the housing price.

![](student_files/student_89_0.png)

The above line plot confirms the predictions that if a house if graded highly, it's price will become more valuable. Using grade in this instance will be a great feature to add into the model in the next phase. Within the exploration completed here, it's noteworthy to identify that newer houses result in a higher grade, which created a feedback loop driving the value of the house upwards. 

### Renovations vs. Condition, Grade & Price:

This final exploration will use the *renovation* column to determine if renovating a house will increase it's condition, grade, and price. The condtion of a house is relative to age and grade, coded 1-5. An average score of 3 indicates some evidence of deferred maintenance and normal obsolescence with age in that a few minor repairs are needed, along with some refinishing. All major components still functional and contributing toward an extended life expectancy. Effective age and utility is standard for like properties of its class and usage.
The grade of the house represents the construction quality of improvements. An average grade of construction and design is a 7 (grades range from 1-13), and is commonly seen in plats and older sub-divisions. To complete this analysis, the *renovation* column will be compared with the condition of the house, grade of the house, and it's price. 

* **Task:** Predict if renovations play a role in housing prices.


* **Question:** *Can renovating a house increase not only it's condition and grade, but also it's value?*


* **Prediction:** Renovations will positively impact all three predictors.

![](student_files/student_94_0.png)

**Interpretation:** Surprisingly, according to this visualization, if an owner has renovated the house it has a minimal impact on the overall condiiton for properties in King County. This might be due to the fact that a condition grade takes into effect the age of the house as one of its' metrics. Let's find out if this is also true for the *grade* of the house:

![](student_files/student_96_0.png)

Again, surprisingly, if an owner has renovated the house it has a minimal impact on the overall grade for property. However, there is a slightly larger margin in this visualization. Notice the white dot in these boolean columns is different. If a house has been renovated, the mean grade for the house is an 8. If a house hasn't been renovated, the mean grade for the house is an 7. There is something to be said about renovating a house. Finally, let's check if renovation has a correlation to price: 

![](student_files/student_98_0.png)

Here we see a distinct difference between renovating a house and not renovating a house when identifying its' value. Renovating a house results in a higher value of almost one standard deviation than not renovating a house. This is an important feature to account for when building the model. This was the final visual created during the exploration step. It's time to move forward with one of the most important pieces of Data Science, which is building a model. 

### Waterfront:

![](student_files/student_101_0.png)

# Modeling

Modeling is about prediction. In this model, we will be utilizing linear regression. In simple linear regression, there is a one-to-one relationship between the target variable and the predictor variable. However in multiple linear regression, there is a many-to-one relationship. Instead of using one input features, several are employed to identify the best relationship within your data. Furthermore, data models often use multiple iterations of linear regression to view the same data and ensure that all processes and correlations have been fulfilled to create a best-fit model. The mathematical equation for linear regression is as follows: 


Revisiting the goal of this project, Jumpstart Realty wants to predict the future sale price of houses in King County as accurately as possible for the purpose of offering future buyers with high-caliber information to build trust between buyers and realtors. With the appropriate features within this dataset selected, Jumpstart will be able to build a future model to accurately estimate future values of properties. 


### Simple Linear Regression

To begin building an accurate model, let's start by building a linear regression that only takes one feature and pairs it with our target. In this case, *sqft_total* will become our feature to be paired with our housing *price*. This linear regression will established a relationship between the feature and the target using a best-fit line. We will attempt to draw a line that comes closest to the data by finding the slope and intercept. Also, to make further predictions, we will also be using an OLS summary to analyze our model, features, and residuals. 


**Interpretation:** 
* *Model Results:*
    * R-Squared = 0.050 $\rightarrow$ Total square footage predicts 5% of the variance for price.
    * F-Statistic = 1110. $\rightarrow$ Cannot confirm total square footage is signficant. 
* *Features:*
    * P-Value = 0 $\rightarrow$ Total square footage predicting price isn't random. 
    * Coefficient = 0.0001 $\rightarrow$ Total square footage has little correlation with price. 
        * $y = 0.0001*\text(sqft.total)$
* *Residuals:*
    * Skewness = 1.797 $\rightarrow$ Residuals are positively skewed.
    * Kurtosis = 7.565 $\rightarrow$ Residuals do not fall within 3 standard deviations of mean.
    * Jarque-Bera = 29428.026 $\rightarrow$ Data is not normal. 

![](student_files/student_109_0.png)

Using this simple linear regression model, we found that *sqft_total* predicts only 5% of the variance of our target *price*. The resulting F-statistic doesn't prove that the use of total square footage will predict the target. Our P-value indicates that this relationship isn't random, and the coefficient reveals little correlation. The data, given by our results, implies skewness and much of the data doen't fall within 3 standard deviations from the mean. Overall, using only this feature to predict future housing prices will not yield enough to build a satisfactory model. 

### Multiple Linear Regression: Model 1

Building an accurate model takes time, contemplation, and experimentation about which features should be included. Multiple regressions are based on the assumption that there is a linear relationship between both the target and the features. It also assumes no major correlation between the independent variables, which is why we checked for multicollinearity in the data scrubbing phase.


**Interpretation:** 
* *Model Results:*
    * R-Squared = 0.570 $\rightarrow$ Selected features predicts 57% of the variance for price.
    * F-Statistic = 2314. $\rightarrow$ Cannot confirm features are signficant. 
* *Features:*
    * P-Values = 0 $\rightarrow$ Features predicting price isn't random. 
    * P-Values (sqft_lot) $\rightarrow$ Lot square footage predicting price is random
    * Coefficients = 0.0001 $\rightarrow$ Most significant coefficients have a positive correlation with price. 
* *Residuals:*
    * Skewness = 1.197 $\rightarrow$ Residuals are positively skewed.
    * Kurtosis = 6.994 $\rightarrow$ Residuals do not fall within 3 standard deviations of mean.
    * Jarque-Bera = 18911 $\rightarrow$ Data is not normal. 

![](student_files/student_115_0.png)


Using this multiple linear regression model, we found that the selected features predict 57% of the variance of our target *price*. The resulting F-statistic doesn't prove that the use of the features will predict the target with total accuracy. The P-value indicates that most relationships aren't random (except for *sqft_lot)*, and the coefficient reveals positive correlation. The data, given by our results, implies skewness and much of the data doen't fall within 3 standard deviations from the mean. Overall, using these selected features yield a decent model for us to build upon. NOTICE: However, the condition number is large, 1.42e+07. This might indicate that there are strong multicollinearity or other numerical problems. This will be fixed in the next model.

### Multiple Linear Regression: Model 2

As said previously, building an accurate model takes time and experimentation. Multiple regressions are based on the assumption that there is a linear relationship between both the target and the features. It also assumes no major correlation between the independent variables, which is why we checked for multicollinearity in the data scrubbing phase. In the previous model, there was a factor that lead to significant multicollinearity. This will have to be fixed with model 2.


**Interpretation:** 
* *Model Results:*
    * R-Squared = 0.561 $\rightarrow$ Selected features predicts 56.1% of the variance for price.
    * F-Statistic = 3344. $\rightarrow$ Cannot confirm features are signficant. 
* *Features:*
    * P-Values = 0 $\rightarrow$ Features predicting price isn't random. 
    * Coefficient = -2.73 $\rightarrow$ Most significant coefficients have a positive correlation with price. 
    * Feature Coefficients > 0 $\rightarrow$ All coefficients have a positive correlation with price. 

* *Residuals:*
    * Skewness = 1.177 $\rightarrow$ Residuals are positively skewed.
    * Kurtosis = 6.771 $\rightarrow$ Residuals do not fall within 3 standard deviations of mean.
    * Jarque-Bera = 17235 $\rightarrow$ Data is not normal.


![](student_files/student_125_0.png)


This iteration of a multiple linear regression model finds that the selected features predict 56.1% of the variance of our target *price*. The resulting F-statistic doesn't prove that the use of the features will predict the target with total accuracy. The P-value indicates that all relationships aren't random and all feature coefficients reveal positive correlation. The data, given by our results, implies skewness and much of the data doen't fall within 3 standard deviations from the mean. Overall, using these selected features yields a valid model. The condition number is acceptable and inicates reasonable correlation. Additionally, a constant was added to the regression to aid the model in finding a best-fit line. 

### Multiple Linear Regression: Model 3


**Interpretation:** 
* *Model Results:*
    * R-Squared = 0.500 $\rightarrow$ Selected features predicts 50% of the variance for price.
    * F-Statistic = 2620. $\rightarrow$ Cannot confirm features are signficant. 
* *Features:*
    * P-Values = 0 $\rightarrow$ Features predicting price isn't random. 
    * Feature Coefficients >= 0 $\rightarrow$ Most coefficients have a positive correlation with price. 

* *Residuals:*
    * Skewness = 1.153 $\rightarrow$ Residuals are positively skewed.
    * Kurtosis = 6.440 $\rightarrow$ Residuals do not fall within 3 standard deviations of mean.
    * Jarque-Bera = 14952 $\rightarrow$ Data is not normal.


![](student_files/student_130_0.png)


This iteration of a multiple linear regression model finds that the selected features predict 50% of the variance of our target price. The resulting F-statistic doesn't prove that the use of the features will predict the target with total accuracy. The P-value indicates that all relationships aren't random and most feature coefficients reveal positive correlation. The data, given by our results, implies skewness and much of the data doen't fall within 3 standard deviations from the mean. Overall, using these selected features yields an averagly acceptable model. The condition number is acceptable and inicates reasonable correlation. In this iteration, the constant was removed and resulted in a worse R-squared score. 

### Multiple Linear Regression: Model 4

**Interpretation:** 
* *Model Results:*
    * R-Squared = 0.561 $\rightarrow$ Selected features predicts 56.1% of the variance for price.
    * F-Statistic = 3344. $\rightarrow$ Cannot confirm features are signficant. 
* *Features:*
    * P-Values = 0 $\rightarrow$ Features predicting price isn't random. 
    * Coefficient = -2.73 $\rightarrow$ Most significant coefficients have a positive correlation with price. 
    * Feature Coefficients > 0 $\rightarrow$ All coefficients have a positive correlation with price. 

* *Residuals:*
    * Skewness = 1.177 $\rightarrow$ Residuals are positively skewed.
    * Kurtosis = 6.771 $\rightarrow$ Residuals do not fall within 3 standard deviations of mean.
    * Jarque-Bera = 17235 $\rightarrow$ Data is not normal.

![](student_files/student_135_0.png)


This iteration of a multiple linear regression model yields the same results as iteration 2. The selected features predict 56.1% of the variance of our target *price*. The resulting F-statistic doesn't prove that the use of the features will predict the target with total accuracy. The P-value indicates that all relationships aren't random and all feature coefficients reveal positive correlation. The data, given by our results, implies skewness and much of the data doen't fall within 3 standard deviations from the mean. Overall, using these selected features yields a valid **final** model. The condition number is acceptable and inicates reasonable correlation. A constant was added to the regression to aid the model in finding a best-fit line. 

### Final Model

The final model and its' parameters determine the linear regression, which will help predict the future housing prices for Jumpstart Realty. The coefficients in the equation and its' constant value embody the parameters. As stated before, building an accurate model takes time, contemplation, and experimentation. This model phase, inlucing the final model, has experimented with five models that identify proper features with little colliniarity. 


**Interpretation:** 
* *Model Results:*
    * R-Squared = 0.555 $\rightarrow$ Selected features predicts 55.5% of the variance for price.
    * F-Statistic = 3733. $\rightarrow$ Cannot confirm features are signficant. 
* *Features:*
    * P-Values = 0 $\rightarrow$ Features predicting price isn't random. 
    * Coefficient = -2.83 $\rightarrow$ Most significant coefficients have a positive correlation with price.
    * Feature Coefficients > 0 $\rightarrow$ All coefficients have a positive correlation with price. 

* *Residuals:*
    * Skewness = 1.202 $\rightarrow$ Residuals are positively skewed.
    * Kurtosis = 6.784 $\rightarrow$ Residuals do not fall within 3 standard deviations of mean.
    * Jarque-Bera = 17524 $\rightarrow$ Data is not normal.


### Final Model Summary

The final iteration of the multiple linear regression model yields the final model in which Jumpstart Realty will use to predict future housing prices. The selected features predict 55.5% of the variance of our target *price*. The resulting F-statistic doesn't prove that the use of the features will predict the target with total accuracy. The P-value indicates that all relationships aren't random and all feature coefficients reveal positive correlation. The data, given by our results, implies skewness and much of the data doen't fall within 3 standard deviations from the mean. Overall, using these selected features yields a valid **final** model. The condition number is acceptable and inicates reasonable correlation. A constant was added to the regression to aid the model in finding a best-fit line. 

To begin validating the model, a train & test procedure was performed for the purpose of splitting the data into smaller sections and iterating through the model again. The test mean-squarred error resulted in a value of 0.447. The train mean-squarred error resulted in a value of 0.444. If the test error is substantially worse then our train error, this is a sign that our model doesn't generalize well to future cases. In this case, our Train MSE is right on par with the Test MSE. This is a good sign - it means the model will hold true with new data. This model also was satisfactory when splitting the data into differenct sections, using the cross validation results method in SKlearn. The test yielded values very similar to the test MSE.

**Final Regression Equation**:

y = $-2.83$ + $0.218(view)$ + $0.344(grade)$ + $0.231(sqftabove)$ + $0.106(sqftliving15)$ + 
         $0.445(renovatedYN)$ + $0.388(basementYN)$ + $0.952(waterfrontYN)$


![](student_files/student_162_0.png)


### Recommendations

To reiterate the usefulness of this business case, Jumpstart managers, along with senior agents and brokers, will use the resulting model to predict the housing market in King County. The scope of the project included housing data within King County, the economic health of buyers in the county, and the value of predictors that contribute to housing sales. The project will began on Monday May 13th, 2019 and was completed within a week, on Sunday May, 19th 2019. It's expected that all stateholders involved will conclude with the same understanding of the model and resulting datset. Jumpstart Realty looks to offer future buyers with high-caliber information to build trust between buyers and realtors.

**List of Recommendations**: 
    
* *Increase View Count*: The more views a house receives, the higher average value the house will obtain. Use this to identify properties that will become 'hot' on the market and capitalize on this. When a property starts to receive for views, increase marketing efforts for that particular property. This could drive value even higher with more offers coming in. 
    
    
* *Make Modern Renovations*: Housing grades are a reflection of design, features, and material quality. The better the grade, the higher value for the house. To improve upon a housing grade, encourage an owner to make renovations. As we have seen duirng the exploration phase, the more modern the features within the house, the higher the value.

    
* *Increase Square Footage*: This goes without saying - more square footage means a higher value for the house. Our model shows that when square footage is larger, the higher the price. To add onto square footage, consider making renovations to the first floor that could add a porch or a sun room. 


* *Neighbors Determine Value*: When the average square footage is higher in the closest 15 properties, the value of the house is increased. This is observable in the suburban zipcodes, such as Bellevue and Mercer Island. Agents and brokers should understand that when a property falls within the top three or five most valuable zipcodes, the higher the average square footage. Buyers with higher average incomes will be interested in these properties. It's also noteworthy that school ratings are also higher.


* *Add Basements to New Developments*: As empty lots become housing developments, make sure to recognize that a house with a basement with incur a higher value. And the more square footage the basement has, the higher the value. Current properties with a basement should be identifed and labeled with the potential for a higher price compared to the same property without a basement. 


* *Waterfronts are Key*: When a property sits on the waterfront, it's value is extremely higher than the same property without a waterfront view. It's important to recognize this fact. It's value is almost 50% higher than properties without a waterfront view.
    

### Conclusion

This wraps up the King County dataset project. In this project, I analyzed the 'King County House Sales' dataset with the purpose of predicting the future sale price of houses in King County as accurately as possible. The project cleaned, explored, and modeled this dataset with a multivariate linear regression to accomplish the purpose. To best illustrate the results of the model, it will be presented from a viewpoint of a hypothetical local startup business, called 'Jumpstart Realty'. Jumpstart Realty looks to offer future buyers with high-caliber information to build trust between buyers and realtors. This project followed a methodology, called OSEMiN, to build a best-fit model. OSEMiN suggested to follow certain relational steps to help understand the data and build the model. The framework is now completed and recommendations have been given.

### Futher Work

If given more time and data, I would like to have known features about current owners of each property iD. Some examples could include the average income, age, children count, and if relatives live within the county. This could add indicators about which customers would fit into each property. Because this dataset given was from a snapshot in time in 2015, I would also like to analyze the ebbs & flows of property values over time. This could identify new trends in 'up & coming' areas of King County and also areas that are in decline. 

Overall I'd like to thank the Jumpstart Realty team for the opportunity to work on this dataset and for the responsibility of presenting these findings.
