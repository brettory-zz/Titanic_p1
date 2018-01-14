---
title: "Predicting survival on the Titanic: Part 1/2 Exploring margins in R"
# author: "Brett Ory"
thumbnailImagePosition: left
thumbnailImage: https://cdn.images.express.co.uk/img/dynamic/151/590x/secondary/titanic-3-775693.jpg
coverImage: https://cdn.images.express.co.uk/img/dynamic/151/590x/secondary/titanic-3-775693.jpg
metaAlignment: center
coverMeta: out
date: 2018-01-09T21:13:14-05:00
categories: ["Personal projects"]
tags: ["kaggle", "plot", "margins", "logistic regression", "predict", "A/B testing"]
---


```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
library(plyr)
library(dplyr)
train <- read.csv("train.csv")
```



<br>

Yes, [this](http://rstudio-pubs-static.s3.amazonaws.com/24969_894d890964fd4308ab537bfde1f784d2.html) [is](https://rpubs.com/srirampsampath/45661) [yet](https://www.youtube.com/watch?v=AEeWmjSBTjE) [another](https://public.tableau.com/en-us/s/gallery/predicting-survival-titanic) [post](https://www.wolfram.com/mathematica/new-in-10/highly-automated-machine-learning/predict-the-survival-of-titanic-passengers.html) [about](https://prezi.com/3z1vocskmrkn/predicting-survival-on-the-titanic/) [using](http://adataanalyst.com/kaggle/4-different-ways-predict-survival-titanic-part-1/) [the](https://www.wits.ac.za/media/migration/files/cs-38933-fix/migrated-pdf/pdfs-2/Titanic.pdf) [open](https://github.com/topics/titanic-survival-prediction) [source](https://blogs.msdn.microsoft.com/cdndevs/2016/05/13/would-you-have-survived-the-titanic-try-this-step-by-step-machine-learning-experiment-to-find-out/) [Titanic](http://www.ritchieng.com/machine-learning-project-titanic-survival/) [dataset](http://anthony.boyles.cc/portfolio/PredictingSurvivalOnTheTitanic.html) [to](https://rapidminer.com/resource/rapidminer-advanced-analytics-demonstration/) [predict](http://krex.k-state.edu/dspace/bitstream/handle/2097/20541/MichaelWhitley2015.pdf;sequence=1) [whether](https://gallery.cortanaintelligence.com/Experiment/Machine-Learning-experiment-to-predict-survival-chances-of-Titanic-passengers-1) [someone](http://demos.datasciencedojo.com/demo/titanic/) [would](http://logicalgenetics.com/titanic-survivor-prediction/) [live](https://www.jmp.com/content/dam/jmp/documents/en/academic/case-study-library/case-study-library-12/analytics-cases/logistic-titanicpassengers.pdf) [or](https://www.rosaanalytics.com/project2/) [die](https://towardsdatascience.com/play-with-data-2a5db35b279c). 

At this point, there's not much new I (or anyone) can add to accuracy in predicting survival on the Titanic, so I'm going to focus on using this as an opportunity to explore a couple of R packages and teach myself some new machine learning techniques. I will be doing this over two blog posts. In the first post, I will be using logistic regression to calculate how certain attributes of passengers like gender and cabin class are related to their odds of survival. This gives me a chance to explore the R package [margins](https://cran.r-project.org/web/packages/margins/margins.pdf) for computing marginal predicted probabilities, which is based on the Stata command by the same name. Having a lot of experience with Stata, I am excited to see how intuitive margins is, and what types of plots I can make with it. In the second post I will be using R's [randomForest](https://cran.r-project.org/web/packages/randomForest/randomForest.pdf) package to compare its accuracy with predictions based on the logistic regression. 

First, a quick introduction to the data and research question.

<br>  
   

## The data

The Titanic data is free to download from Kaggle, where they have split it into a training and a test set. The training set is a .csv file measuring the following 12 aspects of 891 passengers on the Titanic. The test data is exactly the same as the train set, except without the variable "Survived".

```{r data table, echo = FALSE, warning = FALSE}
library(knitr)

Varname <- names(train)
Description <- c("Passenger ID", "Survival (0 = No; 1 = Yes)", "Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)", "Name", "Sex",
                 "Age", "Number of Siblings/Spouses Aboard", "Number of parents/children aboard", "Ticket number", 
                 "Passenger Fare (British pound)", "Cabin", "Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)")
table <- as.data.frame(cbind(Varname, Description))
kable(table[1:12,])
```
   
<br><br>   
   

![](https://upload.wikimedia.org/wikipedia/en/7/73/Mr._Clean_logo.png)

## Data cleaning
The data is already fairly clean, but it has a few missings and some variables need to be modified before we can include them in an analysis. 


<br>

### Merge test and train data
We eventually want to use the test dataset (test.csv) to check the accuracy of our predictions based on the train dataset (train.csv), therefore we should clean both .csv files together. After all, it would be hard use a user-generated variable in our prediction model if it doesn't exist in the dataset! We now load and merge the test data with the train data
```{r merge test and train}
test <- read.csv("test.csv")
test$tID <- 1 # test data is marked with a 1
test$Survived <- NA # test data is missing Survived
train <- read.csv("train.csv")
train$tID <- 0 # train data is marked with a 0
test <- test[,c(1,13,2:12)] # put variable names in same order as in train
titanic <- rbind(train, test)
rm(train,test)
```

<br>

### Missing data
First, not all missings are coded as NA, so we convert these to NA
```{r recode missings}
titanic[titanic == ""] <- NA
```

A quick glance at the data tells us that there are 263 missings for Age (20.09%), 1 missing for Fare (0.08%), 1014 missings for Cabin (77.46%), and 2 missing for Embarked (0.15%)
```{r tabulate number of missings}
sapply(titanic, function(x) sum(is.na(x))) 
```

We could try to impute some of the missings using multiple imputation, for example, but for the purposes of this project I will just go with quick and easy techniques:

1. Remove Cabin - Generalized linear models by default use only the complete cases (those without missings on any variable) to conduct an analysis. If we include "Cabin" without imputing the missing values, we will only be performing your analysis on the 23% of the passengers that _aren't_ missing this information. At the same time, imputing data for a variable that has so many missings should raise red flags. The simplest way to solve the problem of missing data in this case is to remove the variable all together. If I was feeling really thorough I could test whether the passengers missing "Cabin" are different in any way from passengers not missing "Cabin", but apparently I'm not feeling that thorough.
2. Age - This also has quite a few missings. I will impute those missing values with the mean age for all other passengers (29.88).
3. Fare - I will also impute missings with the mean (to be calculated--see below). It's only one case, so shouldn't have a huge effect on the outcome.
4. Embarked - This I will impute with the mode, or most frequent port of departure (Southhampton). Again, only two cases are missing data on where they embarked so it shouldn't affect our analysis too much.

<br>

### Categorical to dummy variables
The variables Pclass, Sex, and Embarked are categorical. In order to include them in the analysis we need to convert them into dummy variables where each category is either represented by 0 or 1. For two category variables (e.g. Sex), this is just a matter of making one category represented by a 1 and the other by a 0. For three or more category variables (e.g. Pclass and Embarked), we will need to create N-1 dummy variables. 

Convert Pclass into 1st class and 2nd class variables, will use 3rd class as reference category
```{r categorical to dummy Pclass}
titanic$Fclass <- as.integer(titanic$Pclass==1) # First class
titanic$Sclass <- as.integer(titanic$Pclass==2) # Second class
titanic$Pclass <- NULL # delete old Pclass variable
```

Convert gender into numeric variable, where 1 = women and 0 = men
```{r categorical to dummy gender}
titanic$fem <- as.integer(titanic$Sex=="female")
titanic$Sex <- NULL # delete old Sex variable
```

Convert embarked into categorical variables, with Southampton as reference category
```{r categorical to dummy Embarked}
titanic$Embarked[is.na(titanic$Embarked)] <- 'S' # fill 2 missing with mode
titanic$EmbC <- as.integer(titanic$Embarked=="C")
titanic$EmbQ <- as.integer(titanic$Embarked=="Q")
titanic$Embarked <- NULL # delete old Embarked variable
```

<br>

### Fare
A look at the number of unique values for Ticket tells us that there are only 929 tickets, no missings, and 1309 observations in our data. This indicates that some of the tickets cover multiple passengers. 
```{r number of multifare tickets}
length(unique(titanic$Ticket))
```

The Fare variable refers to the price of the ticket, thus if the ticket was for multiple passengers, the fare will also be the cost of entry for multiple people. Our task then becomes finding which fares represent multiple tickets, how many tickets they represent, and then to calculate a fare per person (price). 

Here I create the variable "count" which counts the number of people admitted per ticket
```{r create price per passenger}
aggdata <- titanic %>% group_by(Ticket) %>% summarize(count=n())
titanic <- merge(titanic,aggdata,by="Ticket")
rm(aggdata)
```

Price per person (Fare divided by count)
```{r}
titanic$price <- titanic$Fare/titanic$count
titanic <- subset(titanic, select=-c(Fare,count)) # delete Fare and count variables
```

There are two final transformations necessary to working with price. First, we need to fill in the missing value with the mean:
```{r price mean impute}
# fill 1 missing value with mean
titanic$price[is.na(titanic$price)] <- mean(titanic$price, na.rm=T)
```

Second, price is positively skewed
```{r histogram price, warning=FALSE}
hist(titanic$price) # see positive skew
```

so we will take the log to make it normally distributed.
```{r log price, warning=FALSE}
titanic$price <- log(titanic$price)
titanic$price[is.infinite(titanic$price)] <- 0
hist(titanic$price) # not exactly normal but much better
```


<br>

### Age
We still need to impute the missings for age and see if the variable is normally distributed before we can call our data clean. 

```{r impute missings for age}
titanic$Age[is.na(titanic$Age)] <- mean(titanic$Age, na.rm=T) 
```

Age also looks normally distributed (more or less), so we can stop here.
```{r Age distribution}
hist(titanic$Age)
```


<br>
   
### Test and train
Finally, our data is clean and our variables have been made. We now want to split the data again into test and train so we can run our analyses on the train data and test them on the test set. 

```{r recreate test and train data}
train <- titanic[titanic$tID==0,]
test  <- titanic[titanic$tID==1,]
```

<br> <br>



## Logisitc regression with predicted marginal probabilities

Now that the data is ready, we turn to our analysis. As previously stated, the goal here is to predict which passengers would have survived the titanic. In this post I'm using logistic regression and marginal predicted probabilities to visualize results.


<br>

### Logistic regression

The first thing we want to do is select the variables we think will be important to the analysis. Here's a reminder of the variables in our dataset.
```{r list names of df}
names(train)
```

Ticket was a useful variable for merging, but probably not useful for determining survival. Same for PassengerID, Name<sup><a href="#fn1" id="ref1">1</a></sup>, and tID. Cabin is another variable that has so many missings it's not useful. I'm not going to do extensive hypothesis testing, but based again on having seen the movie, I have a few expectations:

1. Women and younger passengers will be more likely to survive than men and older passengers. 
2. First and second class passengers will be more likely to survive than third class passengers.
3. The combination of being first and second class and female will yeild even higher odds of survival than either being first/second class or female on its own.  

The first two hypotheses I test by including fem, Age, Fclass, and Sclass in the regression. The third hypothesis I test by interacting Fclass and fem as well as Sclass and fem. 

```{r logistic regression}
smalldata <- subset(train, select=-c(Ticket,PassengerId,Name,Cabin,tID))
model1 <- glm(Survived ~Age+SibSp+Parch+Fclass+Sclass+fem+EmbC+EmbQ+price+fem*Fclass+fem*Sclass,family=binomial(link='logit'),data=smalldata)
summary(model1)
```

<br>

### Results

A logistic regression, like all glms, uses a link function (in this case a logistic function) to transform a non-linear equation to a linear one, because linear regression is computationally easy to perform. What this amounts to is that positive coefficients should be read as an increase in the logged odds of suvival per unit increase in any given predictor variable, while negative coefficients can be interpreted as a decrease in the logged odds. There are ways to transform the logged odds to probabilities both manually and using the R predict() function, but we will get to that in a moment. First, let's see what the general relationship is between predictor variables and log odds of survival.     

First of all, it appears James Cameron was on the right track with his "women and children" scene. The positive and sigifnicant coefficient for "fem" tells us that third class women were more likely to survive than third class men. The positive and significant coefficient for the interaction between Sclass and fem tells us that second class women were both more likely to survive than third class women and second class men. Ditto for first class women. Second class men had no survival advantage over third class men as seen by the lack of significant effect for Sclass. Having siblings or a spouse on the boat decreased odds of survival, maybe because all those gallant Edwardian-era husbands were giving up their spots on a lifeboat for their wives. Sadly, there was no similar gallantry between parents and children. As predicted, the younger you were, the higher your odds of survival. Port of embarkment is not significant, and neither is ticket price. 

<br>

### Margins

A useful function in Stata that is, since March, 2017, also available in R is the margins function. Its purpose is to calculate marginal predicted probabilities, and there are two ways to calculate these. First, there is the Average Marginal Effects (AME) which is the default in the margins package. As far as I understand them, AMEs can be interpreted as the change in the probability of survival per unit change in each variable averaged across all cases. Marginal Effects at the Means (MEM) on the other hand, can be interpreted as the probability of survival for an observed value of a variable (Ex: First class), while holding the other variables constant at their means or other meaningful value (Ex: Age = 29, fem = 1). The main thing to keep in mind is that AMEs should be interpreted as a change in probability while MEMs should be interpreted as absolute probability. I now do a demonstration with AMEs 

```{r  margins, echo = TRUE , warning = FALSE, fig.align = "center"}
library(margins)
margins1 <- margins(model1)
summary(margins1)
plot(margins1, labels=c("Age","EmbC","EmbQ","Fclass","fem","Parch","price","Sclass","SibSp")) # I specify the x-axis labels because the plot() function didn't label my axis in the same order as the data. This way I can be sure each label lines up with the right AME 

```
<br>

### Hypothesis testing

I had a hypothesis about how the effect of class would differ for men and women, so I'm going to use the margins command to illustrate that. By specifying the "at" argument, I calculate marginal effects of each variable for men and women separately. 
```{r margins at, echo = TRUE , warning = FALSE, fig.height = 5, fig.width = 10, fig.align = "center"}
margins2 <- margins(model1, at = list(fem = range(smalldata$fem)))
summary(margins2)
```

note in particular the Fclass and Sclass margins:

```{r}
df <- as.data.frame(summary(margins2))
df <- df[(df$factor=="Fclass" | df$factor=="Sclass"),c(1:3,7:8)]
df
```

I now want to plot the marginal effects of class by gender. I wasn't able to figure out how to do so using the plot() or cplot() functions within the margins package, so I'm just going to do it in ggplot2. Easy peasy.

```{r AME plot, warning=FALSE}
library(ggplot2)
df$fem <- as.factor(df$fem) # fem needs to be explicitly labeled as factor instead of integer for ggplot2 to recognize it as such
p <- ggplot(df, aes(factor,AME)) + 
      geom_bar(stat = "identity", aes(fill=fem), position = position_dodge(width = .9)) + 
      xlab("Class") + 
      ylab("Average marginal effects")
p
```

Remember, these are average marginal effects rather than marginal effects at the means. If you remember from the analysis, the reference category here is third class men. We can read this figure as the average increase in probability of survival for first and second class men and women above the probability of survival for third class men. Interactions can be unintuitive, and so even though all this information was also visible in the regression results, the figure shows us the role of class and gender in a way that's easy to grasp. Here we can easily see, for example, how second class women have a higher probability of survival than first class men.  

<br><br>

### Overall predictive power

Now that we have and understand our results, we want to be able to use the logistic regression model to make predictions on a test set. After the post next week using Random Forest, I will test the best model on the test set in Kaggle. But for illustrative purposes now I'm going to assess the accuracy rate for how close our predictions are to the truth in our model. This may be somewhat misleading because I haven't accounted for the possibility of overfitting the data. Nonetheless:

make predictions
```{r}
titanic.predict = predict(model1, newdata = train,
                          type = 'response')
titanic.predict = ifelse(titanic.predict >0.5, 1, 0)
table(titanic.predict)
```

From the 891 cases in the train data, 275 passengers have a probability greater than .5 that they will survive the shipwreck. Now I will test how accurate those predictions are compared to the actual values

```{r}
Accuracy <- mean(titanic.predict == train$Survived)
Accuracy
```
 <br><br>
 
## Conclusions

81% accurate. Next week I will see how logistic regression fares compared to Random Forest. My only conclusion at this point is that the margins package is fine for calculating margins, but not so good for graphing them. 

This post can be found on [GitHub](https://github.com/brettory/Titanic_p1)

<br>

<sup id="fn1">1. [As others have shown [e.g. this anonymous data scientist](http://rstudio-pubs-static.s3.amazonaws.com/24969_894d890964fd4308ab537bfde1f784d2.html), Name does have some useful information, namely title (unmarried woman, married woman, man, no title), but I'm not going to address that here.]<a href="#ref1" title="Jump back to footnote 1 in the text.">↩</a></sup> 
