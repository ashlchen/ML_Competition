# Logistic model for team project - cross validation + forward feature selection + oversampling

# import necessary packages
```{r,warning=FALSE,message=FALSE message=FALSE, warning=FALSE, r,warning=FALSE}
library(dplyr)
library(caret)
library(e1071)
library(rpart)
library(pROC)
library(glmnet)
library(rpart.plot)
library(ROSE)
library(FSelectorRcpp)
```


# import labeled data
```{r,warning=FALSE,message=FALSE}
data_label <- read.csv("LabelData.csv")
```

# convert outcome "adopter" to be a factor for classification
```{r,warning=FALSE,message=FALSE}
data_label$adopter <- as.factor(data_label$adopter)

```

# user_id not useful as a feature
```{r,warning=FALSE,message=FALSE}
data_label <- data_label %>% select(-user_id)
table(data_label$adopter)
```


# Cross-Validation - split folds
```{r,warning=FALSE,message=FALSE}
cv = createFolds(y= data_label$adopter, k = 5)
auc_cv = c()

```


# logistic regression - forward feature selection
```{r,warning=FALSE,message=FALSE}
best_auc <- 0.5
selected_features <- c()

while(TRUE){

  feature_to_add <- -1

  for(i in setdiff(1:(dim(data_label)[2]-1), selected_features)){

      aucs <- c() # empty vector to store AUC from each fold

      for(test_fold in cv){

      train <- data_label[-test_fold, ] %>% select(selected_features, i, adopter)
      test <- data_label[test_fold, ] %>% select(selected_features, i, adopter)

      train_balanced <- ovun.sample(adopter ~ . , data = train, 
                                   method = "over", 
                                   N = 2*table(train$adopter)[1])$data
      
      logit_model_wf <- glm(adopter ~ . , data = train_balanced, family = "binomial")

      pred_prob_wf <- predict(logit_model_wf, test, type = "response")

      aucs <- c(aucs, auc(test$adopter, pred_prob_wf))
      }

      auc_wf <- mean(aucs) # mean AUC from the current set of features

      if(auc_wf > best_auc){
        best_auc <- auc_wf
        feature_to_add <- i
      }
  }

  if (feature_to_add != -1){
    selected_features <- c(selected_features, feature_to_add)
    print(selected_features) 
    print(best_auc) 
  }
  else break
}
```



```{r message=FALSE, warning=FALSE, r,warning=FALSE}
# to convert to binary (class) predictions
pred_binary <- ifelse(pred_prob_wf > 0.5, 1, 0)


confusionMatrix(factor(pred_binary), factor(test$adopter), positive = "1", mode = "prec_recall")

```


# unlabelled data prediction
```{r message=FALSE, warning=FALSE, r,warning=FALSE}
# import unlabeled data and make predictions
data_unlabel = read.csv("UnlabelData.csv")
pred = predict(logit_model_wf, data_unlabel, type = "class")

# prepare submission
submission = data.frame(user_id = data_unlabel$user_id,
                        prediction = pred)

write.csv(submission, "Team-30-Submission.csv", row.names = FALSE)
```



