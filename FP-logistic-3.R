# Logistic model for team project - cross validation + feature selection k=5

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


# Train a logistic model - filter approach feature selection - k=5 + oversample
```{r, warning=FALSE, message=FALSE}


aucs_k5 = c()
for(test_fold in cv){
  train = data_label[-test_fold, ]
  test = data_label[test_fold, ]
  ig <- information_gain(adopter ~ . , data = train)  
  
  #select top 5 features
  topk <- cut_attrs(ig, k = 5)
  
  topk_train <- train %>% select(topk,adopter)
  topk_test <-  test %>% select(topk,adopter)
  
  topk_train_balanced <- ovun.sample(adopter ~ . , data = topk_train, 
                                   method = "over", 
                                   N = 2*table(topk_train$adopter)[1])$data
  
  logit_model_filter <- glm(adopter ~., data = topk_train_balanced, family = "binomial")
  
  pred_prob_filter <- predict(logit_model_filter, topk_test, type = "response")
  
  aucs_k5 = c(aucs_k5, auc(test$adopter, pred_prob_filter))
}
cat("AUC when K=5 = ", mean(aucs_k5),"\n")

```

```{r message=FALSE, warning=FALSE, r,warning=FALSE}
# to convert to binary (class) predictions
pred_binary <- ifelse(pred_prob_filter > 0.5, 1, 0)


confusionMatrix(factor(pred_binary), factor(topk_test$adopter), positive = "1", mode = "prec_recall")

```
# unlabelled data prediction
```{r message=FALSE, warning=FALSE, r,warning=FALSE}
# import unlabeled data and make predictions
data_unlabel = read.csv("UnlabelData.csv")
pred = predict(logit_model, data_unlabel, type = "class")

# prepare submission
submission = data.frame(user_id = data_unlabel$user_id,
                        prediction = pred)

write.csv(submission, "Team-30-Submission.csv", row.names = FALSE)
```



