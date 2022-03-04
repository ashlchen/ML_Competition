# Logistic model for team project - cross validation

# import necessary packages
```{r,warning=FALSE,message=FALSE message=FALSE, warning=FALSE, r,warning=FALSE}
library(dplyr)
library(caret)
library(e1071)
library(rpart)
library(pROC)
library(glmnet)
library(rpart.plot)
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

# Cross-Validation (or a simple Training-Validation split) is skipped here
```{r,warning=FALSE,message=FALSE}
cv = createFolds(y= data_label$adopter, k = 5)
auc_cv = c()

```

# Train a logistic model
```{r message=FALSE, warning=FALSE, r,warning=FALSE}
for (test in cv) {
  data_train = data_label[-test,]
  data_test = data_label[test,]
  
  logit_model <- glm(adopter ~ . , data = data_train, family = "binomial")
  
  # there are two types of prediction available for categorical model
  # probability and classification
  pred_prob = predict(logit_model, data_test, type = "response")
  
  # to convert to binary (class) predictions
  pred_binary <- ifelse(pred_prob > 0.5, 1, 0)


  confusionMatrix(factor(pred_binary), factor(data_test$adopter), positive = "1")
  
  auc_cv = c(auc_cv, auc(data_test$adopter, pred_prob))
  
}

cat("Average AUC =",mean(auc_cv),"\n")
```

```{r message=FALSE, warning=FALSE, r,warning=FALSE}
# to convert to binary (class) predictions
pred_binary <- ifelse(pred_prob > 0.5, 1, 0)


confusionMatrix(factor(pred_binary), factor(data_test$adopter), positive = "1")

```

```{r message=FALSE, warning=FALSE, r,warning=FALSE}
# import unlabeled data and make predictions
data_unlabel = read.csv("UnlabelData.csv")
pred = predict(tree, data_unlabel, type = "class")

# prepare submission
submission = data.frame(user_id = data_unlabel$user_id,
                        prediction = pred)

write.csv(submission, "Team-30-Submission.csv", row.names = FALSE)
```




