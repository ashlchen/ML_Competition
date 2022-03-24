# This is a starter script for the team project

---
  title: "SP'22 K579: Project"
author: "Team 20: raon, ashlchen, antfried"
output: html_document
---

```{r, warning=FALSE, message=FALSE}  
# import necessary packages
library(dplyr)
library(caret)
library(e1071)
library(rpart)
```







```{r, warning=FALSE, message=FALSE} 
# import labeled data
data_label <- read.csv("LabelData.csv")
```

```{r, warning=FALSE, message=FALSE} 
# convert outcome "adopter" to be a factor for classification
data_label$adopter <- as.factor(data_label$adopter)
# user_id not useful as a feature
data_label <- data_label %>% select(-user_id)
```




```{r, warning=FALSE, message=FALSE} 
#Decision tree

# Cross-Validation (or a simple Training-Validation split) is skipped here

set.seed(1)

# Randomly pick the rows for training partition
train_rows = createDataPartition(y = data_label$adopter,
                                 p = 0.70, list = FALSE)
adopter_train = data_label[train_rows,]
adopter_test = data_label[-train_rows,]
```

```{r, warning=FALSE, message=FALSE} 
# Train a decision tree
tree <- rpart(adopter ~ ., data = adopter_train,
              method = "class", 
              parms = list(split = "information"))
```

```{r, warning=FALSE, message=FALSE} 
# Tree predictions
pred_tree = predict(tree, adopter_test, type = "class")
```


```{r, warning=FALSE, message=FALSE} 
# plot tree
library(rpart.plot)
prp(tree, varlen = 0)
```

```{r, warning=FALSE, message=FALSE} 
# evaluate tree with AUC
# get class membership probabilities
pred_tree_roc = predict(tree, adopter_test, prob = T)
```

```{r, warning=FALSE, message=FALSE} 
# append to the validation/testing set
adopter_test_roc <- adopter_test %>% 
  mutate(prob = pred_tree_roc[,2]) %>% 
  arrange(desc(prob))

library(pROC)

roc_tree <- roc(response = adopter_test_roc$adopter,
                predictor = adopter_test_roc$prob)
```

```{r, warning=FALSE, message=FALSE} 
# plot ROC 
plot(roc_tree, legacy.axes = T, asp = NA)
```

```{r, warning=FALSE, message=FALSE} 
# evaluate tree with Confusion matrix
confusionMatrix(pred_tree, adopter_test[,1], positive = "1")
```









```{r, warning=FALSE, message=FALSE} 
#SVM
set.seed(1)


# need to specify kernel type
# many other parameters available

svm_model <- svm(adopter ~ . , data = adopter_train, kernel= "linear")
```

```{r, warning=FALSE, message=FALSE} 
# use predict() to get predictions

pred <- predict(svm_model, adopter_test)
```

```{r, warning=FALSE, message=FALSE} 
# confusion matrix

confusionMatrix(pred, adopter_test$adopter, positive = "yes")

confusionMatrix(pred, adopter_test$adopter, mode = "prec_recall", positive = "yes")
```

```{r, warning=FALSE, message=FALSE} 
#2 degree poly kernel
svm_model <- svm(adopter ~ . , data = adopter_train, kernel= "polynomial", degree = 2)

pred <- predict(svm_model, adopter_test)

confusionMatrix(pred, adopter_test$adopter, mode = "prec_recall", positive = "yes")
```

```{r, warning=FALSE, message=FALSE} 
#gaus kernel
svm_model <- svm(adopter ~ . , data = adopter_train, kernel= "radial")

pred <- predict(svm_model, adopter_test)

confusionMatrix(pred, adopter_test$adopter, mode = "prec_recall", positive = "yes")
```









```{r, warning=FALSE, message=FALSE} 
# import unlabeled data and make predictions
data_unlabel = read.csv("UnlabelData.csv")
pred = predict(tree, data_unlabel, type = "class")

# prepare submission
submission = data.frame(user_id = data_unlabel$user_id,
                        prediction = pred)

write.csv(submission, "Team-20-Submission.csv", row.names = FALSE)



### Oversampling
data_label_balanced <- ovun.sample(adopter ~ . , data = data_label, 
                                   method = "over", 
                                   N = 2*table(data_label$adopter)[1])$data

# Train a decision tree
tree_balanced <- rpart(adopter ~ ., data = data_label_balanced,
              method = "class", 
              parms = list(split = "information"))


# import unlabeled data and make predictions
data_unlabel = read.csv("UnlabelData.csv")
pred = predict(tree_balanced, data_unlabel, type = "class")

# prepare submission
submission = data.frame(user_id = data_unlabel$user_id,
                        prediction = pred)

write.csv(submission, "Team-30-Submission.csv", row.names = FALSE)
```
