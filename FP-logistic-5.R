# Logistic model for team project - neural network + cross validation + forward feature selection + oversampling

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
train_rows <- createDataPartition(y = data_label$adopter, p = 0.80, list = F)

normalize <- function(x){
    return((x-min(x))/(max(x)-min(x)))
}

# one-hot encoding
# = binary indication column
data_label = data_label %>% mutate_at(1:25, normalize) %>% 
  mutate(yes = ifelse(adopter == 1, 1, 0),
         no = ifelse(adopter == 0, 1, 0)) 


```

```{r,message=FALSE,warning=FALSE}
library(neuralnet)

train <- data_label[train_rows,]
test <- data_label[-train_rows,]

```


# Cross-Validation - split folds
```{r,warning=FALSE,message=FALSE}
cv = createFolds(y= data_label$adopter, k = 5)

```


# neural network - forward feature selection
```{r,warning=FALSE,message=FALSE}
best_auc <- 0.5
selected_features <- c()

while(TRUE){

  feature_to_add <- -1

  for(i in setdiff(1:(dim(data_label)[2]-3), selected_features)){

      aucs <- c() # empty vector to store AUC from each fold

      for(test_fold in cv){

      train <- data_label[-test_fold, ] %>% select(selected_features, i, adopter, yes, no)
      test <- data_label[test_fold, ] %>% select(selected_features, i, adopter, yes, no)

      train_balanced <- ovun.sample(adopter ~ . , data = train, 
                                   method = "over", 
                                   N = 2*table(train$adopter)[1])$data
      
      nn_model = neuralnet(yes + no ~.,
                     data = train[,-26],
                     act.fct = "logistic",
                     linear.output = FALSE,
                     hidden = 2,
                     algorithm = "backprop",
                     learningrate = 0.1)

      pred_prob_nn <- predict(nn_model, test, type = "response")

      aucs <- c(aucs, auc(test$adopter, pred_prob_nn))
      }

      auc_nn <- mean(aucs) # mean AUC from the current set of features

      if(auc_nn > best_auc){
        best_auc <- auc_nn
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


```{r}



pred <- predict(nn_model, test[,1:25])

# convert to class labels
# original class name
outcomes <- c(1, 0)

# highest datapoint means it belongs to that column
# pick column number (1-2-3) based on highest datapoint 
pred_label <- outcomes[max.col(pred)]

# class prediction and actual outcome
confusionMatrix(factor(pred_label), test$adopter, positive = 1, mode = "prec_recall")

```


```{r message=FALSE, warning=FALSE, r,warning=FALSE}
# to convert to binary (class) predictions
pred_binary <- ifelse(pred_prob_nn > 0.5, 1, 0)


confusionMatrix(factor(pred_binary), factor(test$adopter), positive = "1", mode = "prec_recall")

```


# unlabelled data prediction
```{r message=FALSE, warning=FALSE, r,warning=FALSE}
# import unlabeled data and make predictions
data_unlabel = read.csv("UnlabelData.csv")
pred = predict(logit_model_wf, data_unlabel, type = "response")
pred_binary = ifelse(pred > .5, 1, 0)
  
# prepare submission
submission = data.frame(user_id = data_unlabel$user_id,
                        prediction = pred_binary)

write.csv(submission, "Team-20-Submission.csv", row.names = FALSE)
```




