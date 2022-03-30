# Logistic model for team project - NB + cross validation + forward feature selection + oversampling + normalization

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
set.seed(1)
data_label$adopter <- as.factor(data_label$adopter)
train_rows <- createDataPartition(y = data_label$adopter, p =0.70, list = F)
```

# user_id not useful as a feature
```{r,warning=FALSE,message=FALSE}
data_label <- data_label %>% select(-user_id)
table(data_label$adopter)

normalize <- function(x){
    return((x-min(x))/(max(x)-min(x)))
}

# one-hot encoding
# = binary indication column
data_label = data_label %>% mutate_at(1:25, normalize) 

```

```{r,message=FALSE,warning=FALSE}

train <- data_label[train_rows,]
test <- data_label[-train_rows,]



```


# NB Cross-Validation - split folds oversampling
```{r,warning=FALSE,message=FALSE}
set.seed(1)
cv = createFolds(y= data_label$adopter, k = 5)


for(test_fold in cv){
    
  train = data_label[-test_fold, ]
  test = data_label[test_fold, ]  
  
  train_balanced <- ovun.sample(adopter ~ . , data = train, 
                                   method = "over", 
                                   N = 2*table(train$adopter)[1])$data
  
  nb_model = naiveBayes(adopter ~., data = train_balanced)
  
  pred_prob = predict(nb_model, test, type = "raw")
  
  test_roc <- test %>% 
  mutate(prob = pred_prob[,2]) %>% 
  arrange(desc(prob)) %>% 
  mutate(yes = ifelse(adopter == 1, 1, 0))
  
  roc_nb <- roc(response = test_roc$yes, # actual values (binary)
              predictor = test_roc$prob)
  
  aucs = c(aucs, auc(roc_nb))}


print(max(aucs))

```




```{r message=FALSE, warning=FALSE, r,warning=FALSE}
# to convert to binary (class) predictions
pred_binary <- ifelse(test_roc$prob > 0.5, 1, 0)


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




