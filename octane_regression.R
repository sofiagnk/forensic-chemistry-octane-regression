install.packages("tidyverse")
install.packages("pls")
install.packages("caret")
library(tidyverse)
library(pls)
library(caret)
data <- read.csv("octane.csv")
view(data)
# First column = octane (response)
y <- data[, 1]
# Remaining columns = IR predictors
X <- data[, -1]
dataset <- data.frame(octane = y, X)
print(dataset)
set.seed(123)
train_index <- createDataPartition(dataset$octane, p = 0.7, list = FALSE)
train <- dataset[train_index, ]
test  <- dataset[-train_index, ]
x_train <- train[, -1]
y_train <- train$octane
x_train <- train[, -1]
y_train <- train$octane
x_test <- test[, -1]
y_test <- test$octane
# 4. Scale data (VERY important for IR)
preProc <- preProcess(x_train, method = c("center", "scale"))

x_train_sc <- predict(preProc, x_train)
x_test_sc  <- predict(preProc, x_test)

train_sc <- data.frame(octane = y_train, x_train_sc)
test_sc  <- data.frame(octane = y_test, x_test_sc)
print(train_sc)
print(test_sc)

# 5. Multiple Linear Regression (MLR)
mlr_model <- lm(octane ~ ., data = train_sc)

mlr_pred <- predict(mlr_model, newdata = test_sc)

mlr_rmse <- RMSE(mlr_pred, y_test)
summary(mlr_model)
summary(mlr_model)
#6. Principal Component Regression (PCR)
set.seed(123)

pcr_model <- pcr(octane ~ ., data = train_sc,
                 validation = "CV")

# Choose optimal number of components
summary(pcr_model)
#choose 3 components explain the  96.41 % of octane
# Predict using optimal components (can adjust ncomp)
pcr_pred <- predict(pcr_model, x_test_sc, ncomp = 3)

pcr_rmse <- RMSE(pcr_pred, y_test)
print(pcr_rmse)
# 7. Partial Least Squares (PLS)
# =========================================================
set.seed(123)

pls_model <- plsr(octane ~ ., data = train_sc,
                  validation = "CV")

summary(pls_model)
pls_pred <- predict(pls_model, x_test_sc, ncomp = 3)

pls_rmse <- RMSE(pls_pred, y_test)
print(pls_rmse)
# 8. Model comparison
# =========================================================
results <- data.frame(
  Model = c("MLR", "PCR", "PLS"),
  RMSE = c(mlr_rmse, pcr_rmse, pls_rmse)
)

print(results)
# Best model (lowest RMSE)
best_model <- results[which.min(results$RMSE), ]
print(best_model)
# 9. Plots (optional but good for report)
# =========================================================

# Observed vs Predicted - MLR
plot(y_test, mlr_pred,
     main = "MLR: Observed vs Predicted",
     xlab = "Observed", ylab = "Predicted")
abline(0, 1, col = "red")
# PCR
plot(y_test, pcr_pred,
     main = "PCR: Observed vs Predicted",
     xlab = "Observed", ylab = "Predicted")
abline(0, 1, col = "red")

plot(y_test, pls_pred, 
     main= "PLS: Observed vs Predicted",
     xlab= "Observed", ylab = "Predicted")
abline(0, 1, col = "red")
pred_matrix <- data.frame(
  Actual = y_test,
  MLR = mlr_pred,
  PCR = as.vector(pcr_pred),
  PLS = as.vector(pls_pred)
)

head(pred_matrix)
cor(pred_matrix)
print(pred_matrix)
