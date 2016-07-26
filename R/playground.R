# NOTE this file is not intended to be executed as a whole,
# it's an unstructured collection of code snippets
# that I used for experiments or submissions.
# The final submission was made by snippet on lines 151-158

search_space <- readRDS("../output/search-space.rds")
search_space <- distinct(bind_rows(search_space, gs))
saveRDS(search_space, "../output/search-space.rds")

search_space %>% arrange(mean) %>% head(50) %>% arrange(promo_after + promo2_after + holiday_b_before + holiday_c_before + holiday_c_after, mapply(max, promo_after, promo2_after, holiday_b_before, holiday_c_before, holiday_c_after))

set_cores(6)
options(width=120)
gs <- grid_search(train,
                  function(...) cv_glmnet(...),
                  promo_after = 4, promo2_after = 5,
                  holiday_b_before = 5,
                  holiday_c_before = 15,
                  holiday_c_after = 10,
                  alpha = c(1), family = c("g"))


## glmnet train errors

with_id <- train %>%
  filter(is.element(Store, unique(test$Store))) %>%
  remove_before_changepoint() %>%
  log_transform_train() %>%
  remove_outliers_lm() %>%
  mutate(Id = 1:n()) %>%
  select_features(linear_features)

train_tr <- select(with_id, -Id)
test_tr <- select(with_id, -Sales)
actual <- mutate(with_id, Sales = exp(Sales))

pred <- predict_glmnet(train_tr, test_tr, exp_rmspe)

pred$predicted <- log_revert_predicted(pred$predicted)

rmspe_pred <- rmspe_per_store(pred$predicted, actual)
qplot(x = rmspe_pred$rmspe, xlim = c(0.05, 0.25))
view_sales(actual, pred$predicted, stores = rmspe_pred[order(rmspe_pred$rmspe, decreasing=T),]$Store, log_sales = TRUE)


errors <- actual %>%
  left_join(select(pred$predicted, Id, PredictedSales), by = c("Id")) %>% 
  select(PredictedSales, Sales) %>%
  mutate(Error = PredictedSales - Sales,
         LogError = log(PredictedSales) - log(Sales),
         PercentageError = (Sales - PredictedSales)/Sales)

qplot(x = errors$Sales, y = errors$Error)
dev.new()
qplot(x = errors$Sales, y = errors$LogError)
dev.new()
qplot(x = errors$Sales, y = errors$PercentageError)
dev.new()
qplot(x = log(errors$Sales), y = errors$LogError)


## stacking on train errors

errors <- actual %>%
  left_join(select(pred$predicted, Id, PredictedSales), by = c("Id")) %>% 
  select(PredictedSales, Sales)
rmspe(errors$PredictedSales, errors$Sales)
rmspe(errors$PredictedSales * 0.986, errors$Sales)
rmspe(errors$PredictedSales * 0.985, errors$Sales)
rqfit <- rq(Sales ~ PredictedSales + 0, data = errors)
rmspe(errors$PredictedSales * coef(rqfit), errors$Sales)


## glmnet long fold

source("functions.R")

fold <- train %>%
  filter(is.element(Store, unique(test$Store))) %>%
  remove_before_changepoint() %>%
  make_fold(step = 1, predict_interval = 9*7)

train_tr <- fold$train %>%
  log_transform_train() %>%
  select_features(linear_features)

test_tr <- select_features(fold$test, linear_features)

fold_pred_tr <- predict_glmnet(train_tr, test_tr, exp_rmspe)

fold_pred <- log_revert_predicted(fold_pred_tr$predicted)
rmspe(fold_pred$PredictedSales, fold$actual$Sales)

rmspe_pred <- rmspe_per_store(fold_pred, fold$actual)
qplot(x = rmspe_pred$rmspe, xlim = c(0.05, 0.25))
view_sales(train, fold_pred, stores = rmspe_pred[order(rmspe_pred$rmspe, decreasing=T),]$Store)

##  0.1178079
##  0.1153605 with outliers

## stack linear model

library(quantreg)
rqfit <- rq(fold$actual$Sales ~ 0 + fold_pred$PredictedSales)
rmspe(fold_pred$PredictedSales * coef(rqfit), fold$actual$Sales)

lmfit <- lm(fold$actual$Sales ~ 0 + fold_pred$PredictedSales)
rmspe(fold_pred$PredictedSales * coef(lmfit), fold$actual$Sales)


## submit glmnet
train_tr <- train %>%
  remove_before_changepoint() %>%
  filter(is.element(Store, unique(test$Store))) %>%
  log_transform_train() %>%
  remove_outliers_lm() %>%
  select_features(linear_features)
test_tr <- select_features(test, linear_features)

pred <- predict_glmnet(train_tr, test_tr, exp_rmspe, steps = 15, step_by = 3)$predicted %>% log_revert_predicted()

save_predicted('../output/glmnet15.csv', pred)
## 0.11323 	0.12271 (1193rd)
view_sales(train, pred)


## submit glmnet with combinations

combination_features <- readRDS("../output/combination-features.rds")

train_tr <- train %>%
  remove_before_changepoint() %>%
  filter(is.element(Store, unique(test$Store))) %>%
  log_transform_train() %>%
  remove_outliers_lm() %>%
  make_selected_combinations(combination_features) %>%
  select_features(c(linear_features, combination_features))

test_tr <- test %>%
  make_selected_combinations(combination_features) %>%
  select_features(c(linear_features, combination_features))

pred <- predict_glmnet(train_tr, test_tr, exp_rmspe, steps = 15, step_by = 3)$predicted %>% log_revert_predicted()

save_predicted('../output/glmnet_comb.csv', pred)
## 0.11169 	0.12734
view_sales(train, pred)



## mix models
glmpred <- read.csv('../output/glmnet15.csv', as.is = TRUE)
xgbpred <- read.csv('../output/xgb4.csv', as.is = TRUE)
pred <- glmpred %>%
  inner_join(xgbpred, by = "Id") %>%
  mutate(Sales = 0.985 * (Sales.x + Sales.y) / 2) %>%
  select(Id, Sales)
write.csv(pred, '../output/mix6.csv', row.names = F)
## 0.10155 	0.11262 (66th best private )

## https://www.kaggle.com/c/rossmann-store-sales/forums/t/17601/correcting-log-sales-prediction-for-rmspe/99643#post99643

## mix models with comb
glmpred <- read.csv('../output/glmnet_comb.csv', as.is = TRUE)
xgbpred <- read.csv('../output/xgb4.csv', as.is = TRUE)
pred <- glmpred %>%
  inner_join(xgbpred, by = "Id") %>%
  mutate(Sales = 0.985 * (Sales.x + Sales.y) / 2) %>%
  select(Id, Sales)
write.csv(pred, '../output/mix_comb.csv', row.names = F)
## 0.10158 	0.11461 	

## corr glmnet

glmpred <- read.csv('../output/glmnet15.csv', as.is = TRUE)
pred <- glmpred %>% mutate(Sales = 0.985 * Sales)
write.csv(pred, '../output/glmnet15_corr.csv', row.names = F)
## 0.10979 	0.11974 (516th)
glmpred <- read.csv('../output/glmnet_comb.csv', as.is = TRUE)
pred <- glmpred %>% mutate(Sales = 0.985 * Sales)
write.csv(pred, '../output/glmnet_comb_corr.csv', row.names = F)
## 0.10910 	0.12372

## xgb long fold

xgb_features <- c("Store", "DayOfWeek", "Open", "Promo", "SchoolHoliday",
                  "StoreTypeN", "AssortmentN", "CompetitionDistance",
                  "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
                  "Promo2", "Promo2SinceWeek", "Promo2SinceYear",
                  "PromoIntervalN", "Month", "Year", "MDay",
                  decay_features, stairs_features, fourier_names)

## https://www.kaggle.com/abhilashawasthi/rossmann-store-sales/xgb-rossmann/run/86608
xparams <- list(
  objective = "reg:linear", 
  booster = "gbtree",
  eta = 0.02,
  max_depth = 10,
  subsample = 0.9,
  colsample_bytree = 0.7,
  silent = 1
)
nrounds <- 3000

fold <- train %>%
  filter(is.element(Store, unique(test$Store))) %>%
  make_fold(step = 1, predict_interval = 9*7)

train_tr <- fold$train %>%
  log_transform_train() %>%
  select_features(xgb_features)

test_tr <- select_features(fold$test, xgb_features)

validate_tr <- fold$actual %>%
  log_transform_train() %>%
  select_features(xgb_features) %>%
  select(-Id)

gc()
fold_pred_tr <- predict_xgboost(train_tr, test_tr, xgb_exp_rmspe, validate_tr, params = xparams, nrounds = nrounds)

fold_pred <- log_revert_predicted(fold_pred_tr$predicted)
rmspe(fold_pred$PredictedSales, fold$actual$Sales)

rmspe_pred <- rmspe_per_store(fold_pred, fold$actual)
qplot(x = rmspe_pred$rmspe, xlim = c(0.05, 0.25))
view_sales(train, fold_pred, stores = rmspe_pred[order(rmspe_pred$rmspe, decreasing=T),]$Store, log_sales = T)

importance_matrix <- xgb.importance(model = fold_pred_tr$fit)
xgb_features[as.integer(importance_matrix$Feature) + 1]

##  0.1134284


## submit xgb

xgb_features <- c("Store", "DayOfWeek", "Open", "Promo", "SchoolHoliday",
                  "StoreTypeN", "AssortmentN", "CompetitionDistance",
                  "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
                  "Promo2", "Promo2SinceWeek", "Promo2SinceYear",
                  "PromoIntervalN", "Month", "Year", "MDay",
                  decay_features, stairs_features, fourier_names)

## https://www.kaggle.com/abhilashawasthi/rossmann-store-sales/xgb-rossmann/run/86608
xparams <- list(
  objective = "reg:linear", 
  booster = "gbtree",
  eta = 0.02,
  max_depth = 10,
  subsample = 0.9,
  colsample_bytree = 0.7,
  silent = 1
)
nrounds <- 3000

train_tr <- train %>%
  remove_before_changepoint() %>%
  filter(is.element(Store, unique(test$Store))) %>%
  log_transform_train() %>%
  select_features(xgb_features)
test_tr <- select_features(test, xgb_features)
gc()

pred <- predict_xgboost(
  train_tr, test_tr, xgb_expm1_rmspe,
  params = xparams, nrounds = nrounds)$predicted %>% log_revert_predicted()

save_predicted('../output/xgb4.csv', pred)
## 0.11839 (379th)
view_sales(train, pred)

## cv per store small train combinations

rm(train, test)
gc()
with_id <- small_train %>%
  log_transform_train() %>%
  mutate(Id = 1:n()) %>%
  select_features(linear_features)

train_tr <- select(with_id, -Id)
test_tr <- select(with_id, -Sales)
actual <- mutate(with_id, Sales = exp(Sales))

pred <- predict_glmnet(train_tr, test_tr, exp_rmspe_corr)

pred$predicted <- log_revert_predicted(pred$predicted)

view_sales(actual, pred$predicted)

lapply(pred$fit, function(fit) predict(fit, type = "nonzero", s = fit$lambda[fit$best_lambda_idx]))

lapply(pred$fit, function(fit) predict(fit, type = "coefficients", s = fit$lambda[fit$best_lambda_idx]))

useful_features_idx <- sort(unique(unlist(lapply(pred$fit[c("388", "562")], function(fit) predict(fit, type = "nonzero", s = fit$lambda[fit$best_lambda_idx])))))

useful_features <- linear_features[useful_features_idx]

combined <- make_feature_combinations(filter(small_train, is.element(Store, c(388, 562))), useful_features)

with_id <- combined$data %>%
  log_transform_train() %>%
  mutate(Id = 1:n()) %>%
  select_features(c(linear_features, combined$combinations))

train_tr <- select(with_id, -Id)
test_tr <- select(with_id, -Sales)

pred <- predict_glmnet(train_tr, test_tr, exp_rmspe_corr)

useful_features_idx <- sort(unique(unlist(lapply(pred$fit[c("388", "562")], function(fit) predict(fit, type = "nonzero", s = fit$lambda[fit$best_lambda_idx])))))

useful_features <- c(linear_features, combined$combinations)[useful_features_idx]

saveRDS(useful_features, "../output/features.rds")

combination_features <- grep('_', useful_features, value = TRUE)
saveRDS(combination_features, "../output/combination-features.rds")

## small combinations

combination_features <- readRDS("../output/combination-features.rds")

train_tr <- small_train %>%
  log_transform_train() %>%
  make_selected_combinations(combination_features) %>%
  select_features(c(linear_features, combination_features))

per_store <- cv_glmnet_store(train_tr, exp_rmspe_corr, steps = 15, step_by = 3)

lapply(per_store, summary)

per_store <- cv_glmnet_store(train_tr, exp_rmspe, steps = 15, step_by = 3)

per_store <- cv_glmnet_store(train_tr, exp_rmspe_corr, steps = 15, step_by = 3, nlambda = 1000)

train_tr <- small_train %>%
  log_transform_train() %>%
  select_features(c(linear_features))

train_tr <- small_train %>%
  filter(Store == 851) %>%
  log_transform_train() %>%
  select_features(c(linear_features))

bgl <- best_glmnet_lambda(train_tr, exp_rmspe_corr, steps = 13, predict_interval = 6 * 7, step_by = 7, alpha = 1, family = "gaussian", nlambda = 100)

## cv correction

train_tr <- train %>%
  filter(is.element(Store, unique(test$Store))) %>%
  remove_before_changepoint() %>%
  log_transform_train() %>%
  remove_outliers_lm() %>%
  select_features(linear_features)
rm(train, test)
gc(reset = T)

summary(cv_glmnet(train_tr, exp_rmspe))
summary(cv_glmnet(train_tr, exp_rmspe_corr))
summary(cv_glmnet(train_tr, exp_rmspe, steps = 15, step_by = 3))
summary(cv_glmnet(train_tr, exp_rmspe_corr, steps = 15, step_by = 3))
summary(cv_glmnet(train_tr, exp_rmspe_corr, steps = 15, step_by = 3, nlambda = 1000))


## illustration
train_tr <- small_fold$train %>%
  log_transform_train() %>%
  remove_outliers_lm() %>%
  select_features(linear_features)
test_tr <- select_features(small_fold$test, linear_features)

pred <- predict_glmnet(train_tr, test_tr, exp_rmspe, steps = 15, step_by = 3)$predicted %>% log_revert_predicted()

view_sales(small_fold$train, pred, small_fold$actual, stores = 562)

## example cv

train_tr <- small_train %>%
  filter(Store == 851) %>%
  log_transform_train() %>%
  select_features(linear_features)
summary(cv_glmnet(train_tr, exp_rmspe, steps = 15, step_by = 3))
