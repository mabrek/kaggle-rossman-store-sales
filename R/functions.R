source("libraries.R")

## predict_* functions take train and test dataframes (as in csv) and return test dataframe with PredictedSales column added in $predicted field of a list

## adapted from https://www.kaggle.com/shearerp/rossmann-store-sales/store-dayofweek-promo-0-13952
predict_geometric_mean <- function(train, test) {
  train <- train[train$Sales > 0, ]
  preds<-c('Store', 'DayOfWeek', 'Promo')
  mdl <- train %>% group_by_(.dots = preds) %>% summarise(PredictedSales = exp(mean(log(Sales)))) %>% ungroup()
  pred <- test %>% left_join(mdl, by = preds)
  pred$PredictedSales[is.na(pred$PredictedSales)] <- 0
  list(predicted = pred)
}

## adapted from https://www.kaggle.com/shearerp/rossmann-store-sales/interactive-sales-visualization
predict_median <- function(train, test) {
  train <- train[train$Sales > 0, ]
  preds<-c('Store', 'DayOfWeek', 'Promo')
  mdl <- train %>% group_by_(.dots = preds) %>% summarise(PredictedSales = median(Sales)) %>% ungroup()
  pred <- test %>% left_join(mdl, by = preds)
  pred$PredictedSales[is.na(pred$PredictedSales)] <- 0
  list(predicted = pred)
}

checking_mclapply <- function(...) {
  lapply(mclapply(...), function(r) {
    if (inherits(r, "try-error")) {
      print(r)
      stop("see mclapply error above")
    } else {
      r
    }
  })
}

log_transform_train <- function (train) {
  train %>%
    filter(Sales > 0) %>%
    mutate(Sales = log(Sales))
}

log_revert_predicted <- function (predicted) {
  mutate(predicted, PredictedSales = ifelse(Open == 1, exp(PredictedSales), 0))
}

log1p_transform_train <- function (train) {
  train %>%
    filter(Sales > 0) %>%
    mutate(Sales = log1p(Sales))
}

log1p_revert_predicted <- function (predicted) {
  mutate(predicted,
         PredictedSales = ifelse(Open == 1, expm1(PredictedSales), 0))
}

select_features <- function (train, features) {
  select(train, matches("^Sales$"), matches("^Id$"),
         one_of(c("Store", "Date", features)))
}

predict_lm <- function (train, test, save_fit = FALSE) {
  predict_per_store(train, test, function(store_train, store_test) {
    fit <- lm(Sales ~ ., select(store_train, -Date))
    pred <- predict.lm(fit, store_test)
    predicted <- store_test %>% mutate(PredictedSales = pred)
    if (save_fit) {
      list(predicted = predicted, fit = fit)
    } else {
      list(predicted = predicted, fit = NULL)
    }
  })
}

predict_glm <- function (train, test, features = glm_features) {
  predict_per_store(train, test, function(store_train, store_test) {
    store_train <- store_train %>%
      filter(Sales > 0) %>%
      select(one_of(c("Sales", features)))
    fit <- glm(Sales ~ ., store_train, family=poisson())
    pred <- predict(fit, store_test, type = "response")
    predicted <- store_test %>%
      mutate(PredictedSales = ifelse(Open == 1, pred, 0))
    list(predicted = predicted, fit = fit)
  })
}

predict_svr <- function(train, test, features = linear_features,
                        kernel = "linear", ...) {
  predict_per_store(train, test, function(store_train, store_test) {
    store_train <- store_train %>%
      filter(Sales > 0) %>%
      select(one_of(c("Sales", features))) %>%
      mutate(Sales = log(Sales))
    fit <- svm(Sales ~ ., store_train, fitted = FALSE, kernel = kernel, ...)
    pred <- predict(fit, store_test)
    predicted <- store_test %>%
      mutate(PredictedSales = ifelse(Open == 1, exp(pred), 0))
    list(predicted = predicted, fit = fit)
  })
}

predict_glmnet <- function (train, test, eval_function, steps = 13,
                            predict_interval = 6 * 7, step_by = 7,
                            alpha = 1, family = c("gaussian", "poisson"),
                            nlambda = 100) {
  family = match.arg(family)
  predict_per_store(train, test, function(store_train, store_test) {
    bgl <- best_glmnet_lambda(store_train = store_train,
                              eval_function = eval_function,
                              steps = steps,
                              predict_interval = predict_interval,
                              step_by = step_by,
                              alpha = alpha, family = family, nlambda = nlambda)
    fit <- bgl$global_fit
    lambda <- fit$lambda[bgl$best_lambda_idx]
    pred <- predict(fit, as.matrix(select(store_test, -Id, -Date)),
                    type = "response", s = lambda)
    predicted <- mutate(store_test, PredictedSales = pred)
    list(predicted = predicted, fit = fit)
  })
}

predict_per_store <- function(train, test, predict_fun) {
  rlist <- checking_mclapply(unique(test$Store), function(store) {
    store_train <- train %>% filter(Store == store)
    store_test <- test %>% filter(Store == store)
    pr <- predict_fun(store_train, store_test)
    list(predicted = pr$predicted, fit = pr$fit, store = store)
  }, mc.allow.recursive = FALSE, mc.preschedule = TRUE)
  preds <- bind_rows(lapply(rlist, function(l) {l$predicted}))
  fits <- lapply(rlist, function(l) {l$fit})
  names(fits) <- sapply(rlist, function(l) {l$store})
  list(predicted = preds, fit = fits)
}

xgb_exp_rmspe <- function(predicted, dtrain) {
  list(metric = "rmspe",
       value = rmspe(exp(predicted), exp(getinfo(dtrain, "label"))))
}

xgb_expm1_rmspe <- function(predicted, dtrain) {
  list(metric = "rmspe",
       value = rmspe(expm1(predicted), expm1(getinfo(dtrain, "label"))))
}

predict_xgboost <- function(train, test, eval_function,
                            validate = NULL, ...) {
  dtrain <- xgb.DMatrix(data.matrix(select(train, -Sales, -Date)),
                        label = train$Sales)
  dtest <- xgb.DMatrix(data.matrix(select(test, -Id, -Date)))
  if (!is.null(validate)) {
    watchlist <- list(
      validate = xgb.DMatrix(data.matrix(select(validate, -Sales, -Date)),
                             label = validate$Sales),
      train = dtrain)
  } else {
    watchlist <- list(train = dtrain)
  }
  fit <- xgb.train(data = dtrain,
                   feval = eval_function,
                   early.stop.round = 100,
                   maximize = FALSE,
                   watchlist = watchlist,
                   verbose = 1,
                   print.every.n = 100, ...)
  pred <- predict(fit, dtest)
  predicted <- mutate(test, PredictedSales = pred)
  list(predicted = predicted, fit = fit)
}

save_predicted <- function(file_name, predicted) {
  pred <- predicted %>%
    select(Id, PredictedSales) %>%
    rename(Sales = PredictedSales)
  write.csv(pred, file_name, row.names = F)
}

rmspe <- function(predicted, actual) {
  if (length(predicted) != length(actual))
    stop("predicted and actual have different lengths")
  idx <- actual > 0
  sqrt(mean(((actual[idx] - predicted[idx]) / actual[idx]) ^ 2))
}

exp_rmspe <- function(predicted, actual) {
  rmspe(exp(predicted), exp(actual))
}

exp_rmspe_corr <- function(predicted, actual) {
  rmspe(exp(predicted) * 0.985, exp(actual))
}

predict_range_debug <- function(train, test) {
  print("train range:")
  print(range(train$Date))
  print("test range:")
  print(range(test$Date))
}

make_fold <- function(train, step = 1, predict_interval = 6 * 7,
                      step_by = 7) {
  date_range <- range(train$Date)
  dates <- seq(date_range[1], date_range[2], by = "DSTday")
  total <- length(dates)
  last_train <- total - predict_interval - (step - 1) * step_by
  last_train_date <- dates[last_train]
  last_predict <- last_train + predict_interval
  last_predict_date <- dates[last_predict]
  train_set <- train %>% filter(Date <= last_train_date)
  actual <- train %>%
    filter(Date > last_train_date, Date <= last_predict_date) %>%
    mutate(Id = 1:n())
  test_set <- select(actual, -Sales)
  list(train = train_set, test = test_set, actual = actual)
}

cross_validate <- function(train, predict_function, steps = 13,
                           predict_interval = 6 * 7, step_by = 7,
                           verbose = TRUE, ...) {
  if (verbose) cat("predict_interval =", predict_interval,
                   "step_by =", step_by, "steps=", steps, "\n")
  simplify2array(checking_mclapply(1 : steps, function(step) {
    fold <- make_fold(train, step, predict_interval, step_by)
    predicted <- predict_function(fold$train, fold$test, ...)$predicted
    sales <- fold$actual %>%
      left_join(predicted, by = "Id") %>%
      select(PredictedSales, Sales)
    r <- rmspe(sales$PredictedSales, sales$Sales)
    if (verbose) cat("step:", step, "rmspe:", r, "\n")
    r
  }, mc.allow.recursive = FALSE))
}

cv_glmnet <- function(train, eval_function, steps = 13,
                      predict_interval = 6 * 7, step_by = 7,
                      alpha = 1, family = c("gaussian", "poisson"),
                      nlambda = 100) {
  family = match.arg(family)

  rlist <- checking_mclapply(unique(train$Store), function(store) {
    store_train <- train %>% filter(Store == store)
    bgl <- best_glmnet_lambda(store_train = store_train,
                              eval_function = eval_function, steps = steps,
                              predict_interval = predict_interval,
                              step_by = step_by, alpha = alpha,
                              family = family, nlambda = nlambda)
    per_step <- bgl$per_step
    best_lambda_idx <- bgl$best_lambda_idx
    bind_rows(
      lapply(
        1:steps,
        function(step)
          data_frame(predicted = per_step[[step]]$predictions[,best_lambda_idx],
                     actual = per_step[[step]]$actual,
                     step = step)))
  }, mc.allow.recursive = FALSE, mc.preschedule = TRUE)

  (bind_rows(rlist) %>% group_by(step) %>%
   summarise(err = eval_function(predicted, actual)))$err
}

cv_glmnet_store <- function(train, eval_function, steps = 13,
                      predict_interval = 6 * 7, step_by = 7,
                      alpha = 1, family = c("gaussian", "poisson"),
                      nlambda = 100) {
  family = match.arg(family)

  rlist <- checking_mclapply(unique(train$Store), function(store) {
    store_train <- train %>% filter(Store == store)
    bgl <- best_glmnet_lambda(store_train = store_train,
                              eval_function = eval_function, steps = steps,
                              predict_interval = predict_interval,
                              step_by = step_by, alpha = alpha,
                              family = family, nlambda = nlambda)
    per_step_scores <- sapply(bgl$per_step, function(l) l$scores)
    best_lambda_idx <- bgl$best_lambda_idx
    list(store = store, scores = per_step_scores[best_lambda_idx, ])
  }, mc.allow.recursive = FALSE, mc.preschedule = TRUE)
  
  scores <- lapply(rlist, function(l) l$scores)
  names(scores) <- sapply(rlist, function(l) l$store)
  scores
}

best_glmnet_lambda <- function(store_train, eval_function, steps,
                               predict_interval, step_by,
                               alpha, family,
                               nlambda) {
    global_fit <- glmnet(as.matrix(select(store_train, -Sales, -Date)),
                         store_train$Sales, family = family, alpha = alpha,
                         nlambda = nlambda)
    lambdas <- global_fit$lambda
    
    per_step <- lapply(1:steps, function(step) {
      fold <- make_fold(store_train, step, predict_interval, step_by)
      fold_fit <- glmnet(as.matrix(select(fold$train, -Sales, -Date)),
                         fold$train$Sales, family = family,
                         alpha = alpha, lambda = lambdas)
      predictions <- predict(fold_fit,
                             as.matrix(select(fold$test, -Id, -Date)),
                             type = "response")
      if (nrow(fold$test) < 2) {
        predictions <- matrix(0, nrow = nrow(predictions), ncol = ncol(predictions))
      }
      scores <- apply(predictions, 2, function(p) {
        eval_function(p, fold$actual$Sales)
      })
      list(predictions = predictions,
           actual = fold$actual$Sales,
           scores = scores,
           fit = fold_fit)
    })

    best_lambda_idx <-
      order(apply(sapply(per_step, function(l) l$scores), 1, median))[1]

    global_fit$best_lambda_idx <- best_lambda_idx

    list(global_fit = global_fit, per_step = per_step,
         best_lambda_idx = best_lambda_idx)
}

rmspe_per_store <- function(predicted, actual) {
  actual %>% 
    left_join(select(predicted, Id, Store, PredictedSales),
              by = c("Id", "Store")) %>% 
    select(Store, PredictedSales, Sales) %>%
    group_by(Store) %>%
    summarise(rmspe = rmspe(PredictedSales, Sales))
}

set_cores <- function(cores = detectCores()) {
  options(mc.cores = cores)
}

lsd <- function(pos = 1) {
  names(grep("^function$", 
             sapply(ls(pos = pos), function(x) {mode(get(x))}),
             value = T, 
             invert = T))
}

view_sales <- function(train, predicted = NULL, actual = NULL, stores = NULL,
                       hide_zero = log_sales, log_sales = FALSE) {
  data <- select(train, Date, Store, Sales)
  if (!is.null(predicted)) {
    predicted <- select(predicted, Id, Date, Store, PredictedSales)
  }
  if (!is.null(actual)) {
    actual <- select(actual, Id, Date, Store, Sales)
  }
  if (!is.null(predicted) && !is.null(actual)) {
    data <- full_join(data,
                      full_join(actual, predicted,
                                by = c("Id", "Date", "Store")),
                      by = c("Date", "Store")) %>%
      mutate(Sales = ifelse(is.na(Sales.x), Sales.y, Sales.x)) %>%
      select(-Sales.x, -Sales.y)
  } else {
    if (!is.null(predicted)) {
      data <- full_join(data, predicted, by = c("Date", "Store"))
    }
    if (!is.null(actual)) {
      data <- full_join(data, actual, by = c("Date", "Store"))
    }
  }
  if (is.null(stores)) {
    if (!is.null(predicted)) stores <- unique(predicted$Store)
    else stores <- unique(train$Store)
  }
  if (length(stores) > 50) stores <- stores[1:50]
  app <- shinyApp(
    ui = fluidPage(
      lapply(
        1:length(stores),
        function(n) {
          fluidRow(
            style = "padding-bottom: 5px;",
            column(
              dygraphOutput(paste("graph_series_", n, sep = ""),
                            height = "300px"),
              width = 11),
            column(
              textOutput(paste("text_series_", n, sep = "")),
              width = 1))
        })
      ),
    server = function(input, output) {
      lapply(
        1:length(stores),
        function(n) {
          single <- data %>% filter(Store == stores[n])
          if (hide_zero) {
            single <- filter(single, Sales != 0)
          }
          if (log_sales) {
            single <- mutate(single, Sales = log(Sales))
          }
          graph_ts <- cbind(sales = xts(single$Sales, single$Date))
          if (!is.null(single$PredictedSales) &&
               any(!is.na(single$PredictedSales))) {
            if (log_sales) {
              single <- mutate(single, PredictedSales = log(PredictedSales),
                               Error = 10 * (PredictedSales - Sales))
            } else {
              single <- mutate(single, Error = PredictedSales - Sales)
            }
            graph_ts <- cbind(graph_ts,
                              predicted = xts(single$PredictedSales,
                                              single$Date),
                              error = xts(single$Error, single$Date))
          }
          output[[paste("graph_series_", n, sep = "")]] <- renderDygraph({
            d <- dygraph(graph_ts, group = "series") %>%
              dySeries('sales', color = 'blue') %>%
              dyOptions(drawXAxis = (n %% 9 == 0) | (n == length(stores)),
                        drawPoints = TRUE, pointSize = 2, 
                        connectSeparatedPoints = FALSE, drawGapEdgePoints = TRUE)
            if (is.element("predicted", colnames(graph_ts))) {
              d <- dySeries(d, 'predicted', color = 'green')
              d <- dySeries(d, 'error', color = 'red')
            }
            d
          })
          output[[paste("text_series_", n, sep = "")]] <- renderText({
            stores[n]
          })
        })
    })
  runApp(app)  
}

eu_wday <- function(date) {
  (wday(date) + 5) %% 7 + 1
}

zero_na <- function(x) {
  ifelse(is.na(x), 0, x)
}

max_2 <- function(x, y) {
  ifelse(x > y, x, y)
}

make_decay_features <- function(data, promo_after, promo2_after,
                                holiday_b_before,
                                holiday_c_before, holiday_c_after) {
  data <- data %>%
    mutate(
      PromoDecay = Promo *
        max_2(promo_after - as.integer(difftime(Date,
                                                PromoStartedLastDate,
                                                unit = 'days')), 0)) %>%
    mutate(StateHolidayCLastDecay =
             (1 - StateHolidayC) *
             max_2(holiday_c_after -
                   as.integer(
                     difftime(Date, StateHolidayCLastDate, unit = 'days')),
                   0)) %>%
    mutate(
      Promo2Decay = Promo2Active *
        max_2(promo2_after - as.integer(difftime(Date,
                                                 Promo2StartedDate,
                                                 unit = 'days')), 0)) %>%
    mutate(StateHolidayBNextDecay =
             (1 - StateHolidayB) *
             max_2(holiday_b_before -
                   as.integer(
                     difftime(StateHolidayBNextDate, Date, unit = 'days')),
                   0)) %>%
    mutate(StateHolidayCNextDecay =
             (1 - StateHolidayC) *
             max_2(holiday_c_before -
                   as.integer(
                     difftime(StateHolidayCNextDate, Date, unit = 'days')),
                   0))
}

make_after_stairs <- function(data, date_features, days) {
  for (f in date_features) {
    since <- difftime(data$Date, data[[f]], unit = 'days')
    for (d in days) {
      data[, paste(f, d, "after", sep = "")] <-
        as.integer(since >= 0 & since < d)
    }
  }
  data
}

make_before_stairs <- function(data, date_features,
                               days = c(2, 3, 4, 5, 7, 14, 28)) {
  for (f in date_features) {
    before <- difftime(data[[f]], data$Date, unit = 'days')
    for (d in days) {
      data[, paste(f, d, "before", sep = "")] <-
        as.integer(before >= 0 & before < d)
    }
  }
  data
}

scale_log_features <- function(data, features) {
  for (f in features) {
    data[, f] <- data[, f] / max(data[, f])
    data[, paste(f, "Log", sep = "")] <- log1p(data[, f])
    data[, paste(f, "Log", sep = "")] <- data[, paste(f, "Log", sep = "")] /
      max(data[, paste(f, "Log", sep = "")])
  }
  data
}

make_feature_combinations <- function(data, features) {
  cs <- t(combn(setdiff(features,
                        c(log_decay_features, "StateHolidayA", "StateHolidayB",
                          "StateHolidayC")),
                2))
  
  fourier <- grepl("Fourier", cs[, 1], fixed = T) &
    grepl("Fourier", cs[, 2], fixed = T)
  cs <- cs[!fourier, ]
  
  cnames <- paste(cs[, 1], cs[, 2], sep = "_")
  res_names <- c()
  for (i in 1:nrow(cs)) {
    feature <- data[, cs[i, 1]] * data[, cs[i, 2]]
    r <- range(feature)
    if (r[1] != r[2] && !identical(feature, data[, cs[i, 1]]) &&
        !identical(feature, data[, cs[i, 2]])) {
      data[, cnames[i]] <- feature
      res_names <- append(res_names, cnames[i])
    }
  }
  list(data = data, combinations = res_names)
}

make_selected_combinations <- function(data, combinations) {
  for (i in 1:length(combinations)) {
    cs <- strsplit(combinations[i], "_")[[1]]
    data[, combinations[i]] <- data[, cs[1]] * data[, cs[2]]
  }
  data
}

make_month_day <- function(data) {
 for (d in 1:31) {
    data[, paste("MDay", d, sep = "")] <- as.integer(mday(data$Date) == d)
  }
  data
}

make_month <- function(data) {
 for (m in 1:12) {
    data[, paste("Month", m, sep = "")] <- as.integer(month(data$Date) == m)
  }
  data
}

grid_search <- function(train, cv_function,
                        promo_after, promo2_after,
                        holiday_b_before,
                        holiday_c_before, holiday_c_after, ...) {

  space <- expand.grid(promo_after = promo_after, promo2_after = promo2_after,
                       holiday_b_before = holiday_b_before,
                       holiday_c_before = holiday_c_before,
                       holiday_c_after = holiday_c_after, ...,
                       stringsAsFactors = FALSE) %>%
    arrange(promo_after + promo2_after + holiday_b_before +
            holiday_c_before + holiday_c_after,
            mapply(max, promo_after, promo2_after,
                   holiday_b_before, holiday_c_before, holiday_c_after))

  cv_space <- select_(space,
                      .dots = c("-promo_after", "-promo2_after",
                                "-holiday_b_before", "-holiday_c_before",
                                "-holiday_c_after"))

  cv <- space %>% mutate(mean = 100, sd = 100)
  last_min <- 1
  checkpoint_time <- Sys.time()
  for (i in 1:nrow(space)) {
    train_variant <- make_decay_features(train,
                                         space[i, "promo_after"],
                                         space[i, "promo2_after"],
                                         space[i, "holiday_b_before"],
                                         space[i, "holiday_c_before"],
                                         space[i, "holiday_c_after"])
    cvs <- do.call(cv_function, c(list(train = train_variant),
                                  cv_space[i,]))
    cv[i, "mean"] <- mean(cvs)
    cv[i, "sd"] <- sd(cvs)
    if (cv[i, "mean"] < cv[last_min, "mean"]) {
      last_min <- i
      cat("last min mean", cv[i, "mean"], "sd", cv[i, "sd"], "\n")
      print(space[i,])
    }
    if (difftime(Sys.time(), checkpoint_time, unit = "mins") >= 15) {
      saveRDS(cv, paste("../output/gs-", i, ".rds", sep = ""))
      checkpoint_time <- Sys.time()
    }
  }
  cv
}

coef_glmnet <- function(fit_list, idx) {
  coef(fit_list[[idx]], s = fit_list[[idx]]$lambda[fit_list[[idx]]$best_lambda_idx])
}

cross_paste <- function(...) {
  args <- lapply(list(...), as.character)
  cross <- expand.grid(args, stringsAsFactors = FALSE)
  apply(cross, 1, paste, sep = "", collapse = "")
}

remove_outliers_lm <- function(train, features = log_lm_features,
                               z_score = 2.5) {
  with_id <- mutate(train, Id = 1:n())
  with_fit <- predict_lm(select_features(with_id, log_lm_features),
                         with_id)$predicted
  with_fit %>%
    group_by(Store) %>%
    mutate(Error = abs(PredictedSales - Sales),
           ZScore = Error / median(Error)) %>%
    filter(ZScore < 2.5) %>%
    select(-Id, -Error, -ZScore) %>%
    ungroup
}

remove_before_changepoint <- function(train) {
  train %>%
    filter(Store != 837 | Date > "2014-03-16") %>%
    filter(Store != 700 | Date > "2014-01-03") %>%
    filter(Store != 681 | Date > "2013-06-14") %>%
    filter(Store != 986 | Date > "2013-05-22") %>%
    filter(Store != 885 | Date > "2014-05-18") %>%
    filter(Store != 589 | Date > "2013-05-27") %>%
    filter(Store != 105 | Date > "2013-05-20") %>%
    filter(Store != 663 | Date > "2013-10-06") %>%
    filter(Store != 764 | Date > "2013-04-24") %>%
    filter(Store != 364 | Date > "2013-05-31") %>%
    filter(Store != 969 | Date > "2013-03-10") %>%
    filter(Store != 803 | Date > "2014-01-07") %>%
    filter(Store != 91 | Date > "2014-01-14")
}
