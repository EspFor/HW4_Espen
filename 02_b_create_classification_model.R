library(parsnip)
library(yardstick)
library(glmnet)
library(dplyr)

# Create classification model
glm_recipe <- recipe(ad_tot_price ~. , data = finn_train_raw) %>% 
  step_mutate(ad_home_type  = fct_lump(ad_home_type, 4),
              ad_owner_type = fct_lump(ad_owner_type, 2),
              kommune_name = fct_lump(kommune_name, 6),
              bedrooms_missing = is.na(ad_bedrooms),
              is_expensive = as.factor(ad_tot_price > 4000000)) %>%
  step_medianimpute(all_numeric()) %>%
  step_modeimpute(all_nominal()) %>% 
  step_rm(ad_id) %>% 
  prep()

finn_train <- bake(glm_recipe, finn_train_raw)
finn_test  <- bake(glm_recipe, finn_test_raw)

glm_mod <- logistic_reg() %>%
  set_engine("glm") %>%
  fit(
    is_expensive ~ 
      ad_owner_type
    + ad_home_type
    + ad_bedrooms
    + ad_sqm
    + ad_sqm_use
    + ad_expense
    + avg_income
    + ad_built
    + bedrooms_missing
    + fylke_name
    + kommune_name
    + ad_floor
    + ad_energy
    + ad_garage
    + ad_balcony
    + ad_elevator
    + ad_view
    + ad_garden
    + ad_fireplace,
    data = finn_train
  )

# View summary
summary(glm_mod$fit)


x <- model.matrix(is_expensive~., finn_train)[,-1]
y <- as.numeric(finn_train$is_expensive)-1

cv.lasso <- cv.glmnet(x, y, alpha = 1, family = "binomial")
model <- glmnet(x, y, alpha = 1, family = "binomial",
                lambda = cv.lasso$lambda.min)
coef(model)
x.test <- model.matrix(is_expensive ~., finn_test)[,-1]
probabilities <- model %>% predict(newx = x.test)
predicted.classes <- ifelse(probabilities > 0.5, 1, 0)
observed.classes <- as.numeric(finn_test$is_expensive)-1
mean(predicted.classes == observed.classes)





x_glmnet <- finn_train %>%
  select(
    ad_owner_type,
    ad_home_type,
    ad_bedrooms,
    ad_sqm,
    ad_sqm_use,
    ad_expense,
    avg_income,
    ad_built,
    bedrooms_missing,
    fylke_name,
    kommune_name,
    ad_floor,
    ad_energy,
    ad_garage,
    ad_balcony,
    ad_elevator,
    ad_view,
    ad_garden,
    ad_fireplace
  ) %>% 
  data.matrix()

lambdas <- 10^seq(3, -2, by = -.1)

y <- as.numeric(finn_train$is_expensive)-1

fit_glmnet <- glmnet(x = x_glmnet,y = y, aplha = 0, lamba = lambdas, family = "binomial")
summary(fit_glmnet)

opt_lambda <- fit_glmnet$lambda.min
opt_lambda

fit <- fit_glmnet$glmnet.fit
summary(fit)

is_expensive_predicted <- predict(fit, s = opt_lambda, newx = x_glmnet)
is_expensive_predicted

sst <- sum((y - mean(y))^2)
sse <- sum((is_expensive_predicted - y)^2)

# R squared
rsq <- 1 - (sse / sst)
rsq

is_expensive_predicted %>%
  yardstick::roc_auc(y, is_expensive_predicted)


prediction <- predict(glm_mod, finn_test, type = "prob") %>% 
  bind_cols(finn_test) %>% 
  dplyr::rename(estimate     = .pred_TRUE, 
                truth        = is_expensive)

# Evaluate model (NOTE: we need different metrics since this is classification!)
prediction %>%
  yardstick::roc_auc(truth, estimate)

#AUC før endringer: 0.817 (dobbelsjekk)
#AUC etter endringer: 0.881

prediction %>%
  yardstick::roc_curve(truth = truth, estimate = estimate, na_rm = T) %>% 
  autoplot()
