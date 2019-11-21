library(parsnip)
library(yardstick)
library(glmnet)
library(dplyr)
library(recipes)
library(forcats)

# Create classification model
glm_recipe <- recipe(ad_tot_price ~. , data = finn_train_raw) %>% 
  step_mutate(ad_home_type  = fct_lump(ad_home_type, 4),
              ad_owner_type = fct_lump(ad_owner_type, 2),
              bedrooms_missing = is.na(ad_bedrooms),
              is_expensive = as.factor(ad_tot_price > 4000000)) %>%
  step_medianimpute(all_numeric()) %>%
  step_modeimpute(all_nominal()) %>% 
  step_rm(ad_id) %>% 
  prep()

finn_train2 <- bake(glm_recipe, finn_train_raw)
finn_test2  <- bake(glm_recipe, finn_test_raw)

x <- model.matrix(is_expensive~., finn_train2)[,-1]
y <- as.numeric(finn_train2$is_expensive)-1

cv.lasso <- cv.glmnet(x, y, alpha = 1, family = "binomial")
model <- glmnet(x, y, alpha = 1, family = "binomial",lambda = cv.lasso$lambda.min)

x.test <- model.matrix(is_expensive ~., finn_test2)[,-1]
probabilities <- model %>% predict(newx = x.test)
predicted.classes <- ifelse(probabilities > 0.5, 1, 0)
observed.classes <- as.numeric(finn_test2$is_expensive)-1
mean(predicted.classes == observed.classes)


is_expensive_predicted %>%
  yardstick::roc_auc(y, is_expensive_predicted)

prediction <- predict(glm_mod, finn_test2, type = "prob") %>% 
  bind_cols(finn_test2) %>% 
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
