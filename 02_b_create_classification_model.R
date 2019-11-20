library(parsnip)
library(yardstick)

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

prediction <- predict(glm_mod, finn_test, type = "prob") %>% 
  bind_cols(finn_test) %>% 
  dplyr::rename(estimate     = .pred_TRUE, 
                truth        = is_expensive)

# Evaluate model (NOTE: we need different metrics since this is classification!)
prediction %>%
  yardstick::roc_auc(truth, estimate)

prediction %>%
  yardstick::roc_curve(truth = truth, estimate = estimate, na_rm = T) %>% 
  autoplot()
