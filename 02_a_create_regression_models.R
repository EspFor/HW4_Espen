
library(parsnip)
library(yardstick)

# xgboost -----------------------------------------------------------------

dplyr::count(ads, kommune_name, sort = TRUE)


xg_recipe <- recipe(ad_tot_price ~. , data = finn_train_raw) %>% 
  step_mutate(ad_home_type  = fct_lump(ad_home_type, 4),
              ad_owner_type = fct_lump(ad_owner_type, 3),
              ad_kommune_name = fct_lump(ad_kommune_name, 6),
              ad_energy = fct_lump(ad_energy, 7),
              bedrooms_missing = is.na(ad_bedrooms)) %>%
  step_rm(ad_id, 
          ad_price,
          ad_tot_price_per_sqm,
          ad_debt, 
          kommune_no, 
          fylke_name, 
          zip_no, 
          zip_name) %>%
  prep()

finn_train <- bake(xg_recipe, finn_train_raw)
finn_test  <- bake(xg_recipe, finn_test_raw)

#Hyperparameter tuning
#Finner først en bra learning rate

hyper_grid <- expand.grid(
  learning_rate = c(0.3, 0.1, 0.05, 0.01, 0.005),
  MAPE = NA,
  trees = NA,
  time = NA
)

for(i in seq_len(nrow(hyper_grid))) {
  
  set.seed(42)
  
  train_time <- system.time({
    xg_mod <- boost_tree(mode = "regression",
                         trees = 100,
                         min_n = 12,
                         tree_depth = 6,
                         learn_rate = hyper_grid$learning_rate[i],
                         loss_reduction = 0.9) %>% 
      set_engine("xgboost", tree_method = "exact") %>% 
      fit(ad_tot_price ~ ., data = finn_train)
    
    prediction <- predict(xg_mod, finn_test) %>% 
      bind_cols(finn_test_raw) %>% 
      dplyr::rename(estimate = .pred,
                    truth = ad_tot_price) %>% 
      mutate(abs_dev      = abs(truth - estimate),
             abs_dev_perc = abs_dev/truth)
  })
  
  hyper_grid$MAPE[i]  <- mean(abs((prediction$truth-prediction$estimate)/prediction$truth) * 100)
  hyper_grid$time[i]  <- train_time[["elapsed"]]
  
}

arrange(hyper_grid, MAPE) #Setter learn_rate = 0.3
sum(hyper_grid$time)

#Ny search-grid for å optimalisere andre variabler
hyper_grid <- expand.grid(
  learning_rate = 0.1,
  trees = c(5, 50, 100, 200, 500),
  min_n = c(3,6,9, 12),
  tree_depth = c(3,6, 10),
  loss_reduction = c(0.6, 0.7, 0.8, 0.9),
  RMSE = NA,
  time = NA
)

for(i in seq_len(nrow(hyper_grid))) {
  
  set.seed(42)
  
  train_time <- system.time({
    xg_mod <- boost_tree(mode = "regression",
                         trees = hyper_grid$trees[i],
                         min_n = hyper_grid$min_n[i],
                         tree_depth = hyper_grid$tree_depth[i],
                         learn_rate = hyper_grid$learning_rate[i],
                         loss_reduction = hyper_grid$loss_reduction[i]) %>% 
      set_engine("xgboost", tree_method = "exact") %>% 
      fit(ad_tot_price ~ ., data = finn_train)
    
    prediction <- predict(xg_mod, finn_test) %>% 
      bind_cols(finn_test_raw) %>% 
      dplyr::rename(estimate = .pred,
                    truth = ad_tot_price) %>% 
      mutate(abs_dev      = abs(truth - estimate),
             abs_dev_perc = abs_dev/truth)
  })
  
  hyper_grid$RMSE[i]  <- sqrt(min(prediction$abs_dev))
  hyper_grid$trees[i] <- which.min(prediction$abs_dev)
  hyper_grid$time[i]  <- train_time[["elapsed"]]
  
}

arrange(hyper_grid, RMSE)
sum(hyper_grid$time)

#Ny search-grid for å optimalisere andre variabler
hyper_grid <- expand.grid(
  learning_rate = 0.3,
  trees = c(5, 50, seq(from = 100, to = 1100, by = 200)),
  min_n = seq(from = 2, to = 20, by = 3),
  tree_depth = c(6,10,12,15),
  loss_reduction = 0.9, #seq(from = 0.5, to = 1, by = 0.1),
  MAPE = NA,
  time = NA
)

for(i in seq_len(nrow(hyper_grid))) {
  
  set.seed(42)
  
  train_time <- system.time({
    xg_mod <- boost_tree(mode = "regression",
                         trees = hyper_grid$trees[i],
                         min_n = hyper_grid$min_n[i],
                         tree_depth = hyper_grid$tree_depth[i],
                         learn_rate = hyper_grid$learning_rate[i],
                         loss_reduction = hyper_grid$loss_reduction[i]) %>% 
      set_engine("xgboost", tree_method = "exact") %>% 
      fit(ad_tot_price ~ ., data = finn_train)
    
    prediction <- predict(xg_mod, finn_test) %>% 
      bind_cols(finn_test_raw) %>% 
      dplyr::rename(estimate = .pred,
                    truth = ad_tot_price) %>% 
      mutate(abs_dev      = abs(truth - estimate),
             abs_dev_perc = abs_dev/truth)
  })
  
  hyper_grid$MAPE[i]  <- mean(abs((prediction$truth-prediction$estimate)/prediction$truth) * 100)
  hyper_grid$time[i]  <- train_time[["elapsed"]]
  
}

arrange(hyper_grid, MAPE)
sum(hyper_grid$time)

prediction <- predict(xg_mod, finn_test) %>% 
  bind_cols(finn_test_raw) %>% 
  dplyr::rename(estimate = .pred,
                truth = ad_tot_price) %>% 
  mutate(abs_dev      = abs(truth - estimate),
         abs_dev_perc = abs_dev/truth)

prediction %>%
  multi_metric(truth = truth, estimate = estimate)

# Get variable importance:
xgboost::xgb.importance(model = xg_mod$fit) %>% 
  xgboost::xgb.ggplot.importance()

# Check out a particular tree:
xgboost::xgb.plot.tree(model = xg_mod$fit, trees = 50)

# Check distribution of predicted vs truth
prediction %>% 
  select(estimate, truth) %>% 
  rownames_to_column(var = "id") %>% 
  pivot_longer(-id, names_to = "type", values_to = "value") %>% 
  ggplot(aes(x = value, fill = type)) +
  geom_density(alpha = 0.3)

prediction %>% 
  select(estimate, truth, fylke_name) %>% 
  rownames_to_column(var = "id") %>% 
  pivot_longer(-c(id, fylke_name), names_to = "type", values_to = "value") %>% 
  ggplot(aes(x = value, fill = type)) +
  geom_density(alpha = 0.3) +
  facet_wrap(~fylke_name)


