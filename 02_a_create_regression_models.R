
library(parsnip)
library(yardstick)

# xgboost -----------------------------------------------------------------

dplyr::count(ads, kommune_name, sort = TRUE)


xg_recipe <- recipe(ad_tot_price ~. , data = finn_train_raw) %>% 
  step_mutate(ad_home_type  = fct_lump(ad_home_type, 4),
              ad_owner_type = fct_lump(ad_owner_type, 3),
              kommune_name = fct_lump(kommune_name, 6),
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

#Kode for å teste MAPE for default values

# xg_mod <- boost_tree(mode = "regression",
#                      trees = 300,
#                      min_n = 2,
#                      tree_depth = 6,
#                      learn_rate = 0.15,
#                      loss_reduction = 0.9) %>% 
#   set_engine("xgboost", tree_method = "exact") %>% 
#   fit(ad_tot_price ~ ., data = finn_train)
# 
# 
# prediction <- predict(xg_mod, finn_test) %>% 
#   bind_cols(finn_test_raw) %>% 
#   dplyr::rename(estimate     = .pred, 
#          truth        = ad_tot_price) %>% 
#   dplyr::mutate(abs_dev      = abs(truth - estimate),
#          abs_dev_perc = abs_dev/truth)
# 
# multi_metric <- metric_set(mape, rmse, mae, rsq)
# 
# prediction %>%
#   multi_metric(truth = truth, estimate = estimate)

#MAPE med default databehandling og før hyperparameter-tuning: 22.7 (dobbelsjekk)
#MAPE med ytterligere databehandling, men før hyperparameter-tuning: 21.5


#Hyperparameter tuning
#Før endelig grid search prøver jeg å snevre inn en og en parameter med hensyn til tid- og maskinbegrensninger
#Øker parameterne fra default noe før dette. Trees til 300, min_n til 12, 

#Finner først en bra learning rate. Denne er typisk mellom 0,01 og 0,1.

hyper_grid_learnrate <- expand.grid(
  learning_rate = c(0.3, 0.1, 0.08, 0.06, 0.05, 0.01, 0.005),
  MAPE = NA,
  time = NA
)

for(i in seq_len(nrow(hyper_grid_learnrate))) {
  
  set.seed(42)
  
  train_time <- system.time({
    xg_mod <- boost_tree(mode = "regression",
                         trees = 300,
                         min_n = 12,
                         tree_depth = 6,
                         learn_rate = hyper_grid_learnrate$learning_rate[i],
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
  
  hyper_grid_learnrate$MAPE[i]  <- mean(abs((prediction$truth-prediction$estimate)/prediction$truth) * 100)
  hyper_grid_learnrate$time[i]  <- train_time[["elapsed"]]
  
}

arrange(hyper_grid_learnrate, MAPE) #Setter learn_rate i området rundt 0.08. learn_rate=0.3 vil gi stor fare for overfitting

#Finner en fin loss reduction (gamma). Denne er typisk mellom 0 og 5.

hyper_grid_loss_reduction <- expand.grid(
  loss_reduction = c(0,0.5,1,3,5),
  MAPE = NA,
  time = NA
)

for(i in seq_len(nrow(hyper_grid_loss_reduction))) {
  
  set.seed(42)
  
  train_time <- system.time({
    xg_mod <- boost_tree(mode = "regression",
                         trees = 300,
                         min_n = 12,
                         tree_depth = 6,
                         learn_rate = 0.08 ,
                         loss_reduction = hyper_grid_loss_reduction$loss_reduction[i]) %>% 
      set_engine("xgboost", tree_method = "exact") %>% 
      fit(ad_tot_price ~ ., data = finn_train)
    
    prediction <- predict(xg_mod, finn_test) %>% 
      bind_cols(finn_test_raw) %>% 
      dplyr::rename(estimate = .pred,
                    truth = ad_tot_price) %>% 
      mutate(abs_dev      = abs(truth - estimate),
             abs_dev_perc = abs_dev/truth)
  })
  
  hyper_grid_loss_reduction$MAPE[i]  <- mean(abs((prediction$truth-prediction$estimate)/prediction$truth) * 100)
  hyper_grid_loss_reduction$time[i]  <- train_time[["elapsed"]]
  
}

arrange(hyper_grid_loss_reduction, MAPE) #Ingen forskjell i MAPE og liten forskjell i treningstid. Velger 0,9 som var default i oppgaven, og gir litt sikkerhet for overtilpasning.

#Tester forskjellige alternativer for tree_depth. Dybden er som regel i området 3-12

hyper_grid_treedepth <- expand.grid(
  tree_depth = c(3,5,7,9,11,13),
  MAPE = NA,
  time = NA
)

for(i in seq_len(nrow(hyper_grid_treedepth))) {
  
  set.seed(42)
  
  train_time <- system.time({
    xg_mod <- boost_tree(mode = "regression",
                         trees = 300,
                         min_n = 12,
                         tree_depth = hyper_grid_treedepth$tree_depth[i],
                         learn_rate = 0.08,
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
  
  hyper_grid_treedepth$MAPE[i]  <- mean(abs((prediction$truth-prediction$estimate)/prediction$truth) * 100)
  hyper_grid_treedepth$time[i]  <- train_time[["elapsed"]]
  
}

arrange(hyper_grid_treedepth, MAPE) #Velger dybde lik 13, men det er liten gevinst. Så om grid searchen blir for krevende kan den settes ned til 8.


#Tester forskjellige alternativer for min_n. Denne er som regel i området 1-5.

hyper_grid_min_n <- expand.grid(
  min_n = c(1,2,3,4,5),
  MAPE = NA,
  time = NA
)

for(i in seq_len(nrow(hyper_grid_min_n))) {
  
  set.seed(42)
  
  train_time <- system.time({
    xg_mod <- boost_tree(mode = "regression",
                         trees = 300,
                         min_n = hyper_grid_min_n$min_n[i],
                         tree_depth = 13,
                         learn_rate = 0.08,
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
  
  hyper_grid_min_n$MAPE[i]  <- mean(abs((prediction$truth-prediction$estimate)/prediction$truth) * 100)
  hyper_grid_min_n$time[i]  <- train_time[["elapsed"]]
  
}

arrange(hyper_grid_min_n, MAPE) #Liten forskjell i MAPE. Velger å legge min_n i området rundt eller lik 3. Mindre kan øke sjansen for overfitting

#Tester forskjellige antall trær. Denne er som regel i området 100-100.

hyper_grid_trees <- expand.grid(
  trees = c(50,100,200,400,600,800,1000,1200),
  MAPE = NA,
  time = NA
)

for(i in seq_len(nrow(hyper_grid_trees))) {
  
  set.seed(42)
  
  train_time <- system.time({
    xg_mod <- boost_tree(mode = "regression",
                         trees = hyper_grid_trees$trees[i],
                         min_n = 3,
                         tree_depth = 13,
                         learn_rate = 0.08,
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
  
  hyper_grid_trees$MAPE[i]  <- mean(abs((prediction$truth-prediction$estimate)/prediction$truth) * 100)
  hyper_grid_trees$time[i]  <- train_time[["elapsed"]]
  
}

arrange(hyper_grid_trees, MAPE) #Tilsynelatende ingen ekstra gevinster ved å gå over 600 trær, letter i området opp til ca. 700 i grid search


#Endelig total grid search for alle variabler. Loss_reduction settes fast til 0.9 jf. tidligere test.
hyper_grid <- expand.grid(
  learning_rate = c(0.07,0.08,0.09),
  trees = c(seq(from = 500, to = 700, by = 50)),
  min_n = seq(from = 2, to = 5, by = 1),
  tree_depth = c(8,10),
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


