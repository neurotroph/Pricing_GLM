# Loading libraries --------------------------------------------------------------------------------
library(tidyverse)
library(brms)
library(here)

# Loading data -------------------------------------------------------------------------------------
df <- read_csv(here("Data/PricingGLM_SynthData.csv")) %>%
  mutate(Combination = case_when(Feature_1 == 0 & Feature_2 == 0 & Feature_3 == 0 ~ "None",
                                 Feature_1 == 1 & Feature_2 == 0 & Feature_3 == 0 ~ "1",
                                 Feature_1 == 0 & Feature_2 == 1 & Feature_3 == 0 ~ "2",
                                 Feature_1 == 0 & Feature_2 == 0 & Feature_3 == 1 ~ "3",
                                 Feature_1 == 1 & Feature_2 == 1 & Feature_3 == 0 ~ "1 + 2",
                                 Feature_1 == 1 & Feature_2 == 0 & Feature_3 == 1 ~ "1 + 3",
                                 Feature_1 == 0 & Feature_2 == 1 & Feature_3 == 1 ~ "2 + 3",
                                 Feature_1 == 1 & Feature_2 == 1 & Feature_3 == 1 ~ "1 + 2 + 3",
                                 TRUE ~ "Default"),
         # Building a factor with the selected combination
         Combination = fct_infreq(as_factor(Combination)),
         # Scaling & centering alternative prices as additional predictor in model
         PPrice_Alternative1_std = scale(PPrice_Alternative1),
         PPrice_Alternative2_std = scale(PPrice_Alternative2))

# Training model contains only cases without missing response
df_train <- df %>%
  filter(!is.na(PPrice_Combination))

# Building models ----------------------------------------------------------------------------------
# Model 1: Standard Gaussian model
bm_gauss <- brm(PPrice_Combination ~ Feature_1 + Feature_2 + Feature_3 + PPrice_Alternative1_std +
                  PPrice_Alternative2_std,
                prior = c(prior(normal(0, 100), "b"),
                          prior(normal(0, 100), "Intercept")),
                family = gaussian(),
                sample_prior = "only",
                data = df_train,
                chains = 2, cores = 4)

# Model 2: Truncated Gaussian model (response >= 0)
bm_trunc <- brm(PPrice_Combination | trunc(lb = 0) ~ Feature_1 + Feature_2 + Feature_3 +
                  PPrice_Alternative1_std + PPrice_Alternative2_std,
                prior = c(prior(normal(0, 100), "b"),
                          prior(normal(0, 100), "Intercept")),
                family = gaussian(),
                sample_prior = "only",
                data = df_train,
                chains = 2, cores = 4)

# Model 3: Log-transformed Gaussian model
bm_logtr <- brm(log(PPrice_Combination) ~ Feature_1 + Feature_2 + Feature_3 +
                  PPrice_Alternative1_std + PPrice_Alternative2_std,
                prior = c(prior(normal(0, 100), "b"),
                          prior(normal(0, 100), "Intercept")),
                family = gaussian(),
                sample_prior = "only",
                data = df_train,
                chains = 2, cores = 4)

# Model 4: Log-normal model
bm_lognorm <- brm(PPrice_Combination ~ Feature_1 + Feature_2 + Feature_3 + PPrice_Alternative1_std +
                    PPrice_Alternative2_std,
                  prior = c(prior(normal(0, 100), "b"),
                            prior(normal(0, 100), "Intercept")),
                  family = lognormal(),
                  sample_prior = "only",
                  data = df_train,
                  chains = 4, cores = 4)

# Model comparison ---------------------------------------------------------------------------------
# Add LOO and WAIC to all models
bm_gauss <- add_criterion(bm_gauss, c("waic", "loo"), reloo = T)
bm_trunc <- add_criterion(bm_trunc, c("waic", "loo"), reloo = T)
bm_logtr <- add_criterion(bm_logtr, c("waic", "loo"), reloo = T)
bm_lognorm <- add_criterion(bm_lognorm, c("waic", "loo"), reloo = T)

# Print model comparison
print(loo_compare(bm_gauss, bm_trunc, bm_logtr, bm_lognorm, criterion = "loo"),
      digits = 2, simplify = F)
print(loo_compare(bm_gauss, bm_trunc, bm_lognorm, criterion = "loo"),
      digits = 2, simplify = F)

# Posterior predictive checks ----------------------------------------------------------------------
bayesplot::bayesplot_grid(plots = list(pp_check(bm_gauss, type = "intervals"),
                                       pp_check(bm_trunc, type = "intervals"),
                                       pp_check(bm_logtr, type = "intervals"),
                                       pp_check(bm_lognorm, type = "intervals")),
                          titles = c("Gaussian", "Truncated", "Log-Transformed", "Log-Normal"))

# Posterior predictions and comparison to actual results -------------------------------------------
df_pred <- df

# List of models for looping through
model_lst <- list(
  Gauss = bm_gauss,
  Truncated = bm_trunc,
  LogTransformed = bm_logtr,
  LogNormal = bm_lognorm
)

for (m in names(model_lst)) {
  pred <- predict(model_lst[[m]], newdata = df)
  df_pred[,paste0("PPrice_Pred_", m)] <- pred[,1]
  df_pred[,paste0("PPrice_Pred_", m, "_lb")] <- pred[,3]
  df_pred[,paste0("PPrice_Pred_", m, "_ub")] <- pred[,4]
}

str(df_pred)

df_pred %>%
  group_by(Combination) %>%
  summarise(
    N = n(),
    `Mean (actual)` = mean(PPrice_Combination, na.rm = T),
    `Mean (Gaussian)` = mean(PPrice_Pred_Gauss, na.rm = T),
    `Mean (Truncated)` = mean(PPrice_Pred_Truncated, na.rm = T),
    `Mean (Log-Transf.)` = mean(exp(PPrice_Pred_LogTransformed), na.rm = T),
    `Mean (Log-Norm.)` = mean(PPrice_Pred_LogNormal, na.rm = T)
  )
