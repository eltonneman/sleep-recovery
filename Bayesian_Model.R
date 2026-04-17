# DS4420 Final Project: Athlete Sleep Quality Prediction


library(tidyverse)
library(brms)
library(cmdstanr)
library(bayesplot)  
library(loo)
library(caret)

# Load Data

data_dir <- "/Users/Rithik/Documents/University/Year3/Spring/DS4420 | ML 2/Final Project/sleep-recovery-1/data"

wellness_raw <- map_dfr(1:16, function(i) {
  filepath <- file.path(data_dir, paste0("wellness_", i, ".csv"))
  read_csv(filepath, show_col_types = FALSE) %>%
    mutate(athlete_id = as.character(i))
})

# Preprocessing
wellness <- wellness_raw %>%
  mutate(
    date = as.Date(effective_time_frame),
    
    sleep_quality = pmin(pmax(round(sleep_quality), 1), 4)
  ) %>%
  arrange(athlete_id, date) %>%
  group_by(athlete_id) %>%
  # Lag-1 features
  mutate(
    lag_fatigue        = lag(fatigue),
    lag_stress         = lag(stress),
    lag_soreness       = lag(soreness),
    lag_sleep_duration = lag(sleep_duration_h),
    lag_readiness      = lag(readiness)
  ) %>%
  ungroup() %>%
  # Drop first observation per athlete
  drop_na(lag_fatigue, lag_stress, lag_soreness,
          lag_sleep_duration, lag_readiness, mood, sleep_quality)

# Predictors

predictors <- c("lag_fatigue", "lag_stress", "lag_soreness",
                "lag_sleep_duration", "lag_readiness", "mood")

# Z-score all predictors
wellness_scaled <- wellness %>%
  mutate(
    across(all_of(predictors), scale),
    sleep_quality = factor(sleep_quality, ordered = TRUE)
  )

# Priors

# Hierarchical model priors
priors <- c(
  prior(normal(0, 1),   class = b),
  prior(exponential(1), class = sd),
  prior(lkj(2),         class = cor)
)

# Population model priors
priors_population <- c(
  prior(normal(0, 1), class = b)
)

# Fit Model - Personalized

fit <- brm(
  formula = sleep_quality ~ lag_fatigue + lag_stress + lag_soreness +
    lag_sleep_duration + lag_readiness + mood +
    (1 + lag_fatigue + lag_stress + lag_soreness +
       lag_sleep_duration + lag_readiness + mood | athlete_id),
  data    = wellness_scaled,
  family  = cumulative("logit"),
  prior   = priors,
  backend = "cmdstanr",
  chains  = 4,
  iter    = 2000,
  warmup  = 1000,
  cores   = 4,
  seed    = 42,
  file    = "models/sleep_quality_ordinal_v4",
  control = list(adapt_delta = 0.95)
)

# Fit Model - Population

fit_population <- brm(
  formula = sleep_quality ~ lag_fatigue + lag_stress + lag_soreness +
    lag_sleep_duration + lag_readiness + mood,
  data    = wellness_scaled,
  family  = cumulative("logit"),
  prior   = priors_population,
  backend = "cmdstanr",
  chains  = 4,
  iter    = 2000,
  warmup  = 1000,
  cores   = 4,
  seed    = 42,
  file    = "models/sleep_quality_ordinal_population_v4"
)

# Convergence

print(summary(fit))

fit_summary <- summary(fit)
print(fit_summary$fixed)
print(fit_summary$random)

# Classification

set.seed(42)
pred_draws <- posterior_predict(fit, ndraws = 1000)

pred_class <- apply(pred_draws, 2, function(x) {
  as.integer(names(which.max(table(x))))
})

true_class <- as.integer(wellness_scaled$sleep_quality)

accuracy <- mean(pred_class == true_class)
cat("Hierarchical Bayesian Accuracy:", round(accuracy, 3), "\n")

cm <- confusionMatrix(
  factor(pred_class, levels = 1:4),
  factor(true_class,  levels = 1:4)
)
print(cm)

macro_f1 <- mean(cm$byClass[, "F1"], na.rm = TRUE)
cat("Macro F1 (Hierarchical Bayesian):", round(macro_f1, 3), "\n")

# Population model metrics
set.seed(42)
pred_draws_pop <- posterior_predict(fit_population, ndraws = 1000)

pred_class_pop <- apply(pred_draws_pop, 2, function(x) {
  as.integer(names(which.max(table(x))))
})

accuracy_pop <- mean(pred_class_pop == true_class)
cat("Population Bayesian Accuracy:", round(accuracy_pop, 3), "\n")

cm_pop     <- confusionMatrix(
  factor(pred_class_pop, levels = 1:4),
  factor(true_class,      levels = 1:4)
)
macro_f1_pop <- mean(cm_pop$byClass[, "F1"], na.rm = TRUE)
cat("Macro F1 (Population Bayesian):", round(macro_f1_pop, 3), "\n")

# LOO-CV

loo_fit        <- loo(fit)
loo_population <- loo(fit_population)

print(loo_compare(loo_fit, loo_population))

cat("\nHierarchical model LOO:\n"); print(loo_fit)
cat("\nPopulation model LOO:\n");   print(loo_population)

# Population Coefficient Plot

mcmc_intervals(
  as_draws_array(fit),
  pars = c("b_lag_fatigue", "b_lag_stress", "b_lag_soreness",
           "b_lag_sleep_duration", "b_lag_readiness", "b_mood"),
  prob = 0.80,
  prob_outer = 0.95
) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "Population-Level Posterior Coefficients",
       x = "Estimate", y = "Predictor")

# Athlete Mood Slopes
athlete_effects <- coef(fit)$athlete_id

# Per-athlete mood slopes
mood_slope_df <- athlete_effects[, , "mood"] %>%
  as.data.frame() %>%
  rownames_to_column("athlete_id") %>%
  arrange(Estimate)

ggplot(mood_slope_df, aes(x = reorder(athlete_id, Estimate),
                          y = Estimate,
                          ymin = Q2.5, ymax = Q97.5)) +
  geom_pointrange() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray") +
  coord_flip() +
  labs(title = "Per-Athlete Posterior Slope: Mood -> Sleep Quality",
       x = "Athlete ID", y = "Slope Estimate")

# LOO Comparison

loo_comparison_df <- data.frame(
  model = c("Hierarchical\n(Personalized)", "Population\n(Generalized)"),
  elpd  = c(loo_fit$estimates["elpd_loo", "Estimate"],
            loo_population$estimates["elpd_loo", "Estimate"]),
  se    = c(loo_fit$estimates["elpd_loo", "SE"],
            loo_population$estimates["elpd_loo", "SE"])
)

elpd_diff <- round(loo_comparison_df$elpd[1] - loo_comparison_df$elpd[2], 1)
y_mid     <- mean(loo_comparison_df$elpd)

ggplot(loo_comparison_df, aes(x = model, y = elpd, color = model)) +
  geom_point(size = 5) +
  geom_errorbar(aes(ymin = elpd - se, ymax = elpd + se),
                width = 0.1, linewidth = 0.7) +
  annotate("text", x = 1.5, y = y_mid,
           label = paste0("DELTA ELPD = ", elpd_diff), size = 4) +
  annotate("segment", x = 1.05, xend = 1.95,
           y = loo_comparison_df$elpd[1], yend = loo_comparison_df$elpd[2],
           linetype = "dashed", color = "gray") +
  scale_color_manual(values = c("Hierarchical\n(Personalized)" = "green",
                                "Population\n(Generalized)"    = "gray")) +
  labs(
    title    = "Model Comparison: LOO-CV Predictive Accuracy",
    subtitle = "Higher ELPD = better out-of-sample prediction",
    x        = "Model",
    y        = "ELPD (Expected Log Predictive Density)",
    caption  = "Error bars = +/- 1 SE"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none",
        plot.title    = element_text(face = "bold"),
        plot.subtitle = element_text(color = "gray"))

# Model Comparison
summary_table <- data.frame(
  Model    = c("Hierarchical Bayesian", "Population Bayesian",
               "MLP Personalized",     "MLP Generalized"),
  Accuracy = c(round(accuracy, 3),     round(accuracy_pop, 3),
               0.478,                  0.459),
  Macro_F1 = c(round(macro_f1, 3),     round(macro_f1_pop, 3),
               0.321,                  0.237)
)

print(summary_table)