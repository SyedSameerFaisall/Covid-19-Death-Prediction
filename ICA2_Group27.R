#  STAT0023 ICA2 -- Analysis of COVID-19 Deaths by MSOA

library(MASS)
library(mgcv)
library(RColorBrewer)

# Plotting plot defaults 
plot_cex_main <- 1.0
plot_cex_lab  <- 0.85
plot_cex_axis <- 0.8

# Section 1. DATA LOADING AND FEATURE ENGINEERING
cat("=== 1. Loading data and engineering features ===\n")

covid <- read.csv("UKCovidWave1.csv")
covid$Deaths[covid$Deaths == -1] <- NA
covid$Region <- factor(covid$Region)
covid$RUCode <- factor(covid$RUCode, levels = c("A1","B1","C1","C2","D1","D2","E1","E2"))

# Proportion variables (normalising counts by MSOA population since proportions make areas comparable and reduce scale bias)
# Proportion aged 65+ (dominant risk factor, Williamson et al.)
covid$PropOld <- (covid$Age65.74 + covid$Age75.84 + covid$Age85.89 + covid$Age90.) / covid$PopTot

# Ethnicity: individual components (avoid sum to reduce collinearity)
covid$PropAsian    <- covid$EthAsian / covid$PopTot
covid$PropBlack    <- covid$EthBlack / covid$PopTot
covid$PropBornNonEU <- covid$BornNonEU / covid$PopTot

# Care homes and communal establishments (major COVID risk factor)
covid$PropCareHome <- (covid$LACare + covid$PrivCareNurs + covid$PrivCareNoNurs) / covid$PopTot
covid$PropComm     <- covid$PopComm / covid$PopTot

# Deprivation: households deprived in 3 or 4 dimensions
covid$PropDepriv34 <- (covid$HHDepriv3 + covid$HHDepriv4) / covid$HH

# Health problems (proxy for comorbidity burden per MSOA)
covid$PropHealthPrb <- covid$HH_HealthPrb / covid$HH

# Education: proportion with no qualifications (lower SES / occupational risk proxy)
covid$Pop16Plus  <- covid$Age16.17 + covid$Age18.19 + covid$Age20.24 + covid$Age25.29 + covid$Age30.44 + covid$Age45.59 +
  covid$Age60.64 + covid$Age65.74 + covid$Age75.84 + covid$Age85.89 + covid$Age90.
covid$PropNoQual <- covid$NoQual / covid$Pop16Plus

# Key workers (caring, machine, elementary -- couldn't work from home)
total_workers <- (covid$WrkMgr + covid$WrkProf + covid$WrkProfTech + covid$WrkAdmin + covid$WrkSkilled + covid$WrkCaring +
                    covid$WrkSales + covid$WrkMachine + covid$WrkElementary)
covid$PropKeyWorker <- (covid$WrkCaring + covid$WrkElementary + covid$WrkMachine) / total_workers

# Public transport use (virus transmission vector)
covid$PropPublicTransport <- (covid$MetroUsers + covid$TrainUsers + covid$BusUsers) / covid$PopTot

# Housing quality and overcrowding
covid$PropShared <- (covid$DwellShared2 + covid$DwellShared3.) / covid$Dwell
covid$PropNoCH   <- covid$HHNoCH / covid$HH

# Sex ratio (males at higher COVID risk per Williamson et al.)
covid$PropMale <- covid$PopM / covid$PopTot

# Offset for count models and death rate for EDA
covid$LogPopTot <- log(covid$PopTot)
covid$DeathRate <- covid$Deaths / covid$PopTot * 1e5

cat("Feature engineering complete:", ncol(covid), "columns\n")

# 2. TRAIN / PREDICTION SPLIT
cat("\n=== 2. Splitting data ===\n")

train <- covid[!is.na(covid$Deaths), ]
pred  <- covid[is.na(covid$Deaths), ]
cat("Training set:", nrow(train), "MSOAs | Prediction set:", nrow(pred), "MSOAs\n")

# 3. RUCode HIERARCHICAL CLUSTERING (on training set only)
#    Reduce 8 urban-rural levels to 4 data-driven groups to avoid sparse factor levels and improve model stability.
cat("\n=== 3. Clustering RUCode levels ===\n")

numeric_cols <- sapply(train, is.numeric)
RUMeans <- aggregate(train[, numeric_cols], by = list(train$RUCode), FUN = mean)
rownames(RUMeans) <- RUMeans[, 1]
RUMeans <- scale(RUMeans[, -1])

Distances <- dist(RUMeans)
ClusTree  <- hclust(Distances, method = "complete")

RUGroups <- cutree(ClusTree, k = 4)
cat("RU Code -> Group mapping:\n"); print(RUGroups)

group_map <- data.frame(RUCode = names(RUGroups), RUGroups = factor(RUGroups))
covid <- merge(covid, group_map, by = "RUCode", all.x = TRUE)
train <- covid[!is.na(covid$Deaths), ]
pred  <- covid[is.na(covid$Deaths), ]

# 4. COVARIATE SELECTION (literature-guided)
cat("\n=== 4. Covariate selection and housing PCA ===\n")
housing_vars <- c("PropShared", "PropNoCH", "HHRooms", "HHBedrooms")
pca_housing  <- prcomp(train[, housing_vars], scale. = TRUE)

cum_var_h <- cumsum(pca_housing$sdev^2) / sum(pca_housing$sdev^2)
n_keep_h  <- max(1, min(which(cum_var_h >= 0.80)))

cat(sprintf("Housing PCA: %d vars -> %d PC(s) retained (%.1f%% variance)\n", length(housing_vars), n_keep_h, cum_var_h[n_keep_h] * 100))
cat("Loadings:\n")
print(round(pca_housing$rotation[, 1:n_keep_h, drop = FALSE], 3))

# Project housing PCA scores directly onto train and pred
train_h_scores <- predict(pca_housing, newdata = train[, housing_vars])
pred_h_scores  <- predict(pca_housing, newdata = pred[,  housing_vars])

for (j in seq_len(n_keep_h)) {
  train[[paste0("Housing_PC", j)]] <- train_h_scores[, j]
  pred[[paste0("Housing_PC",  j)]] <- pred_h_scores[,  j]
}

# 5. EXPLORATORY DATA ANALYSIS
cat("\n=== 5. Exploratory Data Analysis ===\n")

# Full candidate set
candidate_set <- c("PropOld", "PropMale", "PropDepriv34", "PropCareHome", "PropComm", "PropHealthPrb", "PropAsian", "PropBlack", "PropBornNonEU", "PropNoQual",
                   "PropKeyWorker", "PropPublicTransport", paste0("Housing_PC", seq_len(n_keep_h)))
cat("\nCandidates (", length(candidate_set), "):\n  ", paste(candidate_set, collapse = ", "), "\n")

# Scatter plots: death rate vs each continuous predictor with GAM smooth
n_col <- 4
n_row <- ceiling(length(candidate_set) / n_col)

pdf("fig1_scatter_predictors.pdf", width = 12, height = 9)
par(mfrow  = c(n_row, n_col), mar = c(3.5, 3.5, 2, 0.5), mgp = c(2, 0.6, 0), oma = c(3, 0, 1.5, 0))

for (v in candidate_set) {
  x  <- train[[v]]
  y  <- train$DeathRate
  ok <- is.finite(x) & is.finite(y)
  
  plot(x[ok], y[ok], pch = 16, cex = 0.15, col = rgb(0, 0, 0.6, 0.12), xlab = v, ylab = "Deaths/100k", main = paste("Death Rate vs", v),
       cex.main = 0.8, cex.lab = 0.72, cex.axis = 0.65)
  
  tryCatch({
    g  <- gam(y ~ s(x), data = data.frame(x = x[ok], y = y[ok]))
    xs <- seq(min(x[ok]), max(x[ok]), length.out = 200)
    lines(xs, predict(g, data.frame(x = xs)), col = "red2", lwd = 1.5)
  }, error = function(e) NULL)
}
mtext("Death Rate (per 100k) vs Selected Continuous Predictors", outer = TRUE, cex = 0.9)
mtext("Figure 1: EDA scatter plots with GAM smooths. Each panel shows death rate per 100k against a candidate predictor.",
      outer = TRUE, side = 1, line = 1.5, cex = 0.72, adj = 0.5)
dev.off()

# Boxplot by RUGroups and Region (side by side)
pdf("fig2_boxplots.pdf", width = 10.5, height = 5.25)
par(mfrow = c(1, 2), mar = c(5, 4, 3, 1), oma = c(3, 0, 0, 0))
boxplot(DeathRate ~ RUGroups, data = train, main = "Death Rate by Urban-Rural Group", xlab = "Group (1=most urban)", 
        ylab = "Deaths per 100k", col = brewer.pal(4, "Set2"), cex.main = plot_cex_main, cex.lab = plot_cex_lab, cex.axis = plot_cex_axis)

# Boxplot by Region
boxplot(DeathRate ~ Region, data = train, main = "Death Rate by Region", xlab = "Region", ylab = "Deaths per 100k",
        col = brewer.pal(max(3, nlevels(factor(train$Region))), "Set3"), cex.main = plot_cex_main, cex.lab = plot_cex_lab, cex.axis = 0.55, las = 1)
mtext("Figure 2: Regional and geographic variation. Left: Death rate by urban-rural group (1=most urban). Right: Death rate by administrative region.",
      outer = TRUE, side = 1, cex = 0.75, adj = 0.5)
dev.off()

# 5a. REGIONAL VARIATION 
cat("\n--- 5a. Death rate (per 100k) by Region ---\n")
reg_stats <- do.call(rbind, tapply(train$DeathRate, train$Region, function(x) {
  x <- x[is.finite(x)]
  c(n = length(x), mean = round(mean(x), 1), median = round(median(x), 1),
    sd = round(sd(x), 1), min = round(min(x), 1), max = round(max(x), 1))
}))
print(reg_stats)
# Flag highest and lowest median regions
reg_medians <- sort(tapply(train$DeathRate[is.finite(train$DeathRate)], train$Region[is.finite(train$DeathRate)], median))
cat(sprintf("  Highest median death rate: %s (%.1f per 100k)\n", names(reg_medians)[length(reg_medians)], reg_medians[length(reg_medians)]))
cat(sprintf("  Lowest  median death rate: %s (%.1f per 100k)\n", names(reg_medians)[1], reg_medians[1]))
cat(sprintf("  Ratio highest/lowest median: %.2fx\n", reg_medians[length(reg_medians)] / reg_medians[1]))

# 5b. URBAN-RURAL VARIATION 
cat("\n--- 5b. Death rate (per 100k) by Urban-Rural Group (1=most urban) ---\n")
ru_stats <- do.call(rbind, tapply(train$DeathRate, train$RUGroups, function(x) {
  x <- x[is.finite(x)]
  c(n = length(x), mean = round(mean(x), 1), median = round(median(x), 1), sd = round(sd(x), 1))}))
print(ru_stats)
ru_medians <- tapply(train$DeathRate[is.finite(train$DeathRate)], train$RUGroups[is.finite(train$DeathRate)], median)
cat(sprintf("  Urban (Grp 1) median: %.1f | Rural (Grp 4) median: %.1f\n", ru_medians[1], ru_medians[4]))

# 5c. CARE HOME & COMMUNAL LIVING 
cat("\n--- 5c. Care home and communal living summary ---\n")
cat(sprintf("  PropCareHome range: [%.4f, %.4f] | mean: %.4f\n",
            min(train$PropCareHome, na.rm=TRUE), max(train$PropCareHome, na.rm=TRUE), mean(train$PropCareHome, na.rm=TRUE)))
cat(sprintf("  PropComm     range: [%.4f, %.4f] | mean: %.4f\n",
            min(train$PropComm, na.rm=TRUE), max(train$PropComm, na.rm=TRUE), mean(train$PropComm, na.rm=TRUE)))
cat(sprintf("  Proportion of MSOAs with PropCareHome > 0: %.1f%%\n", 100*mean(train$PropCareHome > 0, na.rm=TRUE)))

# 5d. DEPRIVATION AND HEALTH 
cat("\n--- 5d. Deprivation and health ---\n")
q_depriv <- quantile(train$PropDepriv34, probs=c(0.25,0.75), na.rm=TRUE)
ok_d <- is.finite(train$DeathRate)
dr_lo <- mean(train$DeathRate[ok_d & train$PropDepriv34 < q_depriv[1]], na.rm=TRUE)
dr_hi <- mean(train$DeathRate[ok_d & train$PropDepriv34 > q_depriv[2]], na.rm=TRUE)
cat(sprintf("  Mean death rate in bottom quartile of deprivation: %.1f per 100k\n", dr_lo))
cat(sprintf("  Mean death rate in top    quartile of deprivation: %.1f per 100k\n", dr_hi))
cat(sprintf("  Ratio (most/least deprived): %.2fx\n", dr_hi / dr_lo))

# 5e. ETHNICITY
cat("\n--- 5e. Ethnicity correlations with death rate ---\n")
for (v in c("PropAsian", "PropBlack")) {
  x <- train[[v]]; y <- train$DeathRate; ok <- is.finite(x) & is.finite(y)
  cat(sprintf("  %-15s | Pearson r = %+.3f | Spearman rho = %+.3f\n",
              v, cor(x[ok], y[ok], method = "pearson"), cor(x[ok], y[ok], method = "spearman")))}
cat(sprintf("  PropAsian vs PropBlack inter-correlation: r = %.3f\n", cor(train$PropAsian, train$PropBlack, use="complete.obs")))

# 5f. COLLINEARITY CHECK
# Correlation matrix of selected continuous predictors
cor_mat <- cor(train[, candidate_set], use = "complete.obs")
n_v     <- length(candidate_set)
# Report highly correlated pairs (|r| > 0.6):
cat("\n--- 5f. Highly correlated predictor pairs (|r| > 0.6) ---\n")
for (i in 1:(n_v - 1)) {
  for (j in (i + 1):n_v) {
    r <- cor_mat[i, j]
    if (abs(r) > 0.6)
      cat(sprintf("  %s vs %s: r = %.3f\n", candidate_set[i], candidate_set[j], r))
  }
}

# Drop collinear variables directly from candidate_set
candidate_set <- setdiff(candidate_set, c("PropBornNonEU", "PropNoQual"))

# 6. MODEL TYPE COMPARISON: LINEAR → POISSON GLM → NEGATIVE BINOMIAL GLM
# RUGroups = urban-rural cluster (4 data-driven groups from RUCode hierarchical clustering — replaces 8 sparse RU categories)
# Region = administrative geographic region (e.g. London, South East); both are included throughout.
cat("\n=== 6. Model type comparison: Linear LM vs Poisson GLM vs NB GLM ===\n")

# Housing PC names — at least Housing_PC1 is always retained (n_keep_h >= 1)
hp_names <- paste0("Housing_PC", seq_len(n_keep_h))

# m0: Linear model baseline — Normal distribution, constant variance assumed
# Residual checks determine whether the normal / constant-variance assumption is defensible
m0 <- lm(as.formula(paste("Deaths ~ as.factor(RUGroups) + as.factor(Region) +",
                          "PropOld + PropMale + PropDepriv34 + PropCareHome + PropComm + PropHealthPrb +",
                          "PropAsian + PropBlack + PropKeyWorker + PropPublicTransport +",
                          paste(hp_names, collapse = " + "))), data = train)
sw  <- shapiro.test(sample(residuals(m0), min(5000, length(residuals(m0)))))  # Shapiro-Wilk (subsample for speed)
bp_var <- var(residuals(m0)[train$Deaths <= median(train$Deaths)]) /
  var(residuals(m0)[train$Deaths > median(train$Deaths)])   # informal variance ratio: low vs high fitted
cat(sprintf("  m0 Linear LM   | R²: %.3f | Shapiro-Wilk p: %.2e | Var ratio (low/high): %.2f\n", summary(m0)$r.squared, sw$p.value, bp_var))
cat("  -> Non-normality (p << 0.05) and heteroscedasticity (ratio ≠ 1) motivate moving to a GLM.\n")

# m1: Poisson GLM — log link + population offset, all covariates linear
# Equidispersion assumed: Var(Y) = mu
m1 <- glm(as.formula(paste("Deaths ~ as.factor(RUGroups) + as.factor(Region) +",
                           "PropOld + PropMale + PropDepriv34 + PropCareHome + PropComm + PropHealthPrb +",
                           "PropAsian + PropBlack + PropKeyWorker + PropPublicTransport +", paste(hp_names, collapse = " + "), "+ offset(LogPopTot)")),
          family = poisson(), data = train)

# m2: Negative Binomial GLM — adds free overdispersion parameter theta
# Var(Y) = mu + mu^2/theta; theta estimated by MLE 
m2 <- glm.nb(as.formula(paste("Deaths ~ as.factor(RUGroups) + as.factor(Region) +",
                              "PropOld + PropMale + PropDepriv34 + PropCareHome + PropComm + PropHealthPrb +",
                              "PropAsian + PropBlack + PropKeyWorker + PropPublicTransport +", paste(hp_names, collapse = " + "), "+ offset(LogPopTot)")), data = train)

pois_disp <- sum(residuals(m1, type = "pearson")^2) / m1$df.residual
theta_est <- m2$theta

cat(sprintf("  m1 Poisson GLM | AIC: %10.1f | Pearson dispersion: %.2f\n", AIC(m1), pois_disp))
cat(sprintf("  m2 NB GLM      | AIC: %10.1f | theta: %.4f\n", AIC(m2), theta_est))
cat(sprintf("  Overdispersion (%.2f >> 1) and AIC gain of %.1f confirm NB family.\n", pois_disp, AIC(m1) - AIC(m2)))

# 7. INITIAL FULL NB GAM — all continuous covariates as smooth terms
#    m3: thin-plate regression splines (bs="ts") + select=TRUE + fixed theta.
#    Key output: EDF per smooth term
#      EDF ≈ 1  → essentially linear  → switch to parametric in m4
#      EDF >> 1 → genuine nonlinearity → retain as smooth in m4
cat("\n=== 7. Initial full NB GAM (m3) ===\n")

m3 <- gam(as.formula(paste( "Deaths ~ as.factor(RUGroups) + as.factor(Region) +",
                            "s(PropOld, bs='ts', k=10) + s(PropMale, bs='ts', k=10) +", "s(PropDepriv34, bs='ts', k=10) + s(PropCareHome, bs='ts', k=10) +",
                            "s(PropComm, bs='ts', k=10) + s(PropHealthPrb, bs='ts', k=10) +", "s(PropAsian, bs='ts', k=10) +",
                            "s(PropBlack, bs='ts', k=10) + s(PropKeyWorker, bs='ts', k=10) +", "s(PropPublicTransport, bs='ts', k=10) +", 
                            paste(paste0("s(", hp_names, ", bs='ts', k=10)"), collapse = " + "), "+ offset(LogPopTot)")),
          family = nb(theta = theta_est), data = train, method = "REML", select = TRUE)

cat(sprintf("m3 AIC: %.2f | Dev.expl: %.1f%%\n", AIC(m3), summary(m3)$dev.expl * 100))
cat("\nSmooth term EDFs (informs smooth vs linear decision in m4):\n")
print(round(summary(m3)$s.table[, c("edf", "Ref.df", "p-value")], 4))

# 8. MIXED SMOOTH / LINEAR NB GAM (m4)
# Guided by EDA scatter plots and EDF from m3:
cat("\n=== 8. Mixed smooth/linear NB GAM (m4) ===\n")

# Classify terms by EDF from m3: EDF > 1.5 -> nonlinear smooth; EDF <= 1.5 -> treat as linear
# select=TRUE in GAM automatically penalises weak smooths toward zero contribution, so weak terms have minimal cost and rarely benefit from explicit removal.
m3_edf   <- summary(m3)$s.table[, "edf"]
m3_names <- gsub("s\\(|\\).*", "", rownames(summary(m3)$s.table))  # strip s(...) wrapper
m4_smooth_vars  <- m3_names[m3_edf > 1.5]
m4_linear_vars  <- setdiff(m3_names[m3_edf <= 1.5], hp_names)  # Housing PCs always smooth
m4_smooth_vars  <- union(m4_smooth_vars, hp_names)              # Housing PCs always smooth
cat(sprintf("  Nonlinear (keep as smooth): %s\n", paste(m4_smooth_vars, collapse = ", ")))
cat(sprintf("  Roughly linear (parametric): %s\n",  paste(m4_linear_vars,  collapse = ", ")))

m4_smooth_terms <- paste(paste0("s(", m4_smooth_vars, ", bs='ts', k=10)"), collapse = " + ")
m4_linear_terms <- if (length(m4_linear_vars) > 0) paste(m4_linear_vars, collapse = " + ") else NULL
m4_formula <- as.formula(paste("Deaths ~ as.factor(RUGroups) + as.factor(Region) +",
                               m4_smooth_terms,
                               if (!is.null(m4_linear_terms)) paste("+", m4_linear_terms), "+ offset(LogPopTot)"))

m4 <- gam(m4_formula, family = nb(theta = theta_est), data = train, method = "REML", select = TRUE)

cat(sprintf("m3 All-smooth | AIC: %.2f\n", AIC(m3)))
cat(sprintf("m4 Mixed      | AIC: %.2f\n", AIC(m4)))

# Select m4 if it has better AIC than m3
if (AIC(m4) < AIC(m3)) {
  current_model <- m4
  cur_smooth    <- m4_smooth_vars
  cur_linear    <- m4_linear_vars
  cat("-> m4 selected (better AIC)\n")
} else {
  current_model <- m3
  cur_smooth    <- m3_names   # all terms were smooth in m3
  cur_linear    <- character(0)
  cat("-> m3 retained (better or equal AIC)\n")
}

# 9. BACKWARD SELECTION — drop linear terms if AIC improves; smooths already penalised by select=TRUE
cat("\n=== 9. Backward selection (linear terms only) ===\n")
cat("Start AIC:", round(AIC(current_model), 2), "\n")

build_formula <- function(smooth_v, linear_v, use_region = TRUE) {
  smooth_t <- if (length(smooth_v) > 0) paste(paste0("s(", smooth_v, ", bs='ts', k=10)"), collapse = " + ") else NULL
  linear_t <- if (length(linear_v) > 0) paste(linear_v, collapse = " + ") else NULL
  paste("Deaths ~", paste(c("as.factor(RUGroups)", if (use_region) "as.factor(Region)", smooth_t, linear_t, "offset(LogPopTot)"), 
                          collapse = " + ")) |> as.formula()}

repeat {
  best_aic <- AIC(current_model); best_var <- NULL
  for (v in cur_linear) {
    m <- tryCatch(gam(build_formula(cur_smooth, setdiff(cur_linear, v)), 
                      family = nb(theta = theta_est), data = train, method = "REML", select = TRUE), error = function(e) NULL)
    if (!is.null(m) && AIC(m) < best_aic) { best_aic <- AIC(m); best_var <- v }
  }
  if (is.null(best_var)) break
  cat(sprintf("Drop %-30s | AIC: %.1f -> %.1f\n", best_var, AIC(current_model), best_aic))
  cur_linear <- setdiff(cur_linear, best_var)
  current_model <- gam(build_formula(cur_smooth, cur_linear), family = nb(theta = theta_est), data = train, method = "REML", select = TRUE)
}
cat("Final AIC after backward selection:", round(AIC(current_model), 2), "\n")
cat(sprintf("  Retained smooth: %s\n  Retained linear: %s\n", paste(cur_smooth, collapse=", "), if(length(cur_linear)>0) paste(cur_linear, collapse=", ") else "none"))

# Region AIC test: keep Region only if it improves AIC
f_noreg <- build_formula(cur_smooth, cur_linear, use_region = FALSE)
m_noreg <- gam(f_noreg, family = nb(theta = theta_est), data = train, method = "REML", select = TRUE)
cat(sprintf("Region AIC test | with = %.1f | without = %.1f\n", AIC(current_model), AIC(m_noreg)))
use_region <- AIC(current_model) < AIC(m_noreg)
if (!use_region) { current_model <- m_noreg; cat("Region dropped\n") } else cat("Region kept\n")

# 10. INTERACTION TESTING
cat("\n=== 10. Interaction testing ===\n")

test_interaction <- function(model, v1, v2, cur_smooth, cur_linear) {
  if (!(v1 %in% c(cur_smooth, cur_linear)) || !(v2 %in% c(cur_smooth, cur_linear))) return(model)
  m_int <- tryCatch(update(model, as.formula(paste0(".~. + ", v1, ":", v2))), error = function(e) NULL)
  if (is.null(m_int)) return(model)
  lrt_p <- tryCatch(anova(model, m_int, test = "Chisq")[2, "Pr(>Chi)"], error = function(e) NA)
  kept  <- AIC(m_int) < AIC(model) - 2 & !is.na(lrt_p) & lrt_p < 0.05
  cat(sprintf("  %-35s | AIC: %.1f vs %.1f (delta: %+.1f) | p = %.4f%s\n",
              paste0(v1, ":", v2), AIC(m_int), AIC(model), AIC(m_int) - AIC(model), lrt_p,
              if (kept) " -> RETAINED" else ""))
  if (kept) m_int else model
}

current_model <- test_interaction(current_model, "PropOld",     "PropCareHome", cur_smooth, cur_linear)
current_model <- test_interaction(current_model, "PropOld",     "PropComm",     cur_smooth, cur_linear)
current_model <- test_interaction(current_model, "PropCareHome","PropDepriv34", cur_smooth, cur_linear)

# 11. MODEL VALIDATION
#   k.check: basis dimension adequacy, k-index close to 1 is fine; << 1 means k is too small
#   Concurvity: Worst-case > 0.8 signals potential instability
cat("\n=== 11. Model validation ===\n")

kc <- k.check(current_model)
print(kc)
low_k <- rownames(kc)[kc[, "k-index"] < 0.9 & !is.na(kc[, "k-index"])]
if (length(low_k) > 0)
  cat(sprintf("WARNING: k may be too small for: %s -- increase k.\n", paste(low_k, collapse = ", ")))

conc      <- concurvity(current_model, full = TRUE)
high_conc <- names(which(conc["worst", ] > 0.8))
cat(sprintf("Worst-case concurvity: max = %.3f%s\n", max(conc["worst", ]),
            if (length(high_conc) > 0) paste0(" | High: ", paste(high_conc, collapse = ", ")) else ""))

# 12. FINAL MODEL — free-theta refit + full diagnostics
cat("\n=== 12. Final model ===\n")

final_model <- gam(formula(current_model), family = nb(), data = train, method = "REML", select = TRUE)
theta    <- final_model$family$getTheta(TRUE)
dev_expl <- summary(final_model)$dev.expl * 100

cat("\n--- FINAL MODEL SUMMARY ---\n")
print(summary(final_model))
cat(sprintf("\nTheta: %.4f | Dev.expl: %.1f%% | AIC: %.2f\n", theta, dev_expl, AIC(final_model)))

pdf("fig3_gam_check.pdf", width = 7.5, height = 7.5)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1), oma = c(2, 0, 0, 0))
gam.check(final_model)
par(mfg = c(1, 1))
title(main = "QQ Plot of Deviance Residuals", cex.main = 0.9)
mtext("Figure 3: GAM basis dimension adequacy checks.",
      outer = TRUE, side = 1, line = 0.8, cex = 0.72, adj = 0.5)
dev.off()

pdf("fig4_smooth_effects.pdf", width = 10.5, height = 7.5)
par(mfrow = c(1, 1), mar = c(5, 4, 2, 1), oma = c(1.5, 0, 0, 0))
plot(final_model, pages = 1, shade = TRUE,
     shade.col = rgb(0.1, 0.1, 0.8, 0.2), all.terms = TRUE, cex.main = plot_cex_main, cex.lab = plot_cex_lab)
mtext("Figure 4: Estimated smooth effects and parametric terms from final NB GAM. Shaded regions show 95% confidence bands.",
      outer = TRUE, side = 1, line = 0.2, cex = 0.75, adj = 0.5)
dev.off()

mu_fit    <- fitted(final_model)
dev_resid <- residuals(final_model, type = "deviance")
lev       <- final_model$hat
thresh    <- 2 * sum(final_model$edf) / nrow(final_model$model)

hi_lev <- which(lev > thresh)
cat(sprintf("High-leverage MSOAs (>2p/n=%.4f): %d of %d\n", thresh, length(hi_lev), nrow(final_model$model)))

# 13. PREDICTIONS AND PREDICTION ERROR STANDARD DEVIATIONS
cat("\n=== 13. Predictions for held-out MSOAs ===\n")

preds  <- predict(final_model, newdata = pred, type = "response", se.fit = TRUE)
mu_hat <- preds$fit
se_mu  <- preds$se.fit

var_y   <- mu_hat + mu_hat^2 / theta
sd_pred <- sqrt(var_y + se_mu^2)

pred_table <- data.frame(ID = pred$ID, Pred = mu_hat, SD = sd_pred)
pred_table <- pred_table[order(pred_table$ID), ]

write.table(pred_table, file = "ICA2_Group27_pred.dat", row.names = FALSE, col.names = FALSE, sep = " ")

cat("Predictions written to ICA2_Group27_pred.dat\n")
cat(sprintf("  Mean predicted deaths:   %.2f\n", mean(mu_hat)))
cat(sprintf("  Median predicted deaths: %.2f\n", median(mu_hat)))
cat(sprintf("  Prediction SD range:     [%.2f, %.2f]\n", min(sd_pred), max(sd_pred)))
cat("\nFirst 6 predictions (ID | Pred | SD):\n"); print(head(pred_table, 6))

# 14. SCORE CALCULATION
cat("\n=== 14. Score calculation ===\n")

score_S <- function(Y, mu_hat, sigma_i) {
  sum(log(sigma_i) + (Y - mu_hat)^2 / (2 * sigma_i^2))
}

# Get predictions on training set for scoring
train_preds  <- predict(final_model, newdata = train, type = "response", se.fit = TRUE)
train_mu_hat <- train_preds$fit
train_se_mu  <- train_preds$se.fit
train_sd_pred <- sqrt(train_mu_hat + train_mu_hat^2 / theta + train_se_mu^2)

# Compute score on training set
my_score <- score_S(train$Deaths, train_mu_hat, train_sd_pred)

cat(sprintf("Model score: %.4f\n", my_score))

cat("\n=== Analysis complete. Figures: fig1-fig4 PDFs | Results: ICA2_Group27_pred.dat ===\n")