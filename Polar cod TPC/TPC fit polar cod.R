## ============================ SETUP ============================

library(cmdstanr)
library(brms)
library(dplyr)

# --- Data ---
setwd("C:/Users/yzzha/OneDrive/Documents/AWI/Polar cod TPC/Polar cod TPC")
fs <- read.delim("egg data R.txt", header = TRUE)
eps <- 1e-6
fs$prop       <- pmin(pmax(fs$hatch, eps), 1 - eps)  # keep in (0,1)
fs$logit_prop <- qlogis(fs$prop)
fs$temp_c     <- fs$temp - mean(fs$temp)

## ============================ MODEL ============================

# Nonlinear Gaussian bump on the logit scale:
# logit(mu) = a + b * exp( -0.5 * ((temp_c - toptc)/exp(lsig))^2 )
bform <- bf(
  logit_prop ~ a + b * exp(-0.5 * ((temp_c - toptc)/exp(lsig))^2),
  a ~ 1, b ~ 1, toptc ~ 1, lsig ~ 1,
  nl = TRUE
)

# Priors
lsig_mu <- log(sd(fs$temp_c))  # numeric constant for prior mean
pri <- c(
  prior(normal(0, 2), nlpar = "a"),
  prior(normal(2, 2), nlpar = "b"),
  prior(normal(0, 2), nlpar = "toptc"),
  prior_string(sprintf("normal(%g, 1)", lsig_mu), nlpar = "lsig"),
  prior(exponential(1), class = "sigma")
)

options(brms.backend = "cmdstanr")

fit_brms <- brm(
  bform, data = fs, family = gaussian(), prior = pri,
  chains = 4, cores = 4, iter = 4000, seed = 42,
  refresh = 0,
  file = "tpc_gauss_logit", file_refit = "on_change",
  control = list(adapt_delta = 0.97, max_treedepth = 12)
)

mu_logit <- posterior_epred(fit_brms, newdata = fs)   # on logit_prop scale (Gaussian)
mu_prop  <- plogis(mu_logit)                          # back to 0..1

y <- fs$prop
R2_draws <- apply(mu_prop, 1, function(m) 1 - var(y - m) / var(y))
c(median = median(R2_draws), quantile(R2_draws, c(.05,.95)))

## =================== POSTERIOR PREDICTIONS =====================

grid <- data.frame(temp = seq(min(fs$temp), max(fs$temp), length.out = 1000))
grid$temp_c <- grid$temp - mean(fs$temp)

ETA <- posterior_linpred(fit_brms, newdata = grid)  # draws x grid (logit)
MU  <- plogis(ETA)                                  # back to [0,1]
t   <- grid$temp

# Credible band + central curve (median)
band_lo <- apply(MU, 2, quantile, 0.025)
band_hi <- apply(MU, 2, quantile, 0.975)
mu_ctr  <- apply(MU, 2, median)    # use colMeans(MU) if you prefer the mean
T_peak_ctr <- t[ which.max(mu_ctr) ]

# T* draws and 95% CI (used for 100% zone and minor zone inner edges)
T_star_draws <- apply(MU, 1, function(y) t[which.max(y)])
T_star_med   <- median(T_star_draws, na.rm = TRUE)
T_star_ci    <- quantile(T_star_draws, c(0.025, 0.975), na.rm = TRUE)
Topt_lo      <- unname(T_star_ci[1])
Topt_hi      <- unname(T_star_ci[2])

Tstar_table <- data.frame(
  metric   = "T* (optimum temperature)",
  median_C = T_star_med,
  lo95_C   = Topt_lo,
  hi95_C   = Topt_hi
) %>% mutate(across(where(is.numeric), ~ round(.x, 3)))
print(Tstar_table, row.names = FALSE)

# Levels for zones on the central curve
Smax  <- max(mu_ctr)
lev80 <- 0.80 * Smax
lev50 <- 0.50 * Smax

# Crossing helpers (linear interpolation)
cross_warm <- function(tt, yy, level){
  iop <- which.max(yy); if (iop >= length(tt)) return(NA_real_)
  i <- which(yy[iop:(length(yy)-1)] >= level & yy[(iop+1):length(yy)] < level)
  if (!length(i)) return(NA_real_)
  j0 <- iop + i[1] - 1; j1 <- j0 + 1
  x0 <- tt[j0]; x1 <- tt[j1]; y0 <- yy[j0]; y1 <- yy[j1]
  x0 + (level - y0) * (x1 - x0) / (y1 - y0)
}
cross_cold <- function(tt, yy, level){
  iop <- which.max(yy); if (iop <= 1) return(NA_real_)
  i <- which(yy[2:iop] >= level & yy[1:(iop-1)] < level)
  if (!length(i)) return(NA_real_)
  j1 <- i[length(i)] + 1; j0 <- j1 - 1
  x0 <- tt[j0]; x1 <- tt[j1]; y0 <- yy[j0]; y1 <- yy[j1]
  x0 + (level - y0) * (x1 - x0) / (y1 - y0)
}

# Crossings at 80% and 50% (both sides) on the central curve
T80_cold <- cross_cold(t, mu_ctr, lev80);  T80_warm <- cross_warm(t, mu_ctr, lev80)
T50_cold <- cross_cold(t, mu_ctr, lev50);  T50_warm <- cross_warm(t, mu_ctr, lev50)

## ===================== TEMPERATURE RANGES ======================

rng_ok <- function(a, b) if (is.finite(a) && is.finite(b) && a < b) c(a, b) else c(NA_real_, NA_real_)

# Minor (100–80%): from the edge of the 100% zone (T* CI) to the 80% crossing
minor_cold <- rng_ok(T80_cold, Topt_lo)
minor_warm <- rng_ok(Topt_hi,  T80_warm)

# Moderate (79–50%): between 80% and 50% boundaries
mod_cold   <- rng_ok(T50_cold, T80_cold)
mod_warm   <- rng_ok(T80_warm, T50_warm)

# Severe (<50%): outside the 50% boundaries
sev_cold   <- rng_ok(min(t),   T50_cold)
sev_warm   <- rng_ok(T50_warm, max(t))

ranges <- rbind(
  data.frame(zone = "Minor (100–80%)",   side = "cold", t_min_C = minor_cold[1], t_max_C = minor_cold[2],
             surv_min_pct = 80,  surv_max_pct = 100),
  data.frame(zone = "Minor (100–80%)",   side = "warm", t_min_C = minor_warm[1], t_max_C = minor_warm[2],
             surv_min_pct = 80,  surv_max_pct = 100),
  data.frame(zone = "Moderate (79–50%)", side = "cold", t_min_C = mod_cold[1],   t_max_C = mod_cold[2],
             surv_min_pct = 50,  surv_max_pct = 79),
  data.frame(zone = "Moderate (79–50%)", side = "warm", t_min_C = mod_warm[1],   t_max_C = mod_warm[2],
             surv_min_pct = 50,  surv_max_pct = 79),
  data.frame(zone = "Severe (<50%)",     side = "cold", t_min_C = sev_cold[1],   t_max_C = sev_cold[2],
             surv_min_pct = 0,   surv_max_pct = 49.999),
  data.frame(zone = "Severe (<50%)",     side = "warm", t_min_C = sev_warm[1],   t_max_C = sev_warm[2],
             surv_min_pct = 0,   surv_max_pct = 49.999)
)

# Console table (rounded) + write to file
ranges_fmt <- ranges %>% mutate(across(where(is.numeric), ~ round(.x, 3)))
print(ranges_fmt, row.names = FALSE)
write.table(ranges_fmt, "tpc_temperature_ranges.txt",
            sep = "\t", row.names = FALSE, quote = FALSE)

## ============================ PLOT =============================

# Optional symbols per Study (if present)
if (!"Study" %in% names(fs)) fs$Study <- factor("All data")
fs$Study <- factor(fs$Study)
studies  <- levels(fs$Study)
pch_vec  <- c(16, 17, 15, 18, 0, 1)           # extend if >6 studies
pch_map  <- setNames(pch_vec[seq_along(studies)], studies)

plot(NA, xlim = range(fs$temp), ylim = c(0, 100),
     xlab = "Temperature (°C)", ylab = "Egg survival (%)")

# 95% credible band (behind everything)
polygon(c(t, rev(t)),
        c(100*band_lo, rev(100*band_hi)),
        border = NA, col = rgb(0.2, 0.2, 0.8, 0.15))

# central fit (median)
lines(t, 100*mu_ctr, lwd = 2, col = "black")

# verticals: 100% zone (T* CI), then 80% (green) & 50% (orange) boundaries
abline(v = Topt_lo, col = "#2C7BE5", lty = 2, lwd = 2)  # blue (optimum zone edges)
abline(v = Topt_hi, col = "#2C7BE5", lty = 2, lwd = 2)

if (is.finite(T80_cold)) abline(v = T80_cold, col = "#2BB673", lty = 3, lwd = 2)
if (is.finite(T80_warm)) abline(v = T80_warm, col = "#2BB673", lty = 3, lwd = 2)

if (is.finite(T50_cold)) abline(v = T50_cold, col = "#F5A623", lty = 3, lwd = 2)
if (is.finite(T50_warm)) abline(v = T50_warm, col = "#F5A623", lty = 3, lwd = 2)

# optional horizontal guides at 80% and 50% of central max
abline(h = 100*lev80, col = "grey87", lty = 2)
abline(h = 100*lev50, col = "grey90", lty = 2)

# raw points by Study with different symbols
for (s in studies) {
  idx <- fs$Study == s
  points(fs$temp[idx], 100*fs$prop[idx],
         pch = pch_map[[s]], col = "grey30")
}

legend("bottomleft", title = "Study",
       legend = studies,
       pch = as.vector(pch_map[studies]),
       col = "grey30", pt.cex = 1, bty = "n")

## ============================ EXPORTS ==========================

# Fit grid with central curve (median) and 95% band
fit_out <- data.frame(
  temp_C           = t,
  central_fit_pct  = 100 * mu_ctr,
  lo95_pct         = 100 * band_lo,
  hi95_pct         = 100 * band_hi
)
write.table(fit_out, "tpc_central_fit_95CI.txt",
            sep = "\t", row.names = FALSE, quote = FALSE)

# T* summary to file (separate small table)
write.table(Tstar_table, "tpc_Tstar_summary.txt",
            sep = "\t", row.names = FALSE, quote = FALSE)

cat("Wrote files:\n",
    "  - tpc_central_fit_95CI.txt (grid, central fit, 95% band)\n",
    "  - tpc_temperature_ranges.txt (ranges: Minor 100–80, Moderate 79–50, Severe <50; minor starts outside 95% CI of T*)\n",
    "  - tpc_Tstar_summary.txt (posterior median and 95% CI of T*)\n", sep = "")

