# Fit mixed-effects models. Run: Rscript mixed_models.R <csv_dir>
suppressMessages({library(lme4); have_tmb <- requireNamespace("glmmTMB", quietly = TRUE)})
args <- commandArgs(TRUE); d0 <- args[1]
out <- list()
for (f in list.files(d0, pattern = "^mm_.*\\.csv$", full.names = TRUE)) {
  d <- read.csv(f); d$arm <- factor(d$arm); d$cell <- factor(d$cell)
  nat <- grepl("_NAT_", f); multi <- length(unique(d$pair)) > 1
  fml <- if (nat) {
    if (multi) y ~ arm * cell + factor(order) + (1|image_id) + (1|qid) else y ~ arm + factor(order) + (1|image_id) + (1|qid)
  } else {
    if (multi) y ~ arm * cell + (1|image_id) + (1|image_id:qid) else y ~ arm + (1|image_id) + (1|image_id:qid)
  }
  m <- tryCatch(glmer(fml, data = d, family = binomial, control = glmerControl(optimizer = "bobyqa")),
                error = function(e) NULL)
  meth <- "glmer"
  if (is.null(m) || length(m@optinfo$conv$lme4$messages) > 0) {
    meth <- "glmmTMB"; m <- if (have_tmb) tryCatch(glmmTMB::glmmTMB(fml, data = d, family = binomial), error = function(e) NULL) else NULL
  }
  if (is.null(m)) { meth <- "GEE (see Python output)"; out[[basename(f)]] <- data.frame(file = basename(f), method = meth); next }
  co <- summary(m)$coefficients; if (is.list(co) && !is.matrix(co)) co <- co$cond
  out[[basename(f)]] <- data.frame(file = basename(f), method = meth, term = rownames(co), est = co[, 1], se = co[, 2])
}
write.csv(do.call(rbind, out), file.path(d0, "mixed_models_lme4_results.csv"), row.names = FALSE)
