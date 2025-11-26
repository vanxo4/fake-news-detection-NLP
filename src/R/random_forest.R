################################################################################
# NUMERIC CLASSIFICATION BASELINE (RANDOM FOREST)
# 1. Data Setup (Load & Clean)
# 2. Train/Test Split (70/30)
# 3. Model Training (Random Forest)
# 4. Evaluation (Confusion Matrix)
# 5. Interpretation (Feature Importance)
################################################################################

# --- 1. Setup & Dependencies ---

if (!require("randomForest")) install.packages("randomForest")
if (!require("caret")) install.packages("caret")
if (!require("e1071")) install.packages("e1071")

library(randomForest)
library(caret)
library(here)

DATA_PATH  <- here("data", "processed", "news_features_numeric.csv")
MODEL_PATH <- here("models", "rf_numeric_v1.rds")
  
cat("Loading dataset from:", DATA_PATH, "\n")
if (!file.exists(DATA_PATH)) stop("Dataset not found! Check the path.")

df <- read.csv(DATA_PATH)

# Preprocessing for classification
# Convert target variable to Factor (Categorical)
df$is_fake <- as.factor(df$is_fake)

# Remove ID column (non-predictive)
df$id <- NULL

# Check Class Balance (Fake vs True count)
cat("Class Balance:\n")
print(table(df$is_fake))

# --- 2. Train / Test Split ---

set.seed(123) # Ensure reproducibility

# 70% Training, 30% Testing
trainIndex <- createDataPartition(df$is_fake, p = 0.7, 
                                  list = FALSE, 
                                  times = 1)

data_train <- df[ trainIndex,]
data_test  <- df[-trainIndex,]

cat("\nTraining Set Rows:", nrow(data_train), "\n")
cat("Testing Set Rows: ", nrow(data_test), "\n")

# --- 3. Model Training ---

cat("\nTraining Random Forest model... (ntree=100)\n")

# Training the model
# ntree=100: Kept low for quick baseline testing (standard is 500)
# importance=TRUE: Calculates variable importance for interpretation
rf_model <- randomForest(is_fake ~ ., 
                         data = data_train, 
                         ntree = 100,       
                         importance = TRUE)

# --- 4. Prediction & Evaluation ---

predicciones <- predict(rf_model, data_test)

cat("\n--- Confusion Matrix & Metrics ---\n")
conf_matrix <- confusionMatrix(predicciones, data_test$is_fake)
print(conf_matrix)

# --- 5. Feature Importance Interpretation ---

cat("\n--- Top Predictive Variables ---\n")

# Visual Plot
varImpPlot(rf_model, main = "Feature Importance (Fake News Detection)")

# Quantitative Ranking (sorted by Mean Decrease Gini)
importance_df <- as.data.frame(importance(rf_model))
importance_df <- importance_df[order(-importance_df$MeanDecreaseGini),] 

print(head(importance_df, 10))

# --- 7. Model Persistence (Export) ---

cat("\nSaving model to:", MODEL_PATH, "\n")
saveRDS(rf_model, file = MODEL_PATH)
cat("✅ Model saved successfully.\n")