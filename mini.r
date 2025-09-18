# ----------------------------
# Student Performance Prediction
# Data Mining Project in R
# ----------------------------

# 1. Load Libraries
install.packages(c("caret", "randomForest", "ggplot2"), dependencies = TRUE)
library(caret)
library(randomForest)
library(ggplot2)

# 2. Load Dataset
setwd("C:/Users/Jayalakshmi.M/Downloads/")  # change path
data <- read.csv("StudentsPerformance.csv", stringsAsFactors = TRUE)

# 3. View & Clean Data
head(data)
summary(data)
str(data)

# Handle missing values (if any)
data <- na.omit(data)

# Convert categorical to factors (if not already)
data$gender <- as.factor(data$gender)
data$race.ethnicity <- as.factor(data$race.ethnicity)
data$parental.level.of.education <- as.factor(data$parental.level.of.education)
data$lunch <- as.factor(data$lunch)
data$test.preparation.course <- as.factor(data$test.preparation.course)

# 4. Create Target Variable: Overall Performance
data$average_score <- rowMeans(data[, c("math.score", "reading.score", "writing.score")])

data$performance <- cut(
  data$average_score,
  breaks = c(-Inf, 60, 75, 100),
  labels = c("Low", "Medium", "High")
)

table(data$performance)

# 5. Exploratory Data Analysis
ggplot(data, aes(x = performance, fill = gender)) +
  geom_bar() +
  labs(title = "Performance Distribution by Gender")

ggplot(data, aes(x = parental.level.of.education, y = average_score, fill = parental.level.of.education)) +
  geom_boxplot() +
  labs(title = "Parental Education vs Student Score") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 6. Train-Test Split
set.seed(123)
trainIndex <- createDataPartition(data$performance, p = 0.7, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# 7. Model Building - Random Forest
rf_model <- randomForest(performance ~ gender + race.ethnicity +
                           parental.level.of.education + lunch +
                           test.preparation.course + math.score + reading.score + writing.score,
                         data = trainData, ntree = 500, mtry = 3, importance = TRUE)

print(rf_model)

# 8. Model Prediction
predictions <- predict(rf_model, newdata = testData)
confusionMatrix(predictions, testData$performance)

# 9. Feature Importance
importance(rf_model)
varImpPlot(rf_model)

# 10. Save Outputs for Power BI
write.csv(data, "cleaned_student_performance.csv", row.names = FALSE)
write.csv(data.frame(Actual = testData$performance, Predicted = predictions),
          "student_predictions.csv", row.names = FALSE)

cat("✅ Data mining completed. Cleaned data & predictions saved for Power BI.\n")
