library(tidyverse)

df <- read_csv("C:/Users/filip/Desktop/Thesis/project/outputs/ragas_extended/manova_input_203.csv")

response <- cbind(df$faithfulness, df$answer_relevancy, df$context_precision, df$context_recall)
model <- manova(response ~ condition * Type, data = df)
summary(model)
summary.aov(model)
TukeyHSD(aov(faithfulness ~ condition * Type, data = df), "condition")

# Factorial interaction test
df$graph <- as.factor(ifelse(df$condition %in% c("hybrid", "graphrag"), 1, 0))
df$community <- as.factor(ifelse(df$condition %in% c("community", "graphrag"), 1, 0))

# Sanity check — should match Table 1
tapply(df$faithfulness, list(df$graph, df$community), mean)

# The test
summary(aov(faithfulness ~ graph * community, data = df))