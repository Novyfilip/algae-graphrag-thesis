library(tidyverse)

# Load results
path <- "C:/Users/filip/Desktop/Thesis/project/outputs/ragas_final"
testset <- read_csv("C:/Users/filip/Desktop/Thesis/project/outputs/ragas_testset.csv")

conditions <- c("baseline", "hybrid", "community", "graphrag")
df <- map_dfr(conditions, \(cond) {
  read_csv(file.path(path, paste0(cond, "_eval.csv"))) |>
    mutate(
      condition = cond,
      query_type = testset$synthesizer_name |>
        str_replace("single_hop_specific_query_synthesizer", "simple") |>
        str_replace("multi_hop_abstract_query_synthesizer", "abstract") |>
        str_replace("multi_hop_specific_query_synthesizer", "relational")
    )
})

# Two-way ANOVA: faithfulness ~ condition * query_type
model <- aov(faithfulness ~ condition * query_type, data = df)
summary(model)

# Post-hoc pairwise comparisons
TukeyHSD(model, "condition")
# MANOVA: all four metrics ~ condition * query_type
response <- cbind(df$faithfulness, df$answer_relevancy, df$context_precision, df$context_recall)
manova_model <- manova(response ~ condition * query_type, data = df)
summary(manova_model)
summary.aov(manova_model)