#install.packages("tidyverse")
library(tidyverse)
library(ggplot2)
#data()
#?msleep
#view(msleep)
#names(msleep)
data1 <- read_csv('MovieCleaned.csv', show_col_types = FALSE)
head(data1, 27)

data1 %>%
  drop_na(rating) %>%
  ggplot(aes(x = rating))+
  geom_bar(fill = "pink")+
  theme_classic()+
  labs(x = "Rating",
       y = "Frequency",
       title = "Frequency of Ratings")


