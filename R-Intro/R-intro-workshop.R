library(ggplot2)

ggplot(data = mpg, aes(cty, hwy, color = drv)) +
  geom_point()


# entering inputs

x <- 6
x #implicit printing
print(x) #explicit printing

message <- "hello world!"
message
print(message)

y <- 11:30
y

#string
"abc"#double quote and single quote serve the same purposes
'abc'
"123"

typeof("123")

#numeric
123
typeof(123)

#integer
123L
typeof(123L)

# NAN and Inf
0/0

1/0

1/ Inf

Inf * 0

NaN * 0

Inf - Inf

#logical
TRUE 
T
FALSE
F

typeof(123)

typeof(123L)

typeof(1 + 3i)

typeof(TRUE)

typeof(F)

# Vectors

numeric_vec <- c(1,2,3,4)
numeric_vec
numeric_vec[1]
numeric_vec[3]
numeric_vec[5]
length(numeric_vec)

char_vec <- c("a", "b", "c")
char_vec

mix_vec <- c(TRUE, "a")
mix_vec
typeof(mix_vec)

# Factors
sizes <- c("small", "medium", "large")

sizes

"medium" > "small" # Is it true? 

real_sizes <- factor(sizes, levels = c("small", "medium", "large"))

real_sizes[2] > real_sizes[1] # Is this true?

real_sizes <- factor(sizes, levels = c("small", "medium", "large"), ordered = TRUE)

real_sizes[2] > real_sizes[1]

# Lists

ls <- list(c(1, 2), "a", T, 1+4i)
ls

ls[1]

ls[[1]]

ls[3]

ls[[3]]

# Data frames
mtcars

?mtcars

dim(mtcars)

ncol(mtcars)

nrow(mtcars)

names(mtcars)

# Tibble

tibble::tibble(mtcars)

tbl <- tibble::tibble(mtcars)
#tibble(rownames_to_column(mtcars, var = "make"))

# Date
library(lubridate)

today()

now()

date <- as.Date("2023-01-01")
date

time_stamp <- as.POSIXlt("2023-01-01 13:01:20")
time_stamp

# check out the lubridate package 


# Missing values 

NA == 5

NA > 6

NA + 5

#check if a variable is NA
is.na(NA)

is.na(5)

# functions

add_one <- function(x){
  
  return(x + 1)
  
}


add_one(1)

add_one(5)

# functional programming (advanced concept)
purrr::map_dbl(c(1:5), add_one)

# R pipes

# h(g(f(x)))

# x |>
#  f() |>
#  g() |>
#  h()

# a concrete pipe example
mpg |>
  ggplot(aes(cty, hwy, color = drv)) +
  geom_point()

# Tidyverse
library(tidyverse)










