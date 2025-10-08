library(tidyverse)

# mpg data visualization
View(mpg)

# Scatter Plots -----------

mpg |> # data
  ggplot(aes(cty, hwy)) + # plot base
  geom_point() # scatter plot

mpg |> #using color to distinguish vehicle classes
  ggplot(aes(cty, hwy, color = class)) + 
  geom_point()

mpg |>
  ggplot(aes(cty, hwy), color = class) + # why doesn't color get applied?
  geom_point()

mpg |>
  ggplot(aes(cty, hwy, color = "blue")) + # Do you see blue?
  geom_point()

mpg |>
  ggplot(aes(cty, hwy)) + 
  geom_point(color = "blue")

mpg |>
  ggplot(aes(cty, hwy, color = displ)) + # using color for displacement
  geom_point()

mpg |>
  ggplot(aes(cty, hwy, size = displ)) + # using size for displacement
  geom_point() # points are overlapped together

mpg |>
  ggplot(aes(cty, hwy, size = displ, color = displ)) +
  geom_point(alpha = 0.3) # adding transparency to the points

mpg |>
  ggplot(aes(cty, hwy, size = displ)) +
  geom_point() +
  scale_size_continuous(range = c(1, 3)) # adjusting the sizes

mpg |>
  ggplot(aes(cty, hwy)) + 
  geom_point() + 
  geom_text(aes(label = model)) # all the text labels are overlapped

mpg |>
  ggplot(aes(cty, hwy)) + 
  geom_point() + 
  geom_text(aes(label = model), 
            vjust = 1, # adjusting text vertically
            hjust = 0, # adjusting text horizontally
            check_overlap = T)  # removing the overlap 

mpg |>
  ggplot(aes(cty, hwy)) + 
  geom_point() +
  facet_wrap(~manufacturer) # faceted by manufacturer

mpg |>
  ggplot(aes(cty, hwy)) + 
  geom_point() +
  facet_wrap(~year) # faceted by year

mpg |>
  ggplot(aes(cty, hwy, color = year)) + 
  geom_point() +
  facet_wrap(~year) # any problems associated with this plot?

mpg |>
  ggplot(aes(cty, hwy, color = factor(year))) + # year should be factor
  geom_point(show.legend = F) +
  facet_wrap(~year) 

mpg |>
  ggplot(aes(cty, hwy, color = factor(year))) + 
  geom_point(show.legend = F) +
  facet_wrap(~year, ncol = 1) # stacking small multiples vertically

mpg |>
  ggplot(aes(cty, hwy, color = factor(year))) + 
  geom_point(show.legend = F) +
  facet_wrap(~year, ncol = 1) +
  scale_color_manual(values = c("orange", "blue")) #manually changing colors

mpg |>
  ggplot(aes(cty, hwy, color = factor(year))) + 
  geom_point(show.legend = F) +
  facet_wrap(~year, ncol = 1) +
  scale_color_viridis_d() #using built-in color palette

mpg |>
  ggplot(aes(cty, hwy)) +
  geom_point() + 
  labs(x = "city MPG", # adding plot labels by labs()
       y = "highway MPG",
       title = "City and Highway MPG Comparisons",
       subtitle = "The mpg data used",
       caption = "made by ggplot2")


mpg |>
  ggplot(aes(cty, hwy)) +
  geom_point() +
  geom_smooth(method = "loess", se = F) + # fitting the data using linear model
  labs(title = "Are City and Hwy MPGs Positively Corrleated?") 

# Other Plots ----------------------

mpg |>
  ggplot(aes(year, displ)) +
  geom_line() # anything wrong with the line plot? 

mpg |>
  ggplot(aes(year, displ)) +
  geom_boxplot() # how can we fix the plot?

mpg |>
  ggplot(aes(factor(year), displ)) +
  geom_boxplot() # it looks right after using factor for year

mpg |>
  ggplot(aes(factor(year), displ)) +
  geom_boxplot() + 
  labs(x = "year", # tidy up the plot
       y = "engine displacement")
  
mpg |>
  ggplot(aes(factor(year), displ, color = factor(year))) +
  geom_boxplot(show.legend = F) + 
  labs(x = "year",
       y = "engine displacement")

mpg |>
  ggplot(aes(factor(year), displ, fill = factor(year))) +
  geom_boxplot(show.legend = F) + 
  labs(x = "year",
       y = "engine displacement")

mpg |>
  ggplot(aes(factor(year), displ, color = factor(year), fill = factor(year))) +
  geom_boxplot(show.legend = F, alpha = 0.5) + #what happens if we remove alpha?
  labs(x = "year",
       y = "engine displacement")

mpg |>
  ggplot(aes(factor(year), displ, color = factor(year), fill = factor(year))) +
  geom_violin(show.legend = F, alpha = 0.5) + #switching to geom_violin
  labs(x = "year",
       y = "engine displacement")

mpg |>
  ggplot(aes(factor(year), displ, color = factor(year), fill = factor(year))) +
  geom_boxplot(show.legend = F, alpha = 0.5) + 
  theme_light() + # adding a different theme
  labs(x = "year",
       y = "engine displacement")


mpg |>
  ggplot(aes(factor(year), displ, color = factor(year), fill = factor(year))) +
  geom_boxplot(show.legend = F, alpha = 0.5) + 
  geom_point(show.legend = F) + # adding a scatterplot on top of boxplot
  theme_light() + # adding a different theme
  labs(x = "year",
       y = "engine displacement")

mpg |>
  count(manufacturer, drv) |>
  ggplot(aes(manufacturer, drv, fill = n)) +
  geom_tile() # the text on the x axis is hard to see

mpg |>
  count(manufacturer, drv) |>
  ggplot(aes(manufacturer, drv, fill = n)) +
  geom_tile() +
  theme(axis.text.x = element_text(angle = 90))

mpg |>
  count(manufacturer, drv) |>
  ggplot(aes(manufacturer, drv, fill = n)) +
  geom_tile() +
  coord_flip()


mpg |>
  count(manufacturer, drv) |>
  ggplot(aes(drv, manufacturer, fill = n)) + # swapping x and y positions
  geom_tile() +
  labs(x = "driving mode",
       y = NULL,
       fill = "count")


# diamonds data visualization
View(diamonds)

# Histograms -------

diamonds |>
  ggplot(aes(price)) +
  geom_histogram()

diamonds |>
  ggplot(aes(price)) +
  geom_histogram() + 
  scale_x_log10() # the x-axis on the log scale

diamonds |>
  ggplot(aes(price)) +
  geom_histogram(bins = 100) + 
  scale_x_log10() 

diamonds |>
  ggplot(aes(price)) +
  geom_histogram(bins = 30) + 
  scale_x_log10() +
  facet_wrap(~cut)

diamonds |>
  ggplot(aes(price, fill = color)) +
  geom_histogram(bins = 15, show.legend = F) + 
  scale_x_log10() +
  facet_wrap(~color) # D is best and J is worst

# Does cut impact diamond price?

diamonds |>
  ggplot(aes(price, cut)) +
  geom_boxplot() +
  scale_x_log10()

# Does weight impact diamond price?

diamonds |>
  ggplot(aes(carat, price)) +
  geom_point(alpha = 0.1) +
  #geom_smooth(method = "lm") +
  scale_y_continuous(labels = scales::dollar) + # add dollar signs
  labs(y = NULL)

# table, depth, and price

diamonds |>
  ggplot(aes(table, depth, color = price)) +
  geom_point(alpha = 0.1) 

diamonds |>
  ggplot(aes(table, depth)) +
  geom_bin2d(bins = 40)

# Does diamond color impact price?
diamonds |>
  ggplot(aes(price, color)) +
  geom_boxplot() +
  scale_x_log10() # D is best and J is worst

diamonds |>
  ggplot(aes(price, clarity, fill = clarity)) +
  geom_violin() +
  scale_x_log10() +#I1 (worst), IF (best)
  theme(legend.position = "none") # another way to remove legend

# x, y, z, and price
diamonds |>
  ggplot(aes(x, y, size = z, color = price)) +
  geom_point(alpha = 0.5)


# Maps -------
map_data("county", "illinois") |>
  ggplot(aes(long, lat, group = group)) + 
  geom_polygon(fill = "white", color = "green") # does it look like IL?


map_data("county", "illinois") |>
  ggplot(aes(long, lat, group = group)) + 
  geom_polygon(fill = "white", color = "green") +
  coord_map()
  
map_data("world") |>
  ggplot(aes(long, lat, group = group)) + 
  geom_polygon()

map_data("world") |>
  ggplot(aes(long, lat, group = group)) + 
  geom_polygon(aes(fill = region, color = region), show.legend = F) +
  theme_void()


