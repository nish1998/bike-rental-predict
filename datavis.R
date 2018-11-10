data = read.csv("hour.csv", header = TRUE)

#histogram
ggplot(data = data) + 
  geom_bar(mapping = aes(x = weekday, y = cnt), stat = "identity")

#boxplots
p <- ggplot(data, aes(workingday, registered, group=workingday))
p + geom_boxplot()
#  scale_x_continuous(breaks = scales::pretty_breaks(n = 24))

ggplot(data = data) +
  geom_jitter(mapping = aes(x = instant, y = cnt))

#density plots
ggplot(data = data) +
  geom_bin2d(mapping = aes(x = windspeed, y = cnt)) +
  theme(aspect.ratio=1)+scale_fill_gradientn(colours=rainbow(7))

ggplot(data = data) + 
  geom_point(mapping = aes(x = windspeed, y = cnt), alpha = 1 / 10)

#histogram - one variable
ggplot(data) + 
  geom_bar(mapping = aes(x = cnt))

