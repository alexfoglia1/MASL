library(gRbase)
library(gRain)
n <- 500
x1 <- rnorm(n)
a <- 3
b <- 5
x2 <- a*x1 + rnorm(n)
x3 <- a*x1 + rnorm(n)
x4 <- a*x1 + rnorm(n)
y = b*x2 + b*x3 + b*x4 + rnorm(n)

y <- matrix(y, nrow = 100, byrow = TRUE)


mytree <- rpart::rpart(y~x1+x2+x3+x4)
rpart.plot::prp(mytree)
cor(cbind(y,x1, x2, x3, x4))

