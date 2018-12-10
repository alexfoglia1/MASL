#####################################
#
#  Regularized estimators
#
#####################################
rm(list = ls())

#################
# The Validation Set Approach

library(ISLR)
?Auto

## select validation sets
set.seed(1)
?sample
train<-sample(392,196)

attach(Auto)
lm.fit<-lm(mpg~horsepower,data=Auto,subset=train)


mean((mpg-predict(lm.fit,Auto))[-train]^2)
lm.fit2<-lm(mpg~poly(horsepower,2),data=Auto,subset=train)
mean((mpg-predict(lm.fit2,Auto))[-train]^2)
lm.fit3<-lm(mpg~poly(horsepower,3),data=Auto,subset=train)
mean((mpg-predict(lm.fit3,Auto))[-train]^2)

rbind(c("Linear", "Dregree2", "Degree3"),c(mean((mpg-predict(lm.fit,Auto))[-train]^2),mean((mpg-predict(lm.fit2,Auto))[-train]^2),mean((mpg-predict(lm.fit3,Auto))[-train]^2)), c(mean((mpg-predict(lm.fit,Auto))[train]^2),mean((mpg-predict(lm.fit2,Auto))[train]^2),mean((mpg-predict(lm.fit3,Auto))[train]^2) ))

# Plot
plot(horsepower,mpg)
horsepowerXX <- min(horsepower):max(horsepower)
mpgYY <- predict(lm.fit, newdata = list(horsepower=horsepowerXX))
mpgYY2 <- predict(lm.fit2, newdata = list(horsepower=horsepowerXX))
mpgYY3 <- predict(lm.fit3, newdata = list(horsepower=horsepowerXX))
lines(horsepowerXX, mpgYY, col="blue")
lines(horsepowerXX, mpgYY2, col="red")
lines(horsepowerXX, mpgYY3, col="darkgreen")

# try again
set.seed(2)
train<-sample(392,196)
lm.fit<-lm(mpg~horsepower,subset=train)
mean((mpg-predict(lm.fit,Auto))[-train]^2)
lm.fit2<-lm(mpg~poly(horsepower,2),data=Auto,subset=train)
mean((mpg-predict(lm.fit2,Auto))[-train]^2)
lm.fit3<-lm(mpg~poly(horsepower,3),data=Auto,subset=train)
mean((mpg-predict(lm.fit3,Auto))[-train]^2)

## LOOCV
require(ISLR)
require(boot)
?cv.glm

plot(mpg~horsepower,data=Auto)
glm.fit<-glm(mpg~horsepower, data=Auto)
cv.glm(Auto,glm.fit)
cv.glm(Auto,glm.fit)$K
cv.glm(Auto,glm.fit)$delta #pretty slow (doesn't use formula (5.2) on page 180)

##Lets write a simple function to use short formula 
loocv <- function(fit){
  h<-lm.influence(fit)$h #fitted
  mean((residuals(fit)/(1-h))^2)
}

loocv(glm.fit)

# cv error su regressione polinomiale
cv.error <- rep(0,5)
degree <- 1:5
for(d in degree){
  glm.fit <- glm(mpg~poly(horsepower,d), data=Auto)
  cv.error[d] <- loocv(glm.fit)
}
plot(degree,cv.error,type="b")

######### Validation

################
## 10-fold CV

cv.error10 <- rep(0,5)
for(d in degree){
  glm.fit <- glm(mpg~poly(horsepower,d), data=Auto)
  cv.error10[d] <- cv.glm(Auto,glm.fit,K=10)$delta[1]
}
lines(degree,cv.error10,type="b",col="red")

glm.fit2 <- glm(mpg~poly(horsepower,2), data=Auto)
glm.fit5 <- glm(mpg~poly(horsepower,5), data=Auto)
summary(glm.fit2)
summary(glm.fit5)
anova(glm.fit2, glm.fit5)

plot(Auto$horsepower, Auto$mpg)
lines(horsepowerXX, predict(glm.fit5, newdata = list(horsepower=horsepowerXX)), col="magenta")
lines(horsepowerXX, mpgYY2, col="blue")

c(min(Auto$horsepower), max(Auto$horsepower))
xxx0<-min(Auto$horsepower):max(Auto$horsepower)

xxx<- poly(xxx0,5)[,1:5]

yyy<- glm.fit5$coeff[1] +glm.fit5$coeff[2]*xxx[,1] + glm.fit5$coeff[3]*xxx[,2] + glm.fit5$coeff[4]*xxx[,3] + glm.fit5$coeff[5]*xxx[,4] +glm.fit5$coeff[6]*xxx[,5]

plot(Auto$horsepower, Auto$mpg)
lines(xxx0,yyy, col="red")


#simulate data from logistic simple regression
set.seed(2615)
n <- 1000
b0 <- 1
b1 <- 1
X <- rnorm(n)
eta <- exp(b0+b1*X)
px <- eta/(1+eta)
y <- rbinom(n,1,px)
B <- 1000
m.sim<-c()
for (i in 1:B){
  samples<-sample(n,n,replace=TRUE)
  m.sim[i] <- glm(y[samples]~X[samples], family = "binomial")$coefficients[2]
}
mean(m.sim)
sd(m.sim)
sum(y)/n
mod0 <- glm(y~X,family="binomial")
summary(mod0)
###########################
## Bootstrap
##### Simulated data

# true model
set.seed(1526)
#set.seed(26) # too lucky
N <- 100
M <- 2.34
Sd <- 1.2
x <- rnorm(N, mean=M, sd=Sd)
hist(x, col="steelblue1", prob=TRUE, ylim = c(0,.4))
lines(density(x, adjust = 1), col="darkblue", lty=2)
curve(dnorm(x, mean=M, sd=Sd),lty=1, col="red", add=T )
abline(v=M, col="red")
#  observed mean
m <- mean(x)  
m
abline(v=m, col="darkblue", lty=2)
### manually
B <- 1000
m.B<-c()
for (i in 1:B){
	dati<-sample(x,N,replace=TRUE)
	m.B[i] <- mean(dati) 
}
hist(m.B, probability = TRUE, col="orchid")
abline(v=mean(m.B))
abline(v=m, col="darkblue", lty=2) 
abline(v=M, col="red") # notice, bias w/r to sample not population
mean(m.B)
m
sqrt(var(m.B))
Sd/sqrt(N)
# Theory: Bias = 0, Var = 1.2^2/100 = 0.0144 

####################################
# Bootstrap with the boot library
library(boot)
statistic = function(d, w) mean(d[w]) #??d=data, w=subset
statistic(x)
statistic(x,1:10)
mean(x[1:10])

x.boot <- boot(x, statistic, R = 1000)
plot(x.boot)
x.boot

# Bias computed on the original sample
mean(x.boot$t) - x.boot$t0

# Variance
var(x.boot$t)
sqrt(var(m.B))


############################################
# Compute confidence intervals for the median
# or for the variance for not-Gaussian data
set.seed(26)
x2 <- rt(N,df=2)
hist(x2)
var(x2)
# by hand
# let us learn "replicate"
?replicate
var.B <- replicate(1000, var(sample(x2,N,replace=TRUE)))
#Conf-int 
quantile(var.B, probs=c(0.025, 0.975)) 

# with boot
statistic <- function(d, w) var(d[w])
statistic(x2)
x2.boot <- boot(x2, statistic, R = 1000)

ci.boot <- boot.ci(x2.boot, type="all")

############################################
## Minimum risk investment - Section 5.2

alpha=function(x,y){
  vx=var(x)
  vy=var(y)
  cxy=cov(x,y)
  (vy-cxy)/(vx+vy-2*cxy)
}
alpha(Portfolio$X,Portfolio$Y)

## What is the standard error of alpha?

alpha.fn=function(data, index){
  with(data[index,],alpha(X,Y))
}

alpha.fn(Portfolio,1:100)

set.seed(1)
alpha.fn (Portfolio,sample(1:100,100,replace=TRUE))

boot.out=boot(Portfolio,alpha.fn,R=1000)
boot.out
plot(boot.out)


# Estimating the Accuracy of a Linear Regression Model

boot.fn=function(data,index)
  return(coef(lm(mpg~horsepower,data=data,subset=index)))
boot.fn(Auto,1:392)
set.seed(1)
boot.fn(Auto,sample(392,392,replace=T))
boot.fn(Auto,sample(392,392,replace=T))
boot(Auto,boot.fn,1000)
summary(lm(mpg~horsepower,data=Auto))$coef
boot.fn=function(data,index)
  coefficients(lm(mpg~horsepower+I(horsepower^2),data=data,subset=index))
set.seed(1)
boot(Auto,boot.fn,1000)
summary(lm(mpg~horsepower+I(horsepower^2),data=Auto))$coef

################################################
########### Best Subset regression
library(ISLR)
help(Hitters)
summary(Hitters)

# delete missing values
Hitters <- na.omit(Hitters)
sum(is.na(Hitters$Salary))
## or w/ with
with(Hitters,sum(is.na(Salary)))

library(leaps)
?regsubsets
regfit.full<-regsubsets(Salary~.,data=Hitters)
summary(regfit.full) # It gives best-subsets up to size 8
regfit.full=regsubsets(Salary~.,data=Hitters, nvmax=19)
reg.summary=summary(regfit.full)
names(reg.summary) # names of included variables
plot(reg.summary$cp,xlab="Number of Variables",ylab="Cp", type = "l")
points(reg.summary$cp,pch=20,col="black")
which.min(reg.summary$cp)
points(10,reg.summary$cp[10],pch=20,col="red")

plot(regfit.full,scale="Cp")
coef(regfit.full,10)
vcov(regfit.full,10)

####################
#Forward Stepwise Selection
regfit.fwd <- regsubsets(Salary~.,data=Hitters,nvmax=19,method="forward")
summary(regfit.fwd)
plot(regfit.fwd,scale="Cp")

mod10 <- lm(Salary~AtBat+Hits+Walks+CAtBat+CRuns+CRBI+CWalks+Division+PutOuts+Assists, data=Hitters)
summary(mod10)

####################
# Model Selection Using a Validation Set
dim(Hitters)
set.seed(1)
train=sample(seq(263),180,replace=FALSE)
train
regfit.fwd=regsubsets(Salary~.,data=Hitters[train,],nvmax=19,method="forward")
summary(regfit.fwd)

val.errors=rep(NA,19)
x.test=model.matrix(Salary~.,data=Hitters[-train,])# notice the -index!
for(i in 1:19){
  coefi=coef(regfit.fwd,id=i)
  pred=x.test[,names(coefi)]%*%coefi
  val.errors[i]=mean((Hitters$Salary[-train]-pred)^2)
}
ya <- min(c(sqrt(val.errors),sqrt(regfit.fwd$rss[-1]/180)))-10
yb <- max(c(sqrt(val.errors),sqrt(regfit.fwd$rss[-1]/180)))+10


plot(sqrt(val.errors),ylab="Root MSE",ylim=c(ya,yb),pch=19,type="b")
points(sqrt(regfit.fwd$rss[-1]/180),col="blue",pch=19,type="b")
points(which.min(sqrt(val.errors)), min(sqrt(val.errors)), col="red", pch=19)
legend("topright",legend=c("Training","Validation"),col=c("blue","black"),pch=19)


### Model Selection by Cross-Validation
set.seed(11)
folds=sample(rep(1:10,length=nrow(Hitters)))
folds
table(folds)
cv.errors=matrix(NA,10,19)
for(k in 1:10){
  best.fit=regsubsets(Salary~.,data=Hitters[folds!=k,],nvmax=19,method="forward")
  x.test=model.matrix(Salary~.,data=Hitters[folds==k,])
  for(i in 1:19){
    coefi=coef(best.fit,id=i)
    pred=x.test[,names(coefi)]%*%coefi
    cv.errors[k,i]=mean((Hitters$Salary[folds==k]-pred)^2)
  }
}
rmse.cv=sqrt(apply(cv.errors,2,mean))
plot(rmse.cv,pch=19,type="b")
which.min(rmse.cv)

################################
######## Ridge Regression and the Lasso
library(glmnet)
x=model.matrix(Salary~.-1,data=Hitters) 
y=Hitters$Salary
fit.ridge=glmnet(x,y,alpha=0)
plot(fit.ridge,xvar="lambda",label=TRUE)

# use cross-validation
cv.ridge=cv.glmnet(x,y,alpha=0)
cv.ridge
plot(cv.ridge)

# in the plot there are 2 vertical lines for
cv.ridge$lambda.min
log(cv.ridge$lambda.min)
cv.ridge$lambda.1se
log(cv.ridge$lambda.1se)

round(cbind(coef(cv.ridge, s=cv.ridge$lambda.min), coef(cv.ridge, s=cv.ridge$lambda.1se)),4)

#Now we fit a lasso model
#for this we use the default `alpha=1`

fit.lasso=glmnet(x,y, alpha=1)
plot(fit.lasso,xvar="lambda",label=TRUE)
plot(fit.lasso,xvar="lambda",label=TRUE, ylim=c(-5,5))
names(fit.lasso)
fit.lasso$lambda

# matrix of coefficients
dim(fit.lasso$beta)
fit.lasso$beta[,28]

# cross-validation
cv.lasso=cv.glmnet(x,y, nfolds = 10)
plot(cv.lasso)
coef(cv.lasso)
names(cv.lasso$glmnet.fit)
cv.lasso$nzero # non-zero coeff
plot(cv.lasso$lambda, cv.lasso$nzero, pch=20, col="red")

##### validation set
lasso.tr=glmnet(x[train,],y[train])
lasso.tr
pred=predict(lasso.tr,x[-train,])
dim(pred)
rmse= sqrt(apply((y[-train]-pred)^2,2,mean))
plot(log(lasso.tr$lambda),rmse,type="b",xlab="Log(lambda)")
lam.best=lasso.tr$lambda[order(rmse)[1]]
lam.best
coef(lasso.tr,s=lam.best)

## Let us explore the role of the variance of the covariates
# simulate with
n <- 1000
p=50
# from a Gaussian distribution with diag Sigma with sigma_jj = j/10

# Y = Xb + e,  all b=1 and e ~standard normal

# move lamda



################
#Analyze prostate cancer data
#################################

# An an example, consider the data from a 1989 study examining the relationship prostate-specific antigen (PSA) and a number of clinical measures in a sample of 97 men who were about to receive a radical prostatectomy
# PSA is typically elevated in patients with prostate cancer, and serves a biomarker for the early detection of the cancer
# The explanatory variables:
# lcavol: Log cancer volume
# lweight: Log prostate weight
# age
# lbph: Log benign prostatic hyperplasia
# svi: Seminal vesicle invasion
# lcp: Log capsular penetration
# gleason: Gleason score
# pgg45: % Gleason score 4 or 5
# Response : lpsa

prostate <- read.table(file.choose(), header = TRUE)
head(prostate)

library(MASS)
lambda <- exp(seq(log(0.01),log(2000),len=99))
mod0 <- lm.ridge(lpsa~., prostate, lambda=lambda)
summary(mod0)

matplot(mod0$lambda, coef(mod0)[,-1], type="l", lty=1, lwd=3, ylab=~beta, xlab=~lambda)
matplot(mod0$lambda, coef(mod0)[,-1], type="l", lty=1, lwd=3, ylab=~beta, xlab=~lambda, log="x")
cbind(RIDGE = coef(mod0)[which.min(mod0$GCV),], OLS=coef(lm(lpsa~., prostate))) 

## Lasso
librari(glmnet)
Xvar <- model.matrix(lpsa~0+.,prostate)
Yvar <- prostate$lpsa

modL <- glmnet(Xvar,Yvar)
plot(modL)

cvmodL <- cv.glmnet(Xvar,Yvar)
plot(cvmodL)



