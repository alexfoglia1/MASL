#################################
# 
# Esercitazione sui modelli grafici 
#
#################################

library(ggm)

## The saturated model
ug1 <- UG(~ X*Y*Z)
ug1
drawGraph(ug1)
bd("X", ug1) # boundary

## X independent of Y given Z
ug2 <- UG(~ X*Z + Y*Z)
ug2
drawGraph(ug2)
bd(c("X", "Y"), ug2)

## Butterfly model defined from the cliques
ug3 <- UG(~ X*Y*Z + Z*W*V)
ug3
drawGraph(ug3)
bd("X", ug3)
bd(c("X", "Y"), ug3)
bd(c("X", "W"), ug3)


################## sample partial correlation distribution
library(mvtnorm)
p<-4
PC <- matrix(c(0.8, 0.5, 0, 0.6,
             0.5, 1.4, -0.6, 0.4,
             0, -0.6, 1.2, -0.3,
             0.6, 0.4, -0.3, 1), p, p) # inverse S matrix
S <- solve(PC)
S
R <- round(parcor(S),4)
R
M <- rep(0,p)

## draw samples from this distribution
n <- 3000 # 100 200 3000
niter <- 10000
r13 <- r32 <- c()
for (i in 1:niter){
X <- rmvnorm(n, mean=M, sigma=S)
VV<-parcor(var(X))
r13[i]<-VV[1,3]	
r32[i]<-VV[3,2]	
}
c(mean(r13), R[1,3])
c(mean(r32), R[3,2]) 

# quando H0 vera
hist(r13, prob=TRUE); lines(density(r13), col="red"); 
curve(dnorm(x, mean=R[1,3], sd=sqrt(var(r13))), col="blue", add=T)
# quando H0 falsa
hist(r32, prob=TRUE); lines(density(r32), col="red")
curve(dnorm(x, mean=R[3,2], sd=sqrt(var(r32))), col="blue", add=T)

# Fisher transformation quando H0 vera
Fr13 <- .5*log((1+r13)/(1-r13)) 
hist(Fr13, prob=TRUE, ylim=c(0,2.3)) 
hist(Fr13, prob=TRUE) 
lines(density(Fr13), col="red")
lines(density(r13), col="darkgreen") 
curve(dnorm(x, mean=R[1,3], sd=sqrt(1/(n-3-(p-2)))), col="blue", add=T)

# Fisher transformation quando H0 falsa
Fr32 <- .5*log((1+r32)/(1-r32)) 
hist(Fr32, prob=TRUE, ylim=c(0,20.6)) 
lines(density(Fr32), col="red")
lines(density(r32), col="darkgreen") 
curve(dnorm(x, mean=R[3,2], sd=sqrt(1/(n-3-(p-2)))), col="blue", add=T)

####################################
# Grafo pag. 

library(mvtnorm)
library(ggm)
n <- 200
p <- 4

PC <- matrix(c(0.8, 0.5, 0, 0.6,
             0.5, 1.4, -0.6, 0.4,
             0, -0.6, 1.2, -0.3,
             0.6, 0.4, -0.3, 1), p, p)
S <- solve(PC)
S
X <- rmvnorm(n, mean=rep(0,p), sigma=S)
pairs(X, panel=panel.smooth)

# varianza campionara
V <- var(X)
round(V,3)

round(parcor(V),3) # 1 ind 3 | 2,4

r.hat <- parcor(V)
r.t <- r.hat[1,3]
df <-n-p
t.test <- r.t * sqrt(df)/sqrt(1-r.t^2)
pt(t.test, df, lower.tail = TRUE)*2 # if t.test <0
pt(t.test, df, lower.tail = FALSE)*2 # if t.test >0

pcor.test(pcor(c(1,3,2,4), V), 2, n=n)

# z-transf.
z.test <- 0.5*log((1+r.t) / (1-r.t)) * sqrt(n - p -1)
pnorm(z.test, lower.tail = TRUE)*2

 # ?? 1 ind 3 | 2 ?
VV <- var(X[,1:3])
round(VV,3)
round(V[1:3,1:3],3)

round(parcor(VV),3) # No!

r.hat <- parcor(VV)
r.t <- r.hat[1,3]
df <-n-3+1
t.test <- r.t * sqrt(df)/sqrt(1-r.t^2)
pt(t.test, df, lower.tail = TRUE)*2
pcor.test(pcor(c(1,3,2), V), 1, n=n)
pcor.test(pcor(c(1,3,2), VV), 1, n=n)




#### esempio dati marks
library(ggm)
data(marks)
pairs(marks, panel=panel.smooth)

PC <-parcor(var(marks))
round(PC,3)
round(cor(marks),3)
round(correlations(marks),3) #sottotriangolare = corr; sovratiangolare=pcor

#Computes the partial correlation between the first two variables given the other variables in a set.
pcor(c("algebra","analysis", "statistics"), var(marks))
parcor(var(marks[,3:5]))
pcor(c("vectors","statistics","algebra"), var(marks))
pcor(c("vectors","statistics","analysis"), var(marks))
parcor(var(marks[,c(2,3,5)]))

# model selection from a complete graph
pcor(c(1,5,2,3,4), var(marks))
pcor.test(pcor(c(1,5,2,3,4), var(marks)), 3, n=88)
pcor.test(pcor(c(1,4,2,3,5), var(marks)), 3, n=88)
pcor.test(pcor(c(2,4,1,3,5), var(marks)), 3, n=88)
pcor.test(pcor(c(2,5,1,3,4), var(marks)), 3, n=88)
# eccetera

# stima da un dato grafo - estimates from a given graph
S <- var(marks)
stima <- fitConGraph(UG(~ mechanics*vectors*algebra+algebra*analysis*statistics), S, n=88)
round(solve(stima$Shat),4)
round(correlations(stima$Shat),3) 
drawGraph(UG(~ mechanics*vectors*algebra+algebra*analysis*statistics))
drawGraph(UG(~ mechanics*vectors*algebra+algebra*analysis*statistics), adjust = TRUE)
# attenzione : il metodo seguente richiederebbe n grande
library(SIN)
fisherz(cor(marks))
fisherz(parcor(var(marks)))
pvals <- sinUG(var(marks), n=88, holm=T)
pvals
plotUGpvalues(pvals)
getgraph(pvals, alpha=0.05, type="UG")
drawGraph(getgraph(pvals, alpha=0.05, type="UG")) 
drawGraph(getgraph(pvals, alpha=0.20, type="UG"))

# selezione del grafo
# fowlbones data : fowl = pollame
library(SIN)
data(fowlbones)
fowlbones

pvals <- sinUG(fowlbones$corr, fowlbones$n, holm=T)
plotUGpvalues(pvals)
edge.matr <- getgraph(pvals, alpha=0.10, type="UG")
edge.matr
drawGraph(edge.matr) # nel pacchetto ggm
rownames(edge.matr)<-colnames(edge.matr)<-c("SL", "SB", "H","U","F", "T")
edge.matr <- getgraph(pvals, alpha=0.10, type="UG")

stima<-fitConGraph(edge.matr, solve(fowlbones$corr), fowlbones$n)
stima
round(parcor(stima$Shat), 4)

##########################################
# Dati mtcars
########################################

#### dati mtcars
data(mtcars)
?mtcars

mtcars
dim(mtcars)
head(mtcars)
tail(mtcars)
summary(mtcars)

attach(mtcars)

############################################
# Dati stress (in ggm)
############################################
data(stress)
n=100
#####frozen shoulder########
#############################################
# Altri pacchetti
###############################################
# installazione
source("http://bioconductor.org/biocLite.R"); biocLite(c("graph","RBGL","Rgraphviz"))
install.packages("gRbase", dependencies=TRUE)
install.packages("gRain", dependencies=TRUE)
install.packages("gRim", dependencies=TRUE)

