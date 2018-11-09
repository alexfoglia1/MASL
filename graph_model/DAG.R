#################################
# 
# Esercitazione sui DAG 
#
#################################

rm(list=ls())

###############
library(gRbase)
library(gRain)
help(package=gRbase)

#### How to construct a DAG
dag0 <- dag(~a, ~b * a, ~c * a * b, ~d * c * e, ~e * a)
iplot(dag0) # note how the script works: ~c * a * b = c depends on both a and b
dag0

edgeList(dag0)
as.adjMAT(dag0)

parents("d",dag0)
children("a",dag0)

ug0 <- moralize(dag0)
iplot(ug0)
edgeList(ug0)
getCliques(ug0)

# check: b ind e ? 
ancestralSet(c("b", "e"), dag0)
iplot(ancestralGraph(c("b", "e"), dag0))
iplot(moralize(ancestralGraph(c("b", "e"), dag0))) # --> no

# check: (d,e) ind b | c ? 
ancestralSet(c("b", "e", "d", "c"), dag0)
iplot(ancestralGraph(c("b", "e", "d", "c"), dag0))
iplot(moralize(ancestralGraph(c("b", "e", "d", "c"), dag0))) 
# --> no (b,e) ind d | c, a 

# check: a ind d 
ancestralSet(c("a", "d"), dag0)
iplot(ancestralGraph(c("a", "d"), dag0))
iplot(moralize(ancestralGraph(c("a", "d"), dag0)))

################# data marks
library(ggm)
data(marks)
S <- var(marks)
stima<-fitConGraph(UG(~ mechanics*vectors*algebra+algebra*analysis*statistics), S, n=88)
round(solve(stima$Shat),4)
round(correlations(stima$Shat),3) 
iplot(ug(~ mechanics:vectors:algebra, ~ algebra:analysis:statistics))

# Let us find a possible dag
library(pcalg)
data(marks)
?pc
eq.dag <- pc(suffStat = list(C=cor(marks), n=88), indepTest=gaussCItest, p=ncol(marks), alpha=0.10, verbose = TRUE)
eq.dag
plot(eq.dag, main="PC DAG") #acc

# Let us suppose an exame order: algebra, analysis, vectors, statistics, mechanics
# A possible DAG is: (use ggm)
gdag1 <- DAG( analysis ~ algebra, statistics ~ algebra:analysis:vectors, mechanics ~ vectors) 
fdag1 <- fitDag(gdag1, S, nrow(marks))
fdag1$dev
iplot(dag( ~ analysis*algebra,  ~ statistics*algebra*analysis*vectors,  ~ mechanics*vectors))

# Manual altrenative
mod1.0<-lm(analysis ~ algebra, data=marks)
summary(mod1.0)
mod2.0 <- lm(vectors ~algebra + analysis, data=marks)
summary(mod2.0)
mod3.0 <- lm(statistics~ vectors + algebra + analysis, data=marks)
summary(mod3.0)
mod4.0 <- lm(mechanics ~ statistics + vectors + algebra + analysis, data=marks)
summary(mod4.0)
mod4.1 <- lm(mechanics ~ vectors + algebra , data=marks)
anova(mod4.1, mod4.0)

gdag2 <- DAG( analysis ~ algebra, vectors ~algebra:analysis, statistics ~ algebra:analysis, mechanics ~ vectors:algebra) 
drawGraph(gdag2)
fdag2 <- fitDag(gdag2, S, nrow(marks))
fdag2$dev
fdag1$dev
###################################
# Another example - NB non viene niente...
# set your working directory
dati <- read.table("survey.txt", header = TRUE)
#A : adult old young # (3) Age
#S : F M             # (2) Sex
#E : high uni        # (2) Education level
#R : big small       # (2) Residence city size
#O : emp self        # (2) Occupation
#T : car other train # (3) mainly used Transport
dati
#####let's start with a possible dag
library(ggm)
# Possible ordering : S A (root), E, R, O, T
#possible graph
drawGraph(DAG(E ~ S + A, R ~ E, O ~ E, T ~ E + O + R))
# procediamo a mano dal dag pieno dato l'ordinamento : S A (root), E, R, O, T

mod0 <-glm(S ~ A , family=binomial, data=dati) # per controllo
summary(mod0)

mod1 <-glm(E ~ S + A, family=binomial, data=dati)
summary(mod1)

mod2 <- glm(R ~ S + A + E, family=binomial, data=dati)
summary(mod2)
mod2.0 <- glm(R ~  A + E, family=binomial, data=dati)
summary(mod2.0)
anova(mod2.0, mod2, test="LR") # OK
mod2.1 <- glm(R ~  E, family=binomial, data=dati)
summary(mod2.1)
anova(mod2.1, mod2, test="LR") # KO

mod3 <- glm(O ~ S + A + E + R, family=binomial, data=dati)
summary(mod3)
mod3.0 <- glm(O ~  A + E , family=binomial, data=dati)
summary(mod3.0)
anova(mod3.0, mod3, test="LR") # OK

mod4 <- glm(T ~ S + A + E + R + O, family=binomial, data=dati)
summary(mod4)
mod4.0 <- glm(T ~  A + E + R + O, family=binomial, data=dati)
summary(mod4.0)
anova(mod4.0, mod4, test="LR") # OK
mod4.1 <- glm(T ~  E + R + O, family=binomial, data=dati)
summary(mod4.1)
anova(mod4.1, mod4, test="LR") # OK
#
mod4.2 <- glm(T ~ 1, family=binomial, data=dati)
summary(mod4.2)
anova(mod4.2, mod4, test="LR") # OK

g1<-DAG(E ~ S + A, R ~ A + E, O ~ A + E, T ~ R)
g1
g1[4,6]<-0
drawGraph(g1)

library(Rgraphviz)
dag0<-as(g1, "graphNEL")
plot(dag0)
parents("E", dag0)
children("E", dag0)
dag0.m <-moralize(dag0)
par(mfrow=c(1,2))
plot(dag0) 
plot(dag0.m)

ancestralSet(c("S", "R"), dag0)
plot(ancestralGraph(c("S", "R"), dag0))
plot(moralize(ancestralGraph(c("S", "R"), dag0)))

# comparison with undirected
tab <- table(dati)
tab<-as.data.frame(tab)
tab #acc
modL0 <- glm(Freq~A*S*R*O*E*T, family=poisson, data=tab)
summary(modL0)
step(modL0)#??
modL0 <- glm(Freq~A*S*R*O*E*T, family=poisson, data=tab)
summary(modL0)
modL1 <- glm(Freq~A+S+R+O+E+T, family=poisson, data=tab) # A ind T | rest
summary(modL1)

# Alternalive assuming a graph
library(abn)
dim(dati)
?fitabn
p<-dim(dati)[2]
nodi<- 1:p
mydag<-matrix(0, nrow=p, ncol=p) # all zeros
names(dati)
colnames(mydag) <- rownames( mydag) <-names(dati)
#E ~ S + A, R ~ E, O ~ E, T ~ E + O + R
mydag[1,3]<-1
# continuare ....

#NB: no multinomial
dati$A[dati$A=="old"]<-"adult" # try alternative
dati$T[dati$T=="train"]<-"other"
# set the distributions
mydistr <- list(A = "binomial", R = "binomial", E = "binomial", O = "binomial", S = "binomial", T = "binomial")
mod.BN <- fitabn( data.df=dati, dag.m=mydag, data.dists=mydistr, verbose=TRUE, create.graph=TRUE)
mod.BN
plot(mod.BN$graph)



#################################################
# High dimensional settings (fake data)
library(pcalg)
data(gmG)
?gmG
gmG$x[1:10,1:5]
plot(gmG$g)
pc.mod <- pc(suffStat=list(C=cor(gmG$x), n=nrow(gmG$x)), indepTest=gaussCItest, p=ncol(gmG$x), alpha=0.01)
plot(pc.mod)
par(mfrow=c(1,2))
plot(gmG$g)
plot(pc.mod)
# attention to labels ordering : 
pc.mod <- pc(suffStat=list(C=cor(gmG$x[,c(4:7,8,1:3)]), n=nrow(gmG$x)), indepTest=gaussCItest, p=ncol(gmG$x), alpha=0.01)
par(mfrow=c(1,2))
plot(gmG$g)
plot(pc.mod)
pc.mod <- pc(suffStat=list(C=cor(gmG$x[,c(4:7,8,1:3)]), n=nrow(gmG$x)), indepTest=gaussCItest, p=ncol(gmG$x), alpha=0.01)
par(mfrow=c(1,2))
plot(gmG$g)
plot(pc.mod)