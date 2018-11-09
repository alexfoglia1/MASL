#Ex. 16-10-2018
#
#(1) conosco il grafo "vero"
#    Significa che voglio stimare le correlazioni parziali (pesi degli archi)
#    che sono ricavabili da sigma inverso
#
#(2) voglio ricavare il grafo "vero"
#    significa che prima di stimare le correlazioni parziali ho bisogno di
#    imparare il grafo: ho bisogno di calcolare sigma inverso, e dove ho 
#    valori molto vicini a zero faccio inferenza su quella correlazione parziale
#    che vale zero, e quindi che nel grafo non ci sia il relativo arco.
#    A questo punto posso stimare le correlazioni marginali parziali, dato il grafo
#
library(ggm)
library(mvtnorm)
S <- var(marks)
stima <- fitConGraph(UG(~ mechanics*vectors*algebra+algebra*analysis*statistics), S, n=88)
#genero dati
X <- rmvnorm(3000,mean = rep(0,5),sigma = stima$Shat)
solve(X)
