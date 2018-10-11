# Esercizio 1 MASL
# Verosimiglianza

#funzione di verosimiglianza
like = function(param,Y,X){
    b0 = param[1]
    b1 = param[2]
    pi = exp(b0+b1*X)/(1+exp(b0+b1*X))
    s = 0
    for(i in 1:length(Y)){
        s = s + (pi[i]^Y[i])*(1-pi[i]^(1-Y[i]))
    }
    return(s)
}

#generate data
n = 100
X1 = rnorm(n)
Y = rbinom(n,1,pi)
l = like(c(2,9),Y,X1)
l
