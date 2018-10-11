# Esercizio 0 MASL (rinomina in pratice1 prima di pushare)
# Generate data from model

# Funzione distribuzione cumulative
rip <- function(X1,x,n){
    temp = sort(X1)
    res = 1:n
    for(i in 1:n){
        res[i]=temp[i]/n
    }
    return(res)
}

n <- 100
X1 <- runif(n)
X2 <- 2+3*X1 + rnorm(n)
plot(density(X1),col="red")
plot(density(X2),col="blue")
ripX1 <- rip(X1,100,n)
ripX2 <- rip(X2,100,n)
plot(ripX1,type="l",col="green")
plot(ripX2,type="l",col="magenta")

