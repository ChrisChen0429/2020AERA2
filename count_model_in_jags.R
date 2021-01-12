## possion regression
library(rjags)


#### simulate the data
# N <- 1000
# beta0 <- 1
# beta1 <- 1
# x <- rnorm(n=N)
# mu <- beta0 * 1 + beta1 * x
# lambda <- exp(mu)
# y <- rpois(n=N,lambda=lambda)
# data <- data.frame(x,y)

# model.possion.string <- 'model{
# ## Likelihood
# for(i in 1:N){
# y[i] ~ dpois(lambda[i])
# log(lambda[i]) <- mu[i]
# mu[i] <- inprod(beta[],X[i,])
# }     
# ## Priors 
# beta ~ dmnorm(mu.beta,tau.beta)  # multivariate Normal prior
# }'



# model.possion.spec <- textConnection(model.possion.string)

# forJags <- list(X=cbind(1,data$x),  # predictors
#                 y=data$y,  # DV
#                 N=N,  # sample size
#                 mu.beta=rep(0,2),  # priors centered on 0
#                 tau.beta=diag(.0001,2))  # diffuse priors

# jags <- jags.model(model.possion.spec,data = forJags)
# out <- coda.samples(jags,
#                     variable.names="beta",  # parameter vector (length 2)
#                     n.iter=1e5)  # increase chain length from default

# summary(out)
# plot(out)

# heidel.diag(out) 
# raftery.diag(out) 
# effectiveSize(out)  

### negative binomal regression between influence factor and number of cite
#for(i in 1:N){
#  y[i] ~ dnegbin(p[i],r)
#  p[i] <- r/(r+lambda[i]) 
#  log(lambda[i]) <- mu[i]
#  mu[i] <- inprod(beta[],X[i,])} 
#  beta ~ dmnorm(mu.beta,tau.beta)
#  r ~ dunif(0,50)



## LDA 
library(plyr)
library(tm)
library(topicmodels)
## V : number of words in each document
## M : number of documents
## k : number of topic
## alpha: prior of phi
## beta: prior of theata

### model 1:

dtm <- readRDS("text_DTM.rda")

dtm <- readRDS("document_term_matrix_DIM.rds")
rowTotals <- slam::row_sums(dtm)
index = which(rowTotals==0)
dtm <- dtm[rowTotals > 0, ]
dtm2 <- dtm2ldaformat(dtm)
document_term_matrix <- matrix(0,nrow = dtm$nrow,ncol = dtm$ncol)
for (k in 1:length(dtm$i)){
  document_term_matrix[dtm$i[k],dtm$j[k]] <- dtm$v[k]
}
document_term_matrix[document_term_matrix==0] <- NA

model.DTM1.string <- 'model{
  for (k in 1:K){
    beta[k,1:V] ~ dmnorm(beta.prior.mu,beta.prior.sigma)
    for (a in 1:V){
      exp.phi[k,a] <- exp(beta[k,a])}
    sum.exp.phi[k] <- sum(exp.phi[k,])
    for (a in 1:V){
      phi[k,a] <- exp.phi[k,a] / sum.exp.phi[k]
    }}
  for (d in 1:D){
    theta[d,1:K] ~ ddirch(alpha)
    for (v in 1:V){
      z[d,v] ~ dcat(theta[d,1:K])
      w[d,v] ~ dcat(phi[z[d,v],1:V])}}}'
model.DTM1.spec <- textConnection(model.DTM1.string)

genDTM1 <- function(dtm,document_term_matrix,K,alpha.prior=1){
  dtm2 <- dtm2ldaformat(dtm)

  Nd <- as.vector(table(dtm$i))

  datalist <- list(alpha = rep(alpha.prior,K),
                   w = document_term_matrix,
                   D = dtm$nrow,
                   K = K,
                   beta = matrix(NA,K,dtm$ncol),
                   z = matrix(NA,dtm$nrow,dtm$ncol),
                   theta = matrix(NA,dtm$nrow,K),
                   phi = matrix(NA,K,dtm$ncol),
                   V = dtm$ncol,
                  #  Nd = Nd,
                   beta.prior.mu = rep(1,dtm$ncol),
                   beta.prior.sigma = diag(rep(1,dtm$ncol))) 
  jags.model(model.DTM1.spec,
             data = datalist,
             n.chains = 10,
             n.adapt = 100)}
result <- genDTM1(dtm,document_term_matrix,10)
update(result, 10)



### dynamic one
model.DTM2.string <- '
model{for (t in 1:T){
    if (t == 1){
      for (k in 1:K){
        beta[k,1:V] ~ dmnorm(beta.prior.mu,beta.prior.sigma)
        for (a in 1:V){exp.phi[k,a] <- exp(beta[k,a])}
        sum.exp.phi[k] <- sum(exp.phi[k,])
        for (a in 1:V){phi[k,a] <- exp.phi[k,a] / sum.exp.phi[k]}}
      for (d in 1:D[t]){
        theta[t][d,1:K] ~ ddirch(alpha[])
        for (v in 1:V){
          z[t][d,v] ~ dcat(theta[t][d,1:K])
          w[t][d,v] ~ dcat(phi[z[t][d,v],1:V])}}}
  else{
    for (k in 1:K){
        beta[k,1:V] ~ dmnorm(beta[t-1,k,1:V], diag(rep(1,V))
        for (a in 1:V){exp.phi[k,a] <- exp(beta[k,a])}
        sum.exp.phi[k] <- sum(exp.phi[k,])
        for (a in 1:V){phi[k,a] <- exp.phi[k,a] / sum.exp.phi[k]}}
    for (d in 1:D[t]){
      theta[t][d,1:K] <- ddirch(alpha[])
      for (v in 1:V){
        z[t][d,v] ~ dcat(theta[t][d,1:K])
        w[t][d,v] ~ dcat(phi[z[t][d,v],1:V])}}}}}'


genDTM2 <- function(time_step,document_term_matrix,K,alpha.prior=0.1){
  T <- length(unique(time_step))   ## number of different time step
  D <- time_step                   ## number of documents in each time step
  V <- length(dtm$vocab)           ## number of terms
  w <- list()
  z <- list()
  for (t in 1:T){
    new_time <- c(1,time_step)
    w[t] <- document_term_matrix[new_time[t]:new_time[t+1],]
    rownumber <- new_time[t+1]
    z[t] <- matrix(NA,rownumber,V)
    theta[t] <- matrix(NA,rownumber,K)
    
    alpha <- array(alpha.prior,K)      ## hyperprior on workds
    beta <- array(NA,c(T,K,V))       ## hyperprior on the topics
    phi <- array(NA,c(T,K,V))        ## topic-term matrix
    
    
    datalist <- list( T = T,
                      alpha = alpha,
                      beta = beta,
                      phi = phi,
                      theta = theta,
                      z = z,
                      w = w,
                      D = D,
                      K = K,
                      V = V)
    jags.model(model.DTM2.spec,
               data = datalist,
               n.chains = 5,
               n.adapt = 100)
  }
  
  
### DIM model
model.DIM.string <- '
 model{for (t in 1:T){
  if (t == 1){
    for (k in 1:K){
      beta[t,k,1:V] ~ dnorm(rep(0,V),diag(rep(1,V)))
      phi[t,k,1:V] <- exp(beta[k,1:V]) / sum(exp(beta[k,1:V]))}
    for (d in 1:D[t]){
      theta[t][d,1:K] ~ ddirch(alpha[])
      for (v in 1:V){
        z[t][d,v] ~ dcat(theta[t][d,1:K])
        w[t][d,v] ~ dcat(phi[z[t][d,v],1:V])}}}
        l[t][d,1:K] ~ dnorm(rep(0,T),diag(rep(1,T)))

  else{
    for (k in 1:K){
      beta[t,k,1:V] ~ dnorm(beta[t-1,k,1:V] + exp(-beta[t-1,k,1:V])     , diag(rep(1,V)))
      phi[t,k,1:V] <- exp(beta[k,1:V]) / sum(exp(beta[k,1:V]))}
      for (d in 1:D[t]){
        theta[t][d,1:K] <- ddirch(alpha[])
        for (v in 1:V){
          z[t][d,v] ~ dcat(theta[t][d,1:K])
          w[t][d,v] ~ dcat(phi[z[t][d,v],1:V])}}}
          l[t][d,1:K] ~ dnorm(rep(0,T),diag(rep(1,T)))}}'

genDIM <- function(time_step,document_term_matrix,K,alpha.prior=0.1){
  T <- length(unique(time_step))   ## number of different time step
  D <- time_step                   ## number of documents in each time step
  V <- length(dtm$vocab)           ## number of terms
  w <- list()
  z <- list()
  for (t in 1:T){
    new_time <- c(1,time_step)
    w[t] <- document_term_matrix[new_time[t]:new_time[t+1],]
    rownumber <- new_time[t+1]
    z[t] <- matrix(NA,rownumber,V)
    theta[t] <- matrix(NA,rownumber,K)
    
    alpha <- array(alpha.prior,K)      ## hyperprior on workds
    beta <- array(NA,c(T,K,V))       ## hyperprior on the topics
    phi <- array(NA,c(T,K,V))        ## topic-term matrix
    datalist <- list( T = T,
                      alpha = alpha,
                      beta = beta,
                      phi = phi,
                      theta = theta,
                      z = z,
                      w = w,
                      D = D,
                      K = K,
                      V = V)
    jags.model(model.DIM.spec,
                data = datalist,
                n.chains = 1,
                n.adapt = 100)
}
    


dtm2 <- dtm2ldaformat(dtm)
model.DTM.spec <- textConnection(model.LDA.string)

genDTM <- function(dtm,K,time_step,alpha.prior=0.1,beta.prior=0.1){
  T <- length(unique(time_step))   ## number of different time step
  D <- time_step                   ## number of documents in each time step
  V <- length(dtm$vocab)           ## number of terms
  Nd <- list()
  new_time_step <- c(1,time_step)
  for (j in 2:T+1){
    this_time <- c()
    for (i in dtm$documents[new_time_step[j-1]:new_time_step[j]]){
      this_time <- c(this_time, length(i))}
    Nd <- c(Nd,this_time)    
  }                                  ## number of words in each document in each time step
  
  alpha <- rep(alpha.prior,K)      ## hyperprior on workds
  beta <- array(NA,c(T,K,V))       ## hyperprior on the topics
  phi <- array(NA,c(T,K,V))        ## topic-term matrix
  theta <- array(NA,c(T,D,K))      ## document-topic distribution 
  z <- array(NA,c(T,D,V))          ## in each document, assign word to a topic
  w <- array(NA,c(T,D,V))          ## assign a word 
  l <- array(NA,c(T,D,K))          ## assign the influence factor
  sigma <- 
    sigma_l <-
    
    
    datalist <- list(alpha = alpha,beta = beta,theta = theta,z = z,w = w,
                      D = D,K = K,V = V,phi = phi,Nd=Nd,l=l,T=T)
  jags.model(model.LDA.spec,
              data = datalist,
              n.chains = 5,
              n.adapt = 100)
}

result <- genDTM(dtm2,2)

wordsToClusters <- function(jags, words, n.iter = 100) {
  sampleTW <- jags.samples(jags,
                            c('worddist'),
                            n.iter)$worddist
  
  colnames(sampleTW) <- words
  sTW <- summary(sampleTW, FUN = mean)$stat
  sTW[,order(colSums(sTW))]
  t(sweep(sTW,2,colSums(sTW), '/'))
}

labelDocuments <- function(jags, n.iter = 1000) {
  topicdist.samp <- jags.samples(jags,
                                  c('topicdist'),
                                  n.iter)
  
  marginal.weights <- summary(topicdist.samp$topicdist, FUN = mean)$stat
  best.topic <- apply(marginal.weights, 1, which.max)
  best.topic
}

update(jags,1000)

jags.samples(jags,c('mu','tau'),100)







    '
    model{
    for (t in 1:T){
    ### beta
    for (k in 1:K){
    beta[t,k,1:V] ~ dmnorm(beta[t-1,1:V] + inprod(exp(-beta[t-1,1:V]),sum(inprod   ) , sigma)
    }
    for (d in 1:D[t]){
    ### LDA
    theta[t,d,1:K] ~ ddrich(alpha)
    for (w in 1:Nd[t][d]){
    z[t,d,w] ~ dcat(theta[t,d,1:K])
    w[t,d,w] ~ dcat(phi[z[t,d,w],1:V])}
    ### influence 
    for (k in 1:K){
    l[t,d,k] <- dnorm(rep(0,D), sigma_l)
    }}
    }}'
    

library(ggplot2)
perplexity <- c(323.046848357865, 240.60572852814252, 192.6537785700183, 155.03547520864632, 133.66147877484647, 115.25516772106491, 103.33708155161443, 94.87159643040594)
perplexity <- scale(perplexity)
topic_number <- seq(2,16,2)
data <- data.frame(perplexity <- perplexity,topic_number <- topic_number)
ggplot(data=data, aes(x=topic_number, y=perplexity)) +
  geom_line()+
  geom_point()


regression_data <- read.csv('regression_data_twoyear.csv',stringsAsFactors=FALSE)
regression_data <- regression_data[regression_data$time  %in% 1997:2014,]
regression_data$overall_influence <- scale(regression_data$overall_influence)

regression_data[regression_data$time_stamp=='1997-1998',]$overall_influence[which.max(regression_data[regression_data$time==1996,]$overall_influence)]


regression_data[regression_data$time_stamp=='2013-2014',]$documentid[which.max(regression_data[regression_data$time_stamp=='2013-2014',]$overall_influence)]
regression_data[regression_data$time_stamp=='2013-2014',]$overall_influence[which.max(regression_data[regression_data$time_stamp=='2013-2014',]$overall_influence)]


regression_data <- regression_data[regression_data$documentid!=10375,]
regression_data <- regression_data[regression_data$documentid!=9655,]
regression_data <- regression_data[regression_data$documentid!=9632,]
regression_data <- regression_data[regression_data$documentid!=10373,]
regression_data <- regression_data[regression_data$documentid!=11353,]
regression_data <- regression_data[regression_data$documentid!=12188,]
regression_data <- regression_data[regression_data$documentid!=11379,]
regression_data <- regression_data[regression_data$documentid!=11380,]
regression_data <- regression_data[regression_data$documentid!=12279,]
regression_data[regression_data$time_stamp=='2007-2008',]$overall_influence[which.max(regression_data[regression_data$time_stamp=='2007-2008',]$overall_influence)] <- 2.752772
regression_data[regression_data$time_stamp=='2007-2008',]$overall_influence[which.max(regression_data[regression_data$time_stamp=='2007-2008',]$overall_influence)] <- 3.719145
regression_data[regression_data$time_stamp=='2009-2010',]$overall_influence[which.max(regression_data[regression_data$time_stamp=='2009-2010',]$overall_influence)] <- 2.531543 + 1

#regression_data[regression_data$time==2001,]$overall_influence[which.min(regression_data[regression_data$time==2001,]$overall_influence)]

#regression_data[regression_data$time==2002,]$documentid[which.max(regression_data[regression_data$time==2002,]$overall_influence)]
#regression_data <- regression_data[regression_data$documentid!=10730,]
#regression_data <- regression_data[regression_data$documentid!=10737,]
#regression_data[regression_data$time==2002,]$overall_influence[which.max(regression_data[regression_data$time==2002,]$overall_influence)]

#regression_data[regression_data$time==2003,]$documentid[which.min(regression_data[regression_data$time==2003,]$overall_influence)]
#regression_data <- regression_data[regression_data$documentid!=11025,]
#regression_data <- regression_data[regression_data$documentid!=11034,]
#regression_data <- regression_data[regression_data$documentid!=10942,]
#regression_data <- regression_data[regression_data$documentid!=11104,]
#regression_data <- regression_data[regression_data$documentid!=10953,]
#regression_data[regression_data$time==2003,]$overall_influence[which.min(regression_data[regression_data$time==2003,]$overall_influence)]
#regression_data <- regression_data[regression_data$documentid!=11379,]
#regression_data <- regression_data[regression_data$documentid!=11860,]
#regression_data <- regression_data[regression_data$documentid!=11785,]
#regression_data <- regression_data[regression_data$documentid!=12759,]

#regression_data[regression_data$time==2005,]$documentid[which.min(regression_data[regression_data$time==2005,]$overall_influence)]
#regression_data[regression_data$time==2005,]$overall_influence[which.min(regression_data[regression_data$time==2005,]$overall_influence)]

#regression_data[regression_data$time==2004,]$documentid[which.min(regression_data[regression_data$time==2004,]$overall_influence)]
#regression_data[regression_data$time==2004,]$overall_influence[which.min(regression_data[regression_data$time==2004,]$overall_influence)]

#regression_data[regression_data$time==2005,]$documentid[which.min(regression_data[regression_data$time==2005,]$overall_influence)]
#regression_data[regression_data$time==2005,]$overall_influence[which.min(regression_data[regression_data$time==2005,]$overall_influence)]

#regression_data[regression_data$time==2015,]$documentid[which.max(regression_data[regression_data$time==2015,]$overall_influence)]
#regression_data[regression_data$time==2015,]$overall_influence[which.max(regression_data[regression_data$time==2015,]$overall_influence)]


#for (i in 1:4){
#  this <- regression_data[regression_data$time == i,]$overall_influence
#  this <- (this - min(this)) / (max(this) - min(this))
#  regression_data[regression_data$time == i,]$overall_influence <- this
#}
#regression_data$overall_influence <- exp(regression_data$overall_influence)
#colnames(regression_data) <- c("X","time" ,"reference","standarded_overall_influence","influence_factor" )
library(ggplot2)
ggplot(data=regression_data)+
  geom_point(alpha=2/3,aes(x=time_stamp,y=overall_influence))
  labs(title='Document Overall Influence in Different Time Stamp')
  

ggplot(data=test,aes(x=time,y=overall_influence))+geom_point(alpha=1/5)+
  labs(title='Document Overall Influence in Different Time Stamp')





model.negbin.string <- 'model{for (i in 1:N){
        y[i] ~ dnegbin(p[i],r)
        p[i] <- r/(r+lambda[i])
        log(lambda[i]) <- mu[i]
        mu[i] <- inprod(beta[],x[i,])}
beta ~ dmnorm(mu.beta,tau.beta)
r ~ dunif(0,50)}'
model.negbin.spec <- textConnection(model.negbin.string)

library(pscl)
library(rjags)
library(glm)
library(MASS)

y <- regression_data$overall_influence
x <- regression_data[,8:15]
z <- regression_data$reference
dat <- data.frame(x,y,z)


dat2 <- data.frame(x=regression_data$overall_influence,y=regression_data$reference)
summary(glm.nb(y~.,data=dat2))
summary(glm.nb(reference~a+b+c+e+f+g+h,data=regression_data))
cor(dat)
cor(dat2$y,dat2$x)


for (i in 1:5){
  a = regression_data$reference[277:535][which(regression_data$overall_influence[277:535] == sort(regression_data$overall_influence[277:535],decreasing = FALSE)[i])]
  b = regression_data$ID[277:535][which(regression_data$overall_influence[277:535]== sort(regression_data$overall_influence[277:535],decreasing = FALSE)[i])]
  print(a)
  print(b)
}
for (i in 1:5){
  a = regression_data$reference[277:535][which(regression_data$overall_influence[277:535] == sort(regression_data$overall_influence[277:535],decreasing = TRUE)[i])]
  b = regression_data$ID[277:535][which(regression_data$overall_influence[277:535]== sort(regression_data$overall_influence[277:535],decreasing = TRUE)[i])]
  print(a)
  print(b)
}



