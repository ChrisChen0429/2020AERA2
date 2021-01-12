library(tm)
library(assertive.files)
library(readr)
library(stopwords)
library(pdftools)
library(topicmodels)
library(doParallel)
library(ggplot2)
library(scales)
library(ldatuning)
library(dplyr)

data <- read.csv('data_reference_doc_DIM.csv')
cleanFun <- function(htmlString) {return(gsub("<.*?>", "", htmlString))}
document <- data$document
for (i in 1:length(document)){
  document[i] <- cleanFun(document[i])
}

for (i in 1:length(document)){
  document[i] <- gsub('\n','',document[i])
}

corpus <- Corpus(VectorSource(document))
processedCorpus <- tm_map(corpus, removeNumbers)
processedCorpus <- tm_map(processedCorpus, removePunctuation)
processedCorpus <- tm_map(processedCorpus, content_transformer(tolower))
stopwords <- stopwords('en','stopwords-iso')
stopwords <- c(stopwords,letters)
stopwords <- unique(stopwords)
processedCorpus <- tm_map(processedCorpus, removeWords, stopwords)
processedCorpus <- tm_map(processedCorpus, stemDocument, language = "en")
processedCorpus <- tm_map(processedCorpus, stripWhitespace)
save(processedCorpus, file = "TCR_DIM.rda")

DTM <- DocumentTermMatrix(processedCorpus)
DTM <- removeSparseTerms(DTM,0.8)
dim(DTM)
saveRDS(DTM,'document_term_matrix_DIM.rds')

document <- c()
for (i in 1:length(processedCorpus)){
  document <- c(document,processedCorpus[[i]]$content)}

data$document <- document
write.csv(data,'final_data.csv')



model.supervisedDIM.string <- '
model{
    for (t in 1:Ttimes){}
    for (k in 1:Ktopics){worddist[k,1:Nwords] ~ ddirch(alphaWords)}
    for (d in 1:Ndocs){
      topicdist[d,1:Ktopics] ~ ddrich(alphaTopics)
      for (w in 1:length[d]){
      wordtopic[d,w] ~ dcat(topicdist[d,1:Ktopics])
      word[d,w] ~ dcat(worddist[wordtopic[d,w],1:Nwords])
}
}
}'

model.LDA.spec <- textConnection(model.LDA.string)

genLDA <- function(mtdm,words,K,alpha.Words=0.1,alpha.topics=0.1){
  word <- do.call(rbind.fill.matrix,lapply(1:ncol(mtdm), function(i) t(rep(1:length(mtdm[,i]),mtdm[,i]))))
  N <- ncol(mtdm)
  Nwords <- length(words)
  alphaTopics <- rep(alpha.topics,K)
  alphaWords <- rep(alpha.Words,Nwords)
  wordtopic <- matrix(NA,nrow(word),ncol(word))
  doclengths <- rowSums(!is.na(word))
  topicdist <- matrix(NA,N,K)
  topicwords <- matrix(NA,K,Nwords)
  datalist <- list(alphaTopics = alphaTopics,
                   alphaWords = alphaWords,
                   topicdist = topicdist,
                   wordtopic = wordtopic,
                   word = word,
                   Ndocs = N,
                   Ktopics = K,
                   length = doclengths,
                   Nwords = Nwords,
                   wordlist = topicwords)
  jags.model(model.LDA.spec,
             data = datalist,
             n.chains = 5,
             n.adapt = 100)
}

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



