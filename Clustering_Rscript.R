#loading the ggfortify package and reading the csv file into the dataframe

library(ggfortify)
df1 <- read.csv('InterviewQ.csv')

#extracting all the protein-expression values as dataframe
p_exp <-data.frame(t(df1[,5:1321]))

#Plotting the clusters using k-means clustering and k = 50 ;
#This plot is saved as Figure1
autoplot(kmeans(p_exp, 50), data = p_exp, label = TRUE, label.size = 3)

#Plotting the clusters using k-means clustering and k = 3 ;
#This plot is saved as Figure2
autoplot(kmeans(p_exp, 3), data = p_exp, label = TRUE, label.size = 3)

#applying principal component analysis and plotting the PC1-PC2 plot 
#This plot is saved as Figure 3
pca_res <- prcomp(p_exp, scale. = TRUE)
autoplot(pca_res, data = p_exp,label = TRUE)





