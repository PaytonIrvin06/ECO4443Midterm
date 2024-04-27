# Payton Irvin and Dave Liska
#====================================
# Contents:
#------------------------------------
# setup/renaming vars:      line 13
# summary statistics:       line 133
# histograms:               line 155
# cost function:            line 180
# basic model estimation:   line 193
# GAM estimation:           line 398
# Re-estimation w/ stats:   line 471
#====================================
install.packages('caret')
install.packages('mgcv')
install.packages('gamclass')
install.packages('gam')
library(gam)
library(gamclass)
library(mgcv)
library(dplyr)
library(stargazer)
library(boot)
library(caret)
library(ROCR)


#Payton's Desktop file path:
data1 <- read.csv("C:/Users/Payton Irvin/Documents/UCF/ECO4443/R/Data/hmda_sw-1.csv", header=TRUE)
attach(data1)

#Payton's Surface file path:
#data1 <- read.csv("/hmda_sw-1.csv", header=TRUE)
attach(data1)

approve <- rep(1,2380)
approve[data1$s7 == 3] <- 0

#housingexpense/income
exp_inc <- rep(0,2380)
exp_inc[data1$s45 > 30] <- 1
data1$exp_inc <- exp_inc

#total debt payments/income
debt_inc <- s46
data1$debt_inc<-debt_inc

#Consumer credit
conscred <- s43

#mortgage credit
mortcred <- s42

#poor mortgage credit
poor_mortcred <- rep(0,2380)
poor_mortcred[mortcred>=3] <-1
data1$poor_mortcred <- poor_mortcred
#poor consumer credit
poor_conscred <- rep(0,2380)
poor_conscred[data1$s43 >= 5] <- 1
data1$poor_conscred <-poor_conscred
#public record
pubrec <- s44

#unemployment region
probunemp <- uria

#self-employed
selfemp <- s27a

#Loan/appraised value (low/medium/high?)
loan_app <- s6/s50

loan_app_low <- rep(0,2380)
loan_app_low[loan_app <= .8] <- 1

loan_app_medium <- rep(0,2380)
loan_app_medium[loan_app>.8 & loan_app<.95] <- 1

loan_app_high <- rep(0,2380)
loan_app_high[loan_app>=.95] <- 1

#denied private mortage insurance
denprivmort <- s53

#race
race <- rep(0,2380)

race[data1$s13 == 3] <- 1
race[data1$s13 == 4] <- 1

#education
school

#marital status
married <- rep(0,2380)
married[s23a == 'M'] <-1

#years in applicable job
years_job <- s26a

#years in applicable work
years_work <-s25a

#loan amount
loan_amt <- s6

#income
income <- s17

#purchase price
price <- s33

#liquid assets
assets <- s35

#number of credit lines
num_lines_cr <- s39

#appraised value
value <- s50

pv_ratio <- price/value
lp_ratio <- loan_amt/price

#create full dataset with new variables
data2 <- data.frame(approve, exp_inc, debt_inc, netw, conscred, mortcred, 
                    poor_mortcred, poor_conscred, pubrec, probunemp, selfemp, loan_app,
                    loan_app_low, loan_app_medium, loan_app_high, denprivmort, race, 
                    school, married, years_job, years_work,loan_amt,income, price,
                    assets, num_lines_cr, value, pv_ratio, lp_ratio)
#........................................................................................

#summary statistics
variables <- data.frame(approve, exp_inc, debt_inc, netw, conscred, mortcred, 
                        poor_mortcred, poor_conscred, pubrec, probunemp, selfemp, loan_app, 
                        loan_app_low, loan_app_medium, loan_app_high, denprivmort, 
                        race, school, married, years_job, years_work,loan_amt,
                        income, price, assets, num_lines_cr, value, pv_ratio,
                        lp_ratio)

variables$school <- ifelse(variables$school == 999999.4, NA, variables$school) 
variables$income <- ifelse(variables$income == 999999.4, NA, variables$income)
variables$years_job <- ifelse(variables$years_job == 999999.4, NA, variables$years_job)
variables$years_work <- ifelse(variables$years_work == 999999.4, NA, variables$years_work)
variables$price <- ifelse(variables$price == 999999.4, NA, variables$price)
variables$price <- ifelse(variables$price == 0, NA, variables$price)
variables$assets <- ifelse(variables$assets == 999999.4, NA, variables$assets)
variables$num_lines_cr <- ifelse(variables$num_lines_cr == 999999.4, NA, variables$num_lines_cr)
class_data <- na.omit(variables)

stargazer(class_data, type='text')

#......................................................................................

#histograms
hist(variables$approve, 100, main = 'Approvals', xlab= 'approve', col = 'maroon')
hist(variables$exp_inc, 100, main= 'Housing Expense/Income', xlab = 'exp_inc', col='navy')
hist(variables$debt_inc, 100, main = 'Debt to income', xlab= 'debt_inc', col='plum4', xlim = range(0,80)) #threw out extreme values
hist(variables$poor_mortcred, 100, main='Poor Mortgage Credit', xlab= 'poor_mortcred', col = 'forestgreen')
hist(variables$poor_conscred, 100, main='Poor Consumer Credit', xlab= 'poor_conscred', col = 'tan1')
hist(variables$selfemp, 100, main='Self Employed', xlab= 'selfemp', col = 'maroon')
hist(variables$loan_app_high, 100, main='High Loan/Appraisal Value', xlab= 'loan_app_high', col = 'navy')
hist(variables$denprivmort, 100, main='Denied Private Mortgage Insurance', xlab= 'denprivmort', col = 'plum4')
hist(variables$race, 100, main='Race', xlab= 'race', col = 'forestgreen')
hist(variables$school, 100, main='Years of Schooling', xlab= 'school', col = 'tan1')
hist(variables$married, 100, main='Married', xlab= 'married', col = 'maroon')
hist(variables$years_job, 100, main='Years Employed on Applicable Job', xlab= 'years_job', col = 'plum4')
hist(variables$years_work, 100, main='Years Employed in Applicable Line of Work', xlab= 'years_work', col = 'forestgreen')
hist(variables$loan_amt, 100, main='Loan Amount', xlab= 'loan_amt', col = 'tan1', xlim = range(0,650)) #threw out extreme values
hist(variables$income, 100, main='Applicant Income', xlab= 'income', col = 'maroon', xlim = range(0,350)) #threw out extreme values
hist(variables$price, 100, main='Price of property', xlab= 'price', col = 'navy')
hist(variables$assets, 100, main='Value of liquid assets', xlab= 'assets', col = 'plum4')
hist(variables$num_lines_cr, 100, main='Number of lines of credit', xlab= 'num_lines_cr', col = 'forestgreen')
hist(variables$value, 100, main='Appraised value of property', xlab= 'value', col = 'tan1')
hist(variables$pv_ratio, 100, main='Price-Value Ratio', xlab= 'Price-Value Ratio', col = 'maroon', xlim = range(0, 500)) #threw out extreme values
hist(variables$lp_ratio, 100, main='loan amount-price ratio', xlab= 'loan amount-price ratio', col = 'plum4')

#......................................................................................
#cross validation
#define cost function

costfunc = function(obs, pred_prob){
  weight1 = 1               
  weight0 = 1              
  c1 <- (obs==1)&(pred_prob<optimal_cutoff)  #false negative
  c0 <- (obs==0)&(pred_prob>=optimal_cutoff) #false positive
  cost <- mean(weight1*c1 + weight0*c0)      #weighted average
  return(cost)              
} 

#......................................................................................

#model estimation
#model 2

model2 <- glm(approve ~ poly(debt_inc,2,raw=TRUE) + denprivmort + poor_conscred + married, data=data2, family=binomial)

prob_seq <- seq(0.01, 1, 0.01) 

cv_cost <- rep(0, length(prob_seq)) 

for(i in 1:length(prob_seq)){ 
  optimal_cutoff = prob_seq[i]
  set.seed(123)
  cv_cost[i] = cv.glm(data=data2, glmfit=model2, cost = costfunc, K=10)$delta[1]
}

optimal_cutoff_cv = prob_seq[which(cv_cost==min(cv_cost))]
optimal_cutoff_cv
#optimal cutoff = 0.59
min(cv_cost)
#min cv_cost =0.09789916

#......................................................................................

#model 3

prob_seq <- seq(0.01, 1, 0.01) 
prob_seq_matrix <- matrix(rep(prob_seq, 3), nrow=3, byrow=TRUE)
prob_seq_matrix[2,]<-prob_seq_matrix[2,]+1
prob_seq_matrix[3,]<-prob_seq_matrix[3,]+2


cv_cost <- matrix(0,nrow=3,ncol=length(prob_seq))


for (i in 1:3){
  model3 <- glm(approve ~ poly(debt_inc,i,raw=TRUE)  + poly(school,i,raw=TRUE) + exp_inc + denprivmort + poor_conscred + married, data=data2, family=binomial)
  for (j in 1:length(prob_seq)) {
    optimal_cutoff = prob_seq[j]
    set.seed(123)
    cv_cost[i,j] = cv.glm(data=data2, glmfit=model3, cost = costfunc, K=10)$delta[1]
  }
}


optimal_cutoff_cv = prob_seq_matrix[which(cv_cost==min(cv_cost))]
optimal_cutoff_cv
# .43
#.43 at poly=1
min(cv_cost)
#0.09957983

#......................................................................................

#model 4

prob_seq <- seq(0.01, 1, 0.01) 
prob_seq_matrix <- matrix(rep(prob_seq, 3), nrow=3, byrow=TRUE)
prob_seq_matrix[2,]<-prob_seq_matrix[2,]+1
prob_seq_matrix[3,]<-prob_seq_matrix[3,]+2

cv_cost <- matrix(0,nrow=3,ncol=length(prob_seq))

for (i in 1:3){
  model4 <- glm(approve ~ poly(debt_inc,i,raw=TRUE)  +  poly(income,i,raw=TRUE) + denprivmort + poor_conscred + married, data=data2, family=binomial)
  for (j in 1:length(prob_seq)) {
    optimal_cutoff = prob_seq[j]
    set.seed(123)
    cv_cost[i,j] = cv.glm(data=data2, glmfit=model4, cost = costfunc, K=10)$delta[1]
  }
}

optimal_cutoff_cv = prob_seq_matrix[which(cv_cost==min(cv_cost))]
optimal_cutoff_cv
#.56
min(cv_cost)
#.09831933

#......................................................................................

#model 5

prob_seq <- seq(0.01, 1, 0.01) 
prob_seq_matrix <- matrix(rep(prob_seq, 3), nrow=3, byrow=TRUE)
prob_seq_matrix[2,]<-prob_seq_matrix[2,]+1
prob_seq_matrix[3,]<-prob_seq_matrix[3,]+2

cv_cost <- matrix(0,nrow=3,ncol=length(prob_seq))

for (i in 1:3){
  model5 <- glm(approve ~ poly(debt_inc,i,raw=TRUE) + poly(years_job,i,raw=TRUE) + denprivmort + poor_conscred + married, data=data2, family=binomial)
  for (j in 1:length(prob_seq)) {
    optimal_cutoff = prob_seq[j]
    set.seed(123)
    cv_cost[i,j] = cv.glm(data=data2, glmfit=model5, cost = costfunc, K=10)$delta[1]
  }
}

optimal_cutoff_cv = prob_seq_matrix[which(cv_cost==min(cv_cost))]
optimal_cutoff_cv
#.6

min(cv_cost)
#0.09789916

#......................................................................................

#model6

model6 <- glm(approve ~ poly(debt_inc,2,raw=TRUE) + I(debt_inc*married) + I(poor_conscred*years_work) + denprivmort + poor_conscred + married, data=data2, family=binomial)

prob_seq <- seq(0.01, 1, 0.01) 

cv_cost <- rep(0, length(prob_seq)) 

for(i in 1:length(prob_seq)){ 
  optimal_cutoff = prob_seq[i]
  set.seed(123)
  cv_cost[i] = cv.glm(data=data2, glmfit=model6, cost = costfunc, K=10)$delta[1]
}

optimal_cutoff_cv = prob_seq[which(cv_cost==min(cv_cost))]
optimal_cutoff_cv
#optimal cutoff = 0.6
min(cv_cost)
#min cv_cost =0.09789916

#...................................................................................
#model 7

prob_seq <- seq(0.01, 1, 0.01) 
prob_seq_matrix <- matrix(rep(prob_seq, 3), nrow=3, byrow=TRUE)
prob_seq_matrix[2,]<-prob_seq_matrix[2,]+1
prob_seq_matrix[3,]<-prob_seq_matrix[3,]+2

cv_cost <- matrix(0,nrow=3,ncol=length(prob_seq))


for (i in 1:3){
  model7 <- glm(approve ~  poly(debt_inc,i,raw=TRUE) + poor_conscred + denprivmort + married + I(denprivmort*married), data=data2, family=binomial)
  for (j in 1:length(prob_seq)) {
    optimal_cutoff = prob_seq[j]
    set.seed(123)
    cv_cost[i,j] = cv.glm(data=data2, glmfit=model7, cost = costfunc, K=10)$delta[1]
  }
}

optimal_cutoff_cv = prob_seq_matrix[which(cv_cost==min(cv_cost))]
optimal_cutoff_cv
#2.61
min(cv_cost)
#0.09789916

#...................................................................................................
#model8

prob_seq <- seq(0.01, 1, 0.01) 
prob_seq_matrix <- matrix(rep(prob_seq, 3), nrow=3, byrow=TRUE)
prob_seq_matrix[1,]<-prob_seq_matrix[1,]+1
prob_seq_matrix[2,]<-prob_seq_matrix[2,]+2
prob_seq_matrix[3,]<-prob_seq_matrix[3,]+3

cv_cost <- matrix(0,nrow=3,ncol=length(prob_seq))


for (i in 1:3){
  model8 <- glm(approve ~  poly(debt_inc,i,raw=TRUE) + poor_conscred + denprivmort, data=class_data, family=binomial)
  for (j in 1:length(prob_seq)) {
    optimal_cutoff = prob_seq[j]
    set.seed(123)
    cv_cost[i,j] = cv.glm(data=class_data, glmfit=model8, cost = costfunc, K=10)$delta[1]
  }
}
optimal_cutoff_cv = prob_seq_matrix[which(cv_cost==min(cv_cost))]
optimal_cutoff_cv
#2.62  -> threshold of 0.62, at poly 2
min(cv_cost)
#.09579832

#..................................................................................................
#model9

prob_seq <- seq(0.01, 1, 0.01) 
prob_seq_matrix <- matrix(rep(prob_seq, 3), nrow=3, byrow=TRUE)
prob_seq_matrix[1,]<-prob_seq_matrix[1,]+1
prob_seq_matrix[2,]<-prob_seq_matrix[2,]+2
prob_seq_matrix[3,]<-prob_seq_matrix[3,]+3

cv_cost <- matrix(0,nrow=3,ncol=length(prob_seq))

for (i in 1:3){
  model9 <- glm(approve ~  poly(debt_inc,i,raw=TRUE) + poly(school,i,raw=TRUE)+ denprivmort + poor_conscred, data=class_data, family=binomial)
  for (j in 1:length(prob_seq)) {
    optimal_cutoff = prob_seq[j]
    set.seed(123)
    cv_cost[i,j] = cv.glm(data=class_data, glmfit=model9, cost = costfunc, K=10)$delta[1]
  }
}

optimal_cutoff_cv = prob_seq_matrix[which(cv_cost==min(cv_cost))]
optimal_cutoff_cv
#1.61
min(cv_cost)
#.09705882

#.................................................................................
#          GAM Estimations
#.................................................................................
# Attempting model 8 as a GAM with smoothing splines

logitgam1 <- gam(approve ~ s(debt_inc) + poor_conscred + denprivmort, data=class_data, family=binomial())


prob_seq <- seq(0.01, 1, 0.01) 

cv_cost <- rep(0, length(prob_seq)) 

for(i in 1:length(prob_seq)){ 
  optimal_cutoff = prob_seq[i]
  set.seed(123)
  cv_cost[i] = cv.glm(data=class_data, glmfit=logitgam1, cost = costfunc, K=10)$delta[1]
}

optimal_cutoff_cv = prob_seq[which(cv_cost==min(cv_cost))]
optimal_cutoff_cv
#optimal cutoff = 0.53
min(cv_cost)
#min cv_cost = 0.09789916
#.................................................................................
# GAM w/ best dummies and continuous variables


logitgam <- gam(approve ~ s(debt_inc) + s(income) + s(loan_app) + s(years_job) + denprivmort + poor_conscred, data=variables, family=binomial)

prob_seq <- seq(0.01, 1, 0.01) 

cv_cost <- rep(0, length(prob_seq)) 

for(i in 1:length(prob_seq)){ 
  optimal_cutoff = prob_seq[i]
  set.seed(123)
  cv_cost[i] = cv.glm(data=variables, glmfit=logitgam, cost = costfunc, K=10)$delta[1]
}

optimal_cutoff_cv = prob_seq[which(cv_cost==min(cv_cost))]
optimal_cutoff_cv
#optimal cutoff = 0.41
min(cv_cost)
#min cv_cost = 0.09663866

#.................................................................................
# GAM w best dummies, pulling out selective continuous vars



logitgam <- gam(approve ~ s(debt_inc) + s(loan_app) + denprivmort + poor_conscred, data=variables, family=binomial)

prob_seq <- seq(0.01, 1, 0.01) 

cv_cost <- rep(0, length(prob_seq)) 

for(i in 1:length(prob_seq)){ 
  optimal_cutoff = prob_seq[i]
  set.seed(123)
  cv_cost[i] = cv.glm(data=variables, glmfit=logitgam, cost = costfunc, K=10)$delta[1]
}

optimal_cutoff_cv = prob_seq[which(cv_cost==min(cv_cost))]
optimal_cutoff_cv
#optimal cutoff = 0.48
min(cv_cost)
#min cv_cost = 0.09495798
#plot.Gam(logitgam, se = TRUE, col = 'green') these plots were working and aren't working now


#.................................................................................
#Re-estimate best model with entire dataset 
GAMlogit <- gam(approve ~ s(debt_inc) + s(loan_app) + denprivmort + poor_conscred, data=variables, family=binomial)

summary(GAMlogit)
#plot.Gam(GAMlogit, se = TRUE, col = 'green')

pred_prob1 <- predict.glm(GAMlogit, type=c("response"))
class_prediction1 <- ifelse(pred_prob1 > 0.48, 1,0)
class_prediction1 <- factor(class_prediction1)
approve <- factor(approve)

y <- table(approve)
table(class_prediction1)
confmat <- table(class_prediction1, approve)
confmat

tpr <- confmat[2,2]/y[[2]]
fpr <- confmat[2,1]/y[[1]]
tnr <- confmat[1,1]/y[[1]]
fnr <- confmat[1,2]/y[[2]]
accuracy<- (confmat[2,2]+confmat[1,1])/(y[[2]]+y[[1]])
error_rate <- (confmat[2,1]+confmat[1,2])/(y[[2]]+y[[1]])

tpr
fpr
tnr
fnr
accuracy
error_rate
#0.09369748

confusionMatrix(class_prediction1, approve, positive="1")

#ROC curve
pred1 <- prediction(pred_prob1, approve)
perf1 <- performance(pred1, "tpr", "fpr")
plot(perf1, colorize=TRUE)

#AUC
unlist(slot(performance(pred1, "auc"), "y.values"))
# 0.8044542






