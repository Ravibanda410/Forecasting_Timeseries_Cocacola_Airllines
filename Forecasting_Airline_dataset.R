library(readr)
#install.packages("xlsx")
#library(xlsx)
library(readxl)

Airlines <- read_excel(file.choose(),1)
View(Airlines)

windows()
plot(Airlines$Passengers,type="o") # type = o means both dot & line. type = l means only line

# As the data is for 12 months of the year, we create 12 dummy variables 

X <- data.frame(outer(rep(month.abb,length = 96), month.abb,"==") + 0 )# Creating dummies for 12 months
View(X)

colnames(X) <- month.abb # Assigning month names 
View(X)
Airlines <- cbind(Airlines,X)
View(Airlines)
Airlines["t"] <- 1:96
View(Airlines)
Airlines["log_Passengers"] <- log(Airlines["Passengers"])
Airlines["t_square"] <- Airlines["t"] * Airlines["t"]
#attach(Airlines)

train <- Airlines[1:84, ]

test <- Airlines[85:96, ]









# We cannot directly tell what is the exact trend. Hence, we will try on different models.  
########################### LINEAR MODEL #############################

linear_model <- lm(Passengers ~ t, data = train)
summary(linear_model)
# We look here only the Residuals & from that we can calculate RMSE values.
linear_pred <- data.frame(predict(linear_model,interval = 'predict', newdata = test))
View(linear_pred)
rmse_linear <- sqrt(mean((test$Passengers - linear_pred$fit)^2, na.rm = T)) # na.rm=T---means if there are any null values in the data then calculate rmse by removing these null values.
rmse_linear
# RMSE is 53.19 and Adjusted R2 Vaue is 0.79


######################### Exponential #################################

expo_model <- lm(log_Passengers ~ t, data = train)
summary(expo_model)

expo_pred <- data.frame(predict(expo_model,interval ='predict', newdata = test))
# As predicted values are logged values,we do exponential of expo_pred$fit to get actual values
rmse_expo <- sqrt(mean((test$Passengers - exp(expo_pred$fit))^2, na.rm = T)) 
rmse_expo 
# RMSE is 46.05  & Adjusted R2 is 0.82
# RMSE has reduced of the exponential model than linear.


######################### Quadratic ####################################

Quad_model <- lm(Passengers ~ t + t_square, data = train)
summary(Quad_model)


Quad_pred <- data.frame(predict(Quad_model,interval ='predict',newdata = test))
rmse_Quad <- sqrt(mean((test$Passengers - Quad_pred$fit)^2,na.rm=T))
rmse_Quad
# RMSE is 48.05 & R2 is 0.79



######################### Additive Seasonality #########################

Add_season_model <- lm(Passengers ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov,data = train)
summary(Add_season_model)

Add_season_pred <- data.frame(predict(Add_season_model,newdata = test,interval ='predict'))
rmse_Add_season <- sqrt(mean((test$Passengers - Add_season_pred$fit)^2, na.rm = T))
rmse_Add_season 
# RMSE is 132.8 and Adjusted R2 is 0.16
# Hence, it may not be additive seasonality model.



######################## Additive Seasonality with Linear #################

Add_sea_Linear_model <- lm(Passengers ~ t + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov, data = train)
summary(Add_sea_Linear_model)

Add_sea_Linear_pred <- data.frame(predict(Add_sea_Linear_model,interval = 'predict', newdata = test))
rmse_Add_sea_Linear <- sqrt(mean((test$Passengers - Add_sea_Linear_pred$fit)^2, na.rm = T))
rmse_Add_sea_Linear 
# RMSE is 35.34 and Adjusted R2 is 0.94



######################## Additive Seasonality with Quadratic #################

Add_sea_Quad_model <- lm(Passengers ~ t + t_square + Jan + Feb + Mar + Apr +  May + Jun + Jul + Aug + Sep + Oct + Nov, data = train)
summary(Add_sea_Quad_model)


Add_sea_Quad_pred <- data.frame(predict(Add_sea_Quad_model,interval = 'predict', newdata = test))
rmse_Add_sea_Quad <- sqrt(mean((test$Passengers - Add_sea_Quad_pred$fit)^2, na.rm = T))
rmse_Add_sea_Quad 
# RMSE is 26.36 and Adjusted R2 is 0.95



######################## Multiplicative Seasonality #########################
# In multiplicative we multiply but we can't multiply directly hence we apply log
multi_sea_model <- lm(log_Passengers ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov, data = train)
summary(multi_sea_model)


multi_sea_pred <- data.frame(predict(multi_sea_model,newdata = test,interval = 'predict'))
rmse_multi_sea <- sqrt(mean((test$Passengers - exp(multi_sea_pred$fit))^2, na.rm = T))
rmse_multi_sea
# RMSE is 140.06 & Adjusted R2 is 0.15



######################## Multiplicative Seasonality Linear trend ##########################

multi_add_sea_model <- lm(log_Passengers ~ t + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov, data = train)
summary(multi_add_sea_model) 


multi_add_sea_pred <- data.frame(predict(multi_add_sea_model, newdata = test,interval = 'predict'))
rmse_multi_add_sea <- sqrt(mean((test$Passengers - exp(multi_add_sea_pred$fit))^2, na.rm = T))
rmse_multi_add_sea 
# RMSE is 10.51 and Adjusted R2 is 0.97
# This is the highest R2 & lowest RMSE




# Preparing table on model and it's RMSE values 

table_rmse <- data.frame(c("rmse_linear","rmse_expo","rmse_Quad","rmse_Add_season","rmse_Add_sea_Quad","rmse_multi_sea","rmse_multi_add_sea"),c(rmse_linear,rmse_expo,rmse_Quad,rmse_Add_season,rmse_Add_sea_Quad,rmse_multi_sea,rmse_multi_add_sea))
View(table_rmse)
colnames(table_rmse) <- c("model","RMSE")
View(table_rmse)
# Here we find that Multiplicative Seasonality with Linear trend has least RMSE value of 10.51



# Now we build the model on the whole dataset of Airlines
new_model <- lm(log_Passengers ~ t + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov,data = Airlines)
summary(new_model)



resid <- residuals(new_model)
resid[1:10]
hist(resid)
windows()
acf(resid,lag.max = 12)
# lag 1 to lag 6 is significant. Hence, we will consider it to build arima.


?arima
# Applying arima
# Auto regression is only used to forecast errors.
k <- arima(resid, order= c(1,0,0)) # perform auto regression with 2nd lag, p=2,d=0,q=0
str(k)

View(data.frame(res=resid, newresid=k$residuals))
windows()
acf(k$residuals,lag.max = 12) # significance problem is removed & all are below threshold ACF values.
pred_res <- predict(arima(k$residuals,order=c(1,0,0)),n.ahead=12)
str(pred_res)
pred_res$pred
acf(k$residuals, lag.max = 12)

write.csv(Airlines, file = "AirlinesData.csv", col.names = F, row.names = F)
getwd()

####################### Predicting new data #############################
#library(readxl)
#test_data < read.csv(file.choose(),1)                           # This file will be attached in the mail.
#View(test_data)



pred_new <- data.frame(predict(new_model,newdata=Airlines,interval = 'predict'))
View(pred_new)
#pred_new$fit <- pred_new$fit + pred_res$pred
#View(pred_new) # improved forecasts


# Our output in the prediction is logged value. Hence, we take exponential of it.
exp(pred_new$fit)


# Hence, we get final forecasted values for next 12 months.