#this is only to update the R vesion
#install.packages("installr")
#library(installr)
#updateR()


cancer = read.csv(paste0("http://archive.ics.uci.edu/ml/machine-learning-databases/",
                         "breast-cancer-wisconsin/breast-cancer-wisconsin.data"), header = FALSE,
                  stringsAsFactors = F) # Load dataset from the UCI repository.
names(cancer) = c("ID", "thickness", "cell_size", "cell_shape", "adhesion",
                  "epithelial_size", "bare_nuclei", "bland_cromatin", "normal_nucleoli", "mitoses",
                  "class") # Add names to the dataset.
######

cancer = as.data.frame(cancer)
cancer$bare_nuclei = replace(cancer$bare_nuclei, cancer$bare_nuclei == "?", NA) # Recode missing values with NA.
cancer = na.omit(cancer) # Remove rows with missing values.
cancer$class = (cancer$class/2) - 1 # Recode the class (outcome) variable to 1 and 2.
######

head(cancer) # Show the first 6 rows of the dataset
######

set.seed(80817) # Set a random seed so that repeated analyses have the same outcome. Seeds are saved on thr PC only and will not allow analyses to be repeated precicesly on other machines.
index = 1:nrow(cancer) #Create an index vector with as many sequential variables as there are rows in the cancer dataset.
testindex = sample(index, trunc(length(index)/3)) #Take a sample of 33.3% of the variables from the index vector.
testset = cancer[testindex, ] #Create a test (validation) dataset with 33.3$ of the data.
trainset = cancer[-testindex, ] #Create a trainig dataset with 66.6% of the data.
######

x_train = data.matrix(trainset[, 2:10]) # Take the features (x) from the training dataset.
y_train = as.numeric(trainset[, 11]) # Take the outcomes (y) from the training dataset.
x_test = data.matrix(testset[, 2:10]) # Take the features (x) from the testing/validation dataset.
y_test = as.numeric(testset[, 11]) # Take the outcomes (y) from the testing/validation dataset.
# You can use the dim() function to assess the dimension of each matrix
dim(x_test) #this is just an example -- i wanted to see that it was working
######

#install.packages('glmnet',repos=getOption('repos')) #Install latest verison  of `glmnet`. Only necessary once.

require(glmnet) # Load glmnet package into this R session.
glm_model = cv.glmnet(x_train, y_train, alpha=1, nfolds=10) # 10-fold cross validation of the LASSO-regulated linear model.
lambda.min = glm_model$lambda.min # Save the lambda value which minimizes the error of the linear model.
glm_coef = round(coef(glm_model,s= lambda.min),2) #Individual coefficients for variable included in th

plot(glm_model) # Plots mean squared error against log(Lambda).

plot(glmnet(x_train,y_train, family="gaussian", alpha=1),"lambda",label=T, main="") #Plots coefficient values againt log(Lambda)
abline(v=log(lambda.min), lty=3) #Adds a vertical line to the plot of line 34 at the minimum level of l
#####

#install.packages("e1071") # Install latest verison of `e1071`. Only necessary once.
require(e1071) # Load e1071 package into this R session.
## Loading required package: e1071
svm_model = svm(x_train, y_train, cost = 1, gamma = c(1/(ncol(x_train)-1)), kernel="radial", cross=10) # Fit the SVM model to the data with a radial kernel and 10-fold cross validation
## Warning in cret$cresults * scale.factor: Recycling array of length 1 in vector-array arithmetic is deprecated.
## Use c() or as.vector() instead.
#########

#install.packages("nnet") # Install latest verison of `nnet`. Only necessary once.
require(nnet) # Load e1071 package into this R session.
## Loading required package: nnet
nnet_model = nnet(x_train, y_train, size=5) #Fit a single-layer neural network to the data with 5 units in the hidden layer.
glm_pred = round(predict(glm_model, x_test, type="response"),0) # Create a vector of predicitons made from the test/validation data set for the linear model.
svm_pred = round(predict(svm_model, x_test, type="response"),0) #Prediction vector for the SVM.
nnet_pred = round(predict(nnet_model, x_test, type="raw"),0) #Prediction vector for the neural network

predictions = data.frame(glm_pred,svm_pred,nnet_pred) # Collate the three prediction vectors into a data frame.
names(predictions) = c("glm","svm","nnet") #Name the columns of the dataframe.
predictions$sum = rowSums(predictions) # Create a new column in the predictions dataset of the sum of the predictions
algorithms_n = 3 #Insert how many algorithms you have in your predictions data frame. In this case there are 3.
predictions$ensemble_votes = round(predictions$sum/algorithms_n) #Create a new column containing the votes of the ensemble.
3
print(predictions$ensemble_votes[1:30]) # Print the first 30 objects in the vector of predictions from
#######

#install.packages("caret") # Install the `caret` package. Only necessary once.
require(caret) # Load the caret package into this R session.
confusionMatrix(as.factor(glm_pred),as.factor(y_test))# Create a confusion matrix for the LASSO linear model.
confusionMatrix(as.factor(svm_pred),as.factor(y_test)) # Create a confusion matrix for the SVM
confusionMatrix(as.factor(nnet_pred),as.factor(y_test)) # Create a confusion matrix for the neural netw
confusionMatrix(as.factor(predictions$ensemble_votes),as.factor(y_test)) # Create a confusion matrix 
#####

#install.packages("pROC") # Install the `pROC` package. Only necessary once.
require(pROC) # Load the caret package into this R session.
roc_glm = roc(as.vector(y_test),as.vector(glm_pred)) #Conduct the ROC analyses
roc_svm = roc(as.vector(y_test), as.vector(svm_pred))
roc_nnet = roc(as.vector(y_test), as.vector(nnet_pred))

plot.roc(roc_glm, ylim=c(0,1), xlim=c(1,0)) #Plot the ROC curves
lines(roc_glm, col="blue")
lines(roc_nnet, col="green")
lines(roc_svm, col="red")
legend("bottomright", legend=c("General Linear Model", "Support Vector Machine", "Neural
Net"), col=c("blue","red","green"), lwd=2)

auc_glm = auc(roc_glm)#Calculate the area under the ROC curve
auc_svm = auc(roc_svm)#Calculate the area under the ROC curve
auc_nnet = auc(roc_nnet)#Calculate the area under the ROC curve

# The code below sets the values for the features to be evaluated by the trained and validated model.
thickness = 8
cell_size = 7
cell_shape = 8
adhesion = 5
epithelial_size = 5
bare_nuclei = 7
bland_cromatin = 9
normal_nucleoli = 8
mitoses = 10


new_data = c(thickness,cell_size,cell_shape,adhesion,
             epithelial_size,bare_nuclei,bland_cromatin,normal_nucleoli ,mitoses) #Comine the data together in vector.
new_pred_glm = predict(glm_model ,data.matrix(t(new_data))
                       ,type="response") #Apply the new data to the validated model
new_pred_svm = predict(svm_model ,data.matrix(t(new_data))
                       ,type="response")
new_pred_nnet = predict(nnet_model ,data.matrix(t(new_data)),type="raw")

print(new_pred_glm) #Print the prediction for the new data from the glm.
print(new_pred_svm) #Print the prediction for the new data from the svm.
print(new_pred_nnet) #Print the prediction for the new data from the nnet.




