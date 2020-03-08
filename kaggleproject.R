#Loading necessary packages
library(caret)
library(caTools)
library(class)
library(plyr)
library(dplyr)
library(gplots)
library(mice)
library(missForest)
library(ROCR)
library(rpart)
library(rpart.plot)
library(tibble)
library(tidyverse)
library(corrplot)
library(RcmdrMisc)
library(gbm)
library(glmnet)
library(xgboost)

#Load in our datasets
setwd("C:\Users\Lam\Documents\KaggleProject")
training = as_tibble(read_csv("kaggletrain.csv"))
testing = as_tibble(read_csv("kaggletest.csv"))

#Create column for SalePrice in testing set, fill in with NA and combine training and testing set into one for cleaning
testing$SalePrice = rep(NA, 1459)
feat = rbind(training,testing)

#Removing suggested outliers by looking at GrLivArea and replacing with mean
feat %>% arrange(desc(GrLivArea)) %>% head()
feat$GrLivArea[c(1299,524,1183,692)] = mean(feat$GrLivArea[feat$Id <= 1460])
feat$GrLivArea[2550] = mean(feat$GrLivArea[feat$Id > 1460])

#Check columns for missing values, will deal with numeric features first
#Replace missing values in LotFrontage by imputing median LotFrontage by Neighborhood
colSums(is.na(feat))

#Rename features with number as the first character for convenience
feat = feat %>% dplyr::rename(FirstFlrSF = `1stFlrSF`, SecondFlrSF = `2ndFlrSF`, ThreeSsnPorch = `3SsnPorch`)

#Create a reusable vector for label encoding ordinal features, nice because many ordinals have same rating scheme
quality = c('None' = 0, 'Po' = 1, 'Fa' = 2, 'TA' = 3, 'Gd' = 4, 'Ex' = 5)

#Features related to Masonry Veneer - 23 rows missing MasVnrArea, 24 rows missing MasVnrType: there is one
#house that actually has Masonry Veneer, but needs type imputed using the mode, and 23 houses with no MV. 
#Find number of houses that truly have no Masonry Veneer
length(which(is.na(feat$MasVnrType) & is.na(feat$MasVnrArea)))

#Find the house that has MasVnrArea, but is missing MasVnrType, impute most common type and 'None' for rest
feat[is.na(feat$MasVnrType) & !is.na(feat$MasVnrArea),c('Id', 'MasVnrType', 'MasVnrArea')]
feat$MasVnrType[2611] = "BrkFace"
feat$MasVnrType[is.na(feat$MasVnrType)] = "None"
feat$MasVnrType = as.factor(feat$MasVnrType)

#Impute 0 for houses with no MV
feat = feat %>% mutate(MasVnrArea = replace_na(MasVnrArea, 0))

#We want to use the median LotFrontage for all houses in a given Neighborhood for a more accurate imputation,
#so it is fine to use summary statistics from the entire data set
feat %>% group_by(Neighborhood) %>% summarise(MedLF = median(LotFrontage, na.rm=TRUE))

feat = feat %>% group_by(Neighborhood) %>%
  mutate(LotFrontage = replace(LotFrontage, is.na(LotFrontage), median(LotFrontage, na.rm=TRUE))) %>%
  ungroup()

#Basement Features - 9 different features: 4 continuous, 5 ordinal. There are features with many NA values
#whereas some only have a few. First, narrow down how many houses have no basement by checking if all 79 NA
#values for BsmtFinType1 and features with 79+ NAs come from the same observations
length(which(is.na(feat$BsmtQual) & is.na(feat$BsmtCond) & is.na(feat$BsmtExposure) & 
               is.na(feat$BsmtFinType1) & is.na(feat$BsmtFinType2)))
#There are 9 houses that have a basement but are missing a basement value that should be there, we can
#impute the mode to fix this.
feat[!is.na(feat$BsmtFinType1) & 
       (is.na(feat$BsmtCond)|is.na(feat$BsmtQual)|is.na(feat$BsmtExposure)|is.na(feat$BsmtFinType2)), 
     c('Id', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2')]
feat$BsmtFinType2[333] = 'Unf'
feat$BsmtExposure[c(949,1488,2349)] = 'None'
feat$BsmtCond[c(2041, 2186, 2525)] = 'TA'
feat$BsmtQual[c(2218,2219)] = 'TA'

#With those values imputed, we can now factorize and encode the remaining houses with no basement for these
#5 values.
feat$BsmtQual[is.na(feat$BsmtQual)] = "None"
feat$BsmtQual = as.integer(revalue(feat$BsmtQual, quality))

feat$BsmtCond[is.na(feat$BsmtCond)] = "None"
feat$BsmtCond = as.integer(revalue(feat$BsmtCond, quality))

feat$BsmtExposure[is.na(feat$BsmtExposure)] = "None"
feat$BsmtExposure = as.integer(revalue(feat$BsmtExposure, 
                                              c('None' = 0, 'No' = 1, 'Mn' = 2, 'Av' = 3, 'Gd' = 4)))

feat$BsmtFinType1[is.na(feat$BsmtFinType1)] = "None"
feat$BsmtFinType1 = as.integer(revalue(feat$BsmtFinType1,
                                              c('None' = 0, 'Unf' = 1, 'LwQ' = 2, 'Rec' = 3,
                                                'BLQ' = 4, 'ALQ' = 5, 'GLQ' = 6)))

feat$BsmtFinType2[is.na(feat$BsmtFinType2)] = "None"
feat$BsmtFinType2 = as.integer(revalue(feat$BsmtFinType2,
                                              c('None' = 0, 'Unf' = 1, 'LwQ' = 2, 'Rec' = 3,
                                                'BLQ' = 4, 'ALQ' = 5, 'GLQ' = 6)))

#Now for the remaining basement values with a couple NAs. Find out which house(s) contain the remaining NAs 
#and verify whether or not they have a basement. Can do this by comparing to a basement column that was 
#previously imputed, since we already verified the absence of a basement. (Using BsmtQual)
feat[(is.na(feat$BsmtFullBath)|is.na(feat$BsmtHalfBath)|is.na(feat$TotalBsmtSF)|is.na(feat$BsmtUnfSF)
     |is.na(feat$BsmtFinSF1)|is.na(feat$BsmtFinSF2)), 
     c('Id','BsmtQual','BsmtHalfBath','TotalBsmtSF','BsmtUnfSF','BsmtFinSF1','BsmtFinSF2')]
#Looks like these houses clearly do not have a basement. The remaining basement variables are integer types,
#so impute 0.
feat = feat %>% mutate(BsmtFinSF1 = replace_na(BsmtFinSF1, 0))
feat = feat %>% mutate(BsmtFinSF2 = replace_na(BsmtFinSF2, 0))
feat = feat %>% mutate(BsmtUnfSF = replace_na(BsmtUnfSF, 0))
feat = feat %>% mutate(TotalBsmtSF = replace_na(TotalBsmtSF, 0))
feat = feat %>% mutate(BsmtFullBath = replace_na(BsmtFullBath, 0))
feat = feat %>% mutate(BsmtHalfBath = replace_na(BsmtHalfBath, 0))

#Garage variables: 5 garage variables with 157+ NAs, and two with just 1 NA each. Again, see if the 5 with
#many NAs come from the same observations.
length(which(is.na(feat$GarageType)&is.na(feat$GarageYrBlt)&is.na(feat$GarageFinish)&is.na(feat$GarageQual)
             &is.na(feat$GarageCond)))
#Looks like they are. Now find the observations that are missing 4/5 of the variables.
feat[!is.na(feat$GarageType) & is.na(feat$GarageYrBlt) & is.na(feat$GarageFinish) &
     is.na(feat$GarageQual) & is.na(feat$GarageCond), 
     c('Id','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond')]
#First, we can just impute all GarageYrBlt NAs with YearBuilt, which has no missing values. 
feat$GarageYrBlt[is.na(feat$GarageYrBlt)] = feat$YearBuilt[is.na(feat$GarageYrBlt)]

#Now to fix the two observations with a garage but are missing values.
feat$GarageFinish[c(2127,2577)] = 'Unf'
feat$GarageQual[c(2127,2577)] = 'TA'
feat$GarageCond[c(2127,2577)] = 'TA'

feat$GarageType[is.na(feat$GarageType)] = "None"
feat$GarageType = as.factor(feat$GarageType)

feat$GarageFinish[is.na(feat$GarageFinish)] = "None"
feat$GarageFinish = as.integer(feat$GarageType, c('None' = 0, 'Unf' = 1,
                                                  'RFn' = 2, 'Fin' = 3))

feat$GarageCond[is.na(feat$GarageCond)] = "None"
feat$GarageCond = as.integer(revalue(feat$GarageCond, quality))

feat$GarageQual[is.na(feat$GarageQual)] = "None"
feat$GarageQual = as.integer(revalue(feat$GarageQual, quality))

#Now for the garage variables with only 1 NA: check if they come from the same observation, then if that
#observation has a garage
length(which(is.na(feat$GarageCars) & is.na(feat$GarageArea)))
feat[is.na(feat$GarageCars) & is.na(feat$GarageArea), c('Id','GarageType','GarageCars','GarageArea')]

#Impute median value for GarageArea and GarageCars using only observations from test set since this observation
#comes from that set (Id > 1460)
feat = feat %>% mutate(GarageArea = replace_na(GarageArea, 480))

feat = feat %>% mutate(GarageCars = replace_na(GarageCars, 2))


#The rest of the features with missing values can be imputed independently and encoded/factorized.
feat$Alley[is.na(feat$Alley)] = "None"
feat$Alley = as.factor(feat$Alley)

feat$Electrical[is.na(feat$Electrical)] = "No System"
feat$Electrical = as.integer(revalue(feat$Electrical, c('No System' = 0, 'Mix' = 1, 'FuseP' = 2, 
                                                                      'FuseF' = 3, 'FuseA' = 4, 'SBrkr' = 5)))

feat$Exterior1st[is.na(feat$Exterior1st)] = "VinylSd"
feat$Exterior1st = as.factor(feat$Exterior1st)

feat$Exterior2nd[is.na(feat$Exterior2nd)] = "VinylSd"
feat$Exterior2nd = as.factor(feat$Exterior2nd)

feat$Fence[is.na(feat$Fence)] = "No Fence"
feat$Fence = as.integer(revalue(feat$Fence, c('No Fence' = 0, 'MnWw' = 1,
                                                            'GdWo' = 2, 'MnPrv' = 3, 'GdPrv' = 4)))

feat$FireplaceQu[is.na(feat$FireplaceQu)] = "None"
feat$FireplaceQu = as.integer(revalue(feat$FireplaceQu, quality))

feat$Functional[is.na(feat$Functional)] = "Typ"
feat$Functional = as.integer(revalue(feat$Functional, c('Sal' = 0, 'Sev' = 1, 'Maj2' = 2,
                                                                      'Maj1' = 3, 'Mod' = 4, 'Min2' = 5,
                                                                      'Min1' = 6, 'Typ' = 7)))

feat$KitchenQual[is.na(feat$KitchenQual)] = "TA"
feat$KitchenQual = as.integer(revalue(feat$KitchenQual, quality))

feat$MiscFeature[is.na(feat$MiscFeature)] = "None"
feat$MiscFeature = as.factor(feat$MiscFeature)

feat$MSZoning[is.na(feat$MSZoning)] = "RL"
feat$MSZoning = as.factor(feat$MSZoning)

feat$PoolQC[is.na(feat$PoolQC)] = "None"
feat$PoolQC = as.integer(revalue(feat$PoolQC, quality))

feat$SaleType[is.na(feat$SaleType)] = "WD"
feat$SaleType = as.factor(feat$SaleType)

feat$Utilities[is.na(feat$Utilities)] = "AllPub"
feat$Utilities = as.integer(revalue(feat$Utilities, c('ELO' = 0, 'NoSeWa' = 1,
                                                                    'NoSewr' = 2, 'AllPub' = 3)))
#Now that all the features with NAs have been taken care of, there are some remaining ordinal features that
#need to be encoded.
feat$LotShape = as.integer(revalue(feat$LotShape, c('IR3' = 0, 'IR2' = 1,
                                                                  'IR1' = 2, 'Reg' = 3)))

feat$LandContour = as.integer(revalue(feat$LandContour, c('Low' = 0, 'HLS' = 1,
                                                                        'Bnk' = 2, 'Lvl' = 3)))

feat$LandSlope = as.integer(revalue(feat$LandSlope, c('Sev' = 0, 'Mod' = 1, 'Gtl' = 2)))

feat$ExterCond = as.integer(revalue(feat$ExterCond, quality))

feat$ExterQual = as.integer(revalue(feat$ExterQual, quality))

feat$HeatingQC = as.integer(revalue(feat$HeatingQC, quality))

feat$CentralAir = as.integer(revalue(feat$CentralAir, c('N' = 0, 'Y' = 1)))

feat$PavedDrive = as.integer(revalue(feat$PavedDrive, c('N' = 0, 'P' = 1, 'Y' = 2)))

#The remaining nominal features can now be factorized.
feat = feat %>% mutate_if(sapply(feat, is.character), as.factor)

#Lastly, some numeric features need to be factorized. While features relating to a specific date are obvious
#candidates, there are also some features that have a numeric encoding scheme for non-numeric housing attributes,
#such as MSSubClass. I will have to factorize this and change the numeric labels to character labels that make
#more sense. 

#The features that are dates with no ordinality can simply be factorized. There are other "year" features
#that do matter and should be left untouched, such as YearBuilt, YearRemodAdd, and GarageYrBlt, because
#recency most likely does have an effect on SalePrice. However, a house sold in January (encoded as 1)
#does not necessarily mean it is worth less than one sold in December (encoded as 12). 
feat$MoSold = as.factor(feat$MoSold)
feat$YrSold = as.factor(feat$YrSold)

#MSSubClass is the only feature is in a numeric format but is actually categorical. Each numeric value
#identifies the type of dwelling involved in the sale, and there are 16 different types of dwellings.
#I will use the specific dwelling types in place of the numbering scheme.
feat$MSSubClass = as.factor(feat$MSSubClass)
feat$MSSubClass = revalue(feat$MSSubClass, c('20' = '1-story 1946 newer styles', '30' = '1-story 1945 older',
                                             '40' = '1-story w/attic', '45' = '1-1.5 story unfinished',
                                             '50' = '1-1.5 story finished', '60' = '2-story 1946 newer',
                                             '70' = '2-story 1945 older', '75' = '2-2.5 story all ages',
                                             '80' = 'Split or multi-level', '85' = 'Split foyer',
                                             '90' = 'Duplex all style/age', '120' = '1-story PUD 1946 newer',
                                             '150' = '1-1.5 story PUD all ages', '160' = '2-story PUD 1946 newer',
                                             '180' = 'PUD multilevel incl split lev/foyer',
                                             '190' = '2 family conversion all styles/ages'))

#With the NA values dealt with, we can start examining the correlations between the numeric features and
#SalePrice. Start by selecting only the numeric features and store their correlation values with respect
#to their features.
numeric = select_if(feat, is.numeric)
corvalues = cor(numeric, use = "pairwise.complete.obs")
#Sort by highest correlation to SalePrice and return a list of features with a correlation to SalePrice >0.5.
sortedcv = as.matrix(sort(corvalues[,'SalePrice'], decreasing = TRUE))
highcors = names(which(apply(sortedcv, 1, function(x) abs(x)>0.5)))
#Create a new correlation matrix with only features highly correlated (>=0.5) to SalePrice and visualize in
#a correlation plot.
cmatrix = corvalues[highcors, highcors]
corrplot.mixed(cmatrix, lower = "number", upper = "square", tl.col = 'red', tl.pos = 'lt', 
               tl.cex = 0.8, cl.cex = 0.8, number.cex = 0.8)

#There are a large number of numeric variables that are highly correlated to SalePrice. Let's make some 
#visualizations of these features to get a better sense of their distributions and start thinking about
#which ones we might want to perform feature engineering with.

#Let's first select the highly correlated features and store them in a new data frame. We can create some
#quick and dirty histograms to take a quick look at their distributions. We can also examine some useful
#statistics, especially skewness, to see if it would make sense later to perform some transformations.
numeric_hc = feat[c(highcors)]
ggplot(gather(numeric_hc), aes(value)) + geom_histogram(bins=10) + facet_wrap(~key, scales= 'free_x')
ns = numSummary(numeric_hc, statistics = c("mean","sd","quantiles","skewness"))
ns$table

ggplot(feat, aes(x=YearBuilt, y=SalePrice)) + geom_col()
ggplot(feat, aes(x=TotalBsmtSF, y=SalePrice)) + geom_smooth()
ggplot(feat, aes(x=GrLivArea, y=SalePrice)) + geom_point() + geom_smooth()
ggplot(feat, aes(x=TotRmsAbvGrd, y=SalePrice)) + geom_smooth() + geom_point()
ggplot(feat, aes(x=TotRmsAbvGrd)) + geom_histogram()
ggplot(feat, aes(GarageCars, SalePrice)) + geom_jitter()
feat %>% select(GarageCars, SalePrice) %>% filter(GarageCars == 1) %>% summarize(mean = mean(SalePrice), sd = sd(SalePrice))
feat %>% select(GarageCars, SalePrice) %>% filter(GarageCars == 2) %>% summarize(mean = mean(SalePrice), sd = sd(SalePrice))
feat %>% select(GarageCars, SalePrice) %>% filter(GarageCars == 3) %>% summarize(mean = mean(SalePrice), sd = sd(SalePrice))
feat %>% select(GarageCars, SalePrice) %>% filter(GarageCars == 4) %>% summarize(mean = mean(SalePrice), sd = sd(SalePrice))
ggplot(feat, aes(GarageArea, SalePrice)) + geom_jitter() + geom_smooth()

feat$`1stFlrSF` = log(feat$`1stFlrSF` + 1)
feat$`2ndFlrSF` = log(feat$`2ndFlrSF` + 1)
feat$BsmtFinSF1 = log(feat$BsmtFinSF1 + 1)
feat$OpenPorchSF = log(feat$OpenPorchSF + 1)
feat$WoodDeckSF = log(feat$WoodDeckSF + 1)
feat$LotArea = log(feat$LotArea + 1)
feat$LotFrontage = log(feat$LotFrontage + 1)
feat$EnclosedPorch = log(feat$EnclosedPorch + 1)
feat$ScreenPorch = log(feat$ScreenPorch + 1)

feat$`1stFlrSF` = log(feat$`1stFlrSF` + 1)
feat$`1stFlrSF` = log(feat$`2ndFlrSF` + 1)
feat$BsmtFinSF1 = log(feat$BsmtFinSF1 + 1)
feat$OpenPorchSF = log(feat$OpenPorchSF + 1)
feat$WoodDeckSF = log(feat$WoodDeckSF + 1)
feat$LotArea = log(feat$LotArea + 1)
feat$LotFrontage = log(feat$LotFrontage + 1)
feat$EnclosedPorch = log(feat$EnclosedPorch + 1)
feat$ScreenPorch = log(feat$ScreenPorch + 1)

feat = feat %>% mutate(ExterQualBin = case_when(ExterQual == 'TA' ~ "Typical/Fair", 
                                                ExterQual == 'Fa' ~ "Typical/Fair", 
                                                ExterQual == 'Gd' ~ "Good/Excellent",
                                                ExterQual == 'Ex' ~ "Good/Excellent"))

feat$ExterQualBin = as.factor(feat$ExterQualBin)

feat = feat %>% mutate(BsmtQualBin = case_when(BsmtQual == 'TA' ~ "Typical/Fair",
                                               BsmtQual == "No Basement" ~ "Typical/Fair",
                                               BsmtQual == 'Fa' ~ "Typical/Fair",
                                               BsmtQual == 'Gd' ~ "Good/Excellent",
                                               BsmtQual == 'Ex' ~ "Good/Excellent"))
feat$BsmtQualBin = as.factor(feat$BsmtQualBin)

feat = feat %>% mutate(KitchenQualBin = case_when(KitchenQual == 'TA' ~ "Typical/Fair",
                                                  KitchenQual == 'Fa' ~ "Typical/Fair",
                                                  KitchenQual == 'Gd' ~ "Good/Excellent",
                                                  KitchenQual == 'Ex' ~ "Good/Excellent"))
feat$KitchenQualBin = as.factor(feat$KitchenQualBin)

feat = feat %>% mutate(GarageQualBin = case_when(GarageQual == 'No Garage' ~ "Poor",
                                                 GarageQual == 'Po' ~ "Poor",
                                                 GarageQual == 'TA' ~ "Typical/Fair",
                                                 GarageQual == 'Fa' ~ "Typical/Fair",
                                                 GarageQual == 'Gd' ~ "Good/Excellent",
                                                 GarageQual == 'Ex' ~ "Good/Excellent"))
feat$GarageQualBin = as.factor(feat$GarageQualBin)

feat = feat %>% mutate(ExterQualBin = fct_relevel(ExterQualBin, "Typical/Fair", "Good/Excellent"),
                       BsmtQualBin = fct_relevel(BsmtQualBin, "Typical/Fair", "Good/Excellent"),
                       KitchenQualBin = fct_relevel(KitchenQualBin, "Typical/Fair", "Good/Excellent"),
                       GarageQualBin = fct_relevel(GarageQualBin, "Typical/Fair", "Good/Excellent"))


feat = feat %>% mutate(OverallScore = OverallCond * OverallQual, 
                       TotalSF = TotalBsmtSF + FirstFlrSF + SecondFlrSF + LowQualFinSF, 
                       TotalRooms = TotRmsAbvGrd + FullBath + HalfBath, 
                       TotalPorchSF = WoodDeckSF + OpenPorchSF + EnclosedPorch + ThreeSsnPorch + ScreenPorch)

feat= feat %>% mutate(OverallScore = OverallCond * OverallQual, 
                      TotalSF = TotalBsmtSF + FirstFlrSF + SecondFlrSF + LowQualFinSF, 
                      TotalRooms = TotRmsAbvGrd + FullBath + HalfBath, 
                      TotalPorchSF = WoodDeckSF + OpenPorchSF + EnclosedPorch + ThreeSsnPorch + ScreenPorch)

feat2 = rbind(feat, feat)
featclean = cbind(feat2, feat) %>% as_tibble()

trainclean = featclean %>% filter(Id <= 1460) 
testclean = featclean %>% filter(Id > 1460) 

rf = randomForest(SalePrice ~. -Id, data=trainclean)
cart = rpart(SalePrice ~. -Id, data=trainclean)
gboost = gbm(SalePrice ~. -Id, distribution = "gaussian", data=trainclean, n.trees = 10000, interaction.depth = 4, shrinkage = 0.01)

x = model.matrix(SalePrice~.-Id, trainclean)[,-1]
y = trainclean$SalePrice

cvridge = cv.glmnet(x, y, alpha=0)
ridge = glmnet(x, y, alpha=0, lambda=cvridge$lambda.min)

cvlasso = cv.glmnet(x, y, alpha=1)
lasso = glmnet(x, y, alpha=1, lambda=cvlasso$lambda.min)
lasso_predictions = predict(lasso, s=cvlasso$lambda.min, newx = x)

cvelastic = train(SalePrice ~.-Id, data=trainclean, method = "glmnet",
                  trControl = trainControl("cv", number = 10), tuneLength = 10)
coef(cvelastic$finalModel, cvelastic$bestTune$lambda)

