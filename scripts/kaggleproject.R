#Loading necessary packages

library(caret);library(class);library(dplyr);library(randomForest)
library(rpart);library(corrplot);library(RcmdrMisc);library(gbm)
library(glmnet);library(xgboost);library(tidyr);library(plyr);library(readr)

################## LOADING IN DATA AND PRELIMINARY PREPARATIONS #######################

training = as_tibble(read_csv("train.csv"))
testing = as_tibble(read_csv("test.csv"))

#Create column for SalePrice in testing set, fill in with NA and combine training and testing set into one for cleaning
testing$SalePrice = rep(NA, 1459)
feat = rbind(training,testing)

#Removing suggested outliers by looking at GrLivArea and replacing with mean
feat %>% arrange(desc(GrLivArea)) %>% head()
feat$GrLivArea[c(1299,524,1183,692)] = mean(feat$GrLivArea[feat$Id <= 1460])
feat$GrLivArea[2550] = mean(feat$GrLivArea[feat$Id > 1460])


################# DATA CLEANING: NA VALUES, ENCODING, FACTORIZING ##########################


#Check columns for missing values, will deal with numeric features first
#Replace missing values in LotFrontage by imputing median LotFrontage by Neighborhood
colSums(is.na(feat))

#Rename features with number as the first character for convenience
feat = feat %>% dplyr::rename(FirstFlrSF = `1stFlrSF`, SecondFlrSF = `2ndFlrSF`, ThreeSsnPorch = `3SsnPorch`)

#Create a reusable vector for label encoding ordinal features, nice because many ordinals have same rating scheme
quality = c("None" = 0, "Po" = 1, "Fa" = 2, "TA" = 3, "Gd" = 4, "Ex" = 5)

#Features related to Masonry Veneer - 23 rows missing MasVnrArea, 24 rows missing MasVnrType: there is one
#house that actually has Masonry Veneer, but needs type imputed using the mode, and 23 houses with no MV. 
#Find number of houses that truly have no Masonry Veneer
length(which(is.na(feat$MasVnrType) & is.na(feat$MasVnrArea)))

#Find the house that has MasVnrArea, but is missing MasVnrType, impute most common type and "None" for rest
feat[is.na(feat$MasVnrType) & !is.na(feat$MasVnrArea),c("Id", "MasVnrType", "MasVnrArea")]
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
     c("Id", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2")]
feat$BsmtFinType2[333] = "Unf"
feat$BsmtExposure[c(949,1488,2349)] = "None"
feat$BsmtCond[c(2041, 2186, 2525)] = "TA"
feat$BsmtQual[c(2218,2219)] = "TA"

#With those values imputed, we can now factorize and encode the remaining houses with no basement for these
#5 values.
feat$BsmtQual[is.na(feat$BsmtQual)] = "None"
feat$BsmtQual = as.integer(revalue(feat$BsmtQual, quality))

feat$BsmtCond[is.na(feat$BsmtCond)] = "None"
feat$BsmtCond = as.integer(revalue(feat$BsmtCond, quality))

feat$BsmtExposure[is.na(feat$BsmtExposure)] = "None"
feat$BsmtExposure = as.integer(revalue(feat$BsmtExposure, 
                                              c("None" = 0, "No" = 1, "Mn" = 2, "Av" = 3, "Gd" = 4)))

feat$BsmtFinType1[is.na(feat$BsmtFinType1)] = "None"
feat$BsmtFinType1 = as.integer(revalue(feat$BsmtFinType1,
                                              c("None" = 0, "Unf" = 1, "LwQ" = 2, "Rec" = 3,
                                                "BLQ" = 4, "ALQ" = 5, "GLQ" = 6)))

feat$BsmtFinType2[is.na(feat$BsmtFinType2)] = "None"
feat$BsmtFinType2 = as.integer(revalue(feat$BsmtFinType2,
                                              c("None" = 0, "Unf" = 1, "LwQ" = 2, "Rec" = 3,
                                                "BLQ" = 4, "ALQ" = 5, "GLQ" = 6)))

#Now for the remaining basement values with a couple NAs. Find out which house(s) contain the remaining NAs 
#and verify whether or not they have a basement. Can do this by comparing to a basement column that was 
#previously imputed, since we already verified the absence of a basement. (Using BsmtQual)
feat[(is.na(feat$BsmtFullBath)|is.na(feat$BsmtHalfBath)|is.na(feat$TotalBsmtSF)|is.na(feat$BsmtUnfSF)
     |is.na(feat$BsmtFinSF1)|is.na(feat$BsmtFinSF2)), 
     c("Id","BsmtQual","BsmtHalfBath","TotalBsmtSF","BsmtUnfSF","BsmtFinSF1","BsmtFinSF2")]
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
     c("Id","GarageType","GarageYrBlt","GarageFinish","GarageQual","GarageCond")]
#First, we can just impute all GarageYrBlt NAs with YearBuilt, which has no missing values. 
feat$GarageYrBlt[is.na(feat$GarageYrBlt)] = feat$YearBuilt[is.na(feat$GarageYrBlt)]

#Now to fix the two observations with a garage but are missing values.
feat$GarageFinish[c(2127,2577)] = "Unf"
feat$GarageQual[c(2127,2577)] = "TA"
feat$GarageCond[c(2127,2577)] = "TA"

feat$GarageType[is.na(feat$GarageType)] = "None"
feat$GarageType = as.factor(feat$GarageType)

feat$GarageFinish[is.na(feat$GarageFinish)] = "None"
feat$GarageFinish = as.integer(feat$GarageType, c("None" = 0, "Unf" = 1,
                                                  "RFn" = 2, "Fin" = 3))

feat$GarageCond[is.na(feat$GarageCond)] = "None"
feat$GarageCond = as.integer(revalue(feat$GarageCond, quality))

feat$GarageQual[is.na(feat$GarageQual)] = "None"
feat$GarageQual = as.integer(revalue(feat$GarageQual, quality))

#Now for the garage variables with only 1 NA: check if they come from the same observation, then if that
#observation has a garage
length(which(is.na(feat$GarageCars) & is.na(feat$GarageArea)))
feat[is.na(feat$GarageCars) & is.na(feat$GarageArea), c("Id","GarageType","GarageCars","GarageArea")]

#Impute median value for GarageArea and GarageCars using only observations from test set since this observation
#comes from that set (Id > 1460)
feat = feat %>% mutate(GarageArea = replace_na(GarageArea, 480))

feat = feat %>% mutate(GarageCars = replace_na(GarageCars, 2))


#The rest of the features with missing values can be imputed independently and encoded/factorized.
feat$Alley[is.na(feat$Alley)] = "None"
feat$Alley = as.factor(feat$Alley)

feat$Electrical[is.na(feat$Electrical)] = "No System"
feat$Electrical = as.integer(revalue(feat$Electrical, c("No System" = 0, "Mix" = 1, "FuseP" = 2, 
                                                                      "FuseF" = 3, "FuseA" = 4, "SBrkr" = 5)))

feat$Exterior1st[is.na(feat$Exterior1st)] = "VinylSd"
feat$Exterior1st = as.factor(feat$Exterior1st)

feat$Exterior2nd[is.na(feat$Exterior2nd)] = "VinylSd"
feat$Exterior2nd = as.factor(feat$Exterior2nd)

feat$Fence[is.na(feat$Fence)] = "No Fence"
feat$Fence = as.integer(revalue(feat$Fence, c("No Fence" = 0, "MnWw" = 1,
                                                            "GdWo" = 2, "MnPrv" = 3, "GdPrv" = 4)))

feat$FireplaceQu[is.na(feat$FireplaceQu)] = "None"
feat$FireplaceQu = as.integer(revalue(feat$FireplaceQu, quality))

feat$Functional[is.na(feat$Functional)] = "Typ"
feat$Functional = as.integer(revalue(feat$Functional, c("Sal" = 0, "Sev" = 1, "Maj2" = 2,
                                                                      "Maj1" = 3, "Mod" = 4, "Min2" = 5,
                                                                      "Min1" = 6, "Typ" = 7)))

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
feat$Utilities = as.integer(revalue(feat$Utilities, c("ELO" = 0, "NoSeWa" = 1,
                                                                    "NoSewr" = 2, "AllPub" = 3)))
#Now that all the features with NAs have been taken care of, there are some remaining ordinal features that
#need to be encoded.
feat$LotShape = as.integer(revalue(feat$LotShape, c("IR3" = 0, "IR2" = 1,
                                                                  "IR1" = 2, "Reg" = 3)))

feat$LandContour = as.integer(revalue(feat$LandContour, c("Low" = 0, "HLS" = 1,
                                                                        "Bnk" = 2, "Lvl" = 3)))

feat$LandSlope = as.integer(revalue(feat$LandSlope, c("Sev" = 0, "Mod" = 1, "Gtl" = 2)))

feat$ExterCond = as.integer(revalue(feat$ExterCond, quality))

feat$ExterQual = as.integer(revalue(feat$ExterQual, quality))

feat$HeatingQC = as.integer(revalue(feat$HeatingQC, quality))

feat$CentralAir = as.integer(revalue(feat$CentralAir, c("N" = 0, "Y" = 1)))

feat$PavedDrive = as.integer(revalue(feat$PavedDrive, c("N" = 0, "P" = 1, "Y" = 2)))

#The remaining nominal features can now be factorized.
feat = feat %>% mutate_if(sapply(feat, is.character), as.factor)

#Lastly, some numeric features need to be factorized. While features relating to a specific date are obvious
#candidates, there are also some features that have a numeric encoding scheme for non-numeric housing attributes,
#such as MSSubClass. I will have to factorize this and change the numeric labels to character labels that make
#more sense. 

#The features that are dates with no ordinality can simply be factorized. There are other "year" features
#that do matter and should be left untouched, such as YearBuilt, YearRemodAdd, and GarageYrBlt, because
#recency most likely does have an effect on SalePrice. However, a house sold in January (encoded as 1)
#does not necessarily mean it is worth less than one sold in December (encoded as 12). Need to factorize
#YrSold as well, but need it to create new features first before turning it into a factor.
feat$MoSold = as.factor(feat$MoSold)


#MSSubClass is the only feature is in a numeric format but is actually categorical. Each numeric value
#identifies the type of dwelling involved in the sale, and there are 16 different types of dwellings.
#I will use the specific dwelling types in place of the numbering scheme.
feat$MSSubClass = as.factor(feat$MSSubClass)
feat$MSSubClass = revalue(feat$MSSubClass, c("20" = "1-story 1946 newer styles", "30" = "1-story 1945 older",
                                             "40" = "1-story w/attic", "45" = "1-1.5 story unfinished",
                                             "50" = "1-1.5 story finished", "60" = "2-story 1946 newer",
                                             "70" = "2-story 1945 older", "75" = "2-2.5 story all ages",
                                             "80" = "Split or multi-level", "85" = "Split foyer",
                                             "90" = "Duplex all style/age", "120" = "1-story PUD 1946 newer",
                                             "150" = "1-1.5 story PUD all ages", "160" = "2-story PUD 1946 newer",
                                             "180" = "PUD multilevel incl split lev/foyer",
                                             "190" = "2 family conversion all styles/ages"))


##################### DATA EXPLORATION AND VISUALIZATION ##############################


#With the NA values dealt with, we can start examining the correlations between the numeric features and
#SalePrice. Start by selecting only the numeric features and store their correlation values with respect
#to their features.
numeric = select_if(feat, is.numeric)
corvalues = cor(numeric, use = "pairwise.complete.obs")
#Sort by highest correlation to SalePrice and return a list of features with a correlation to SalePrice >0.5.
sortedcv = as.matrix(sort(corvalues[,"SalePrice"], decreasing = TRUE))
highcors = names(which(apply(sortedcv, 1, function(x) abs(x)>0.5)))
#Create a new correlation matrix with only features highly correlated (>=0.5) to SalePrice and visualize in
#a correlation plot.
cmatrix = corvalues[highcors, highcors]
corrplot.mixed(cmatrix, lower = "number", upper = "square", tl.col = "red", tl.pos = "lt", 
               tl.cex = 0.8, cl.cex = 0.8, number.cex = 0.8)

#There are a large number of numeric variables that are highly correlated to SalePrice. Let"s make some 
#visualizations of these features to get a better sense of their distributions and start thinking about
#which ones we might want to perform feature engineering with.

#Let"s first select the highly correlated features and store them in a new data frame. We can create some
#quick and dirty histograms to take a quick look at their distributions. We can also examine some useful
#statistics, especially skewness, to see if it would make sense later to perform some transformations.
numeric_hc = feat[c(highcors)]
ggplot(gather(numeric_hc), aes(value)) + geom_histogram(bins=10) + facet_wrap(~key, scales= "free_x")
ns = numSummary(numeric_hc, statistics = c("mean","sd","quantiles","skewness"))
ns$table

#Plot some of the more heavily skewed features
ggplot(feat[!is.na(feat$SalePrice),], aes(SalePrice)) + geom_histogram(bins = 20)
ggplot(feat, aes(GrLivArea)) + geom_histogram(bins = 20)
ggplot(feat, aes(TotalBsmtSF)) + geom_histogram(bins = 20)
ggplot(feat, aes(FirstFlrSF)) + geom_histogram(bins = 20)

#See if log transforming normalizes the distribution
ggplot(feat[!is.na(feat$SalePrice),], aes(SalePrice)) + geom_histogram(bins = 20) + scale_x_log10() + xlab("SalePrice Log Scaled")
ggplot(feat, aes(GrLivArea)) + scale_x_log10() + geom_density() + xlab("GrLivArea Log Scaled")
ggplot(feat, aes(TotalBsmtSF)) + geom_histogram(bins = 20) + scale_x_log10() + xlab("TotalBsmtSF Log Scaled")
ggplot(feat, aes(FirstFlrSF)) + geom_histogram(bins = 20) + scale_x_log10() + xlab("FirstFlrSF Log Scaled")

#SalePrice vs Quality Features
ggplot(feat[!is.na(feat$SalePrice),], aes(x=OverallQual, y=SalePrice)) + geom_col()
ggplot(feat[!is.na(feat$SalePrice),], aes(x=ExterQual, y=SalePrice)) + geom_col()
ggplot(feat[!is.na(feat$SalePrice),], aes(x=KitchenQual, y=SalePrice)) + geom_col()
ggplot(feat[!is.na(feat$SalePrice),], aes(x=BsmtQual, y=SalePrice)) + geom_col()
ggplot(feat[!is.na(feat$SalePrice),], aes(x=FireplaceQu, y=SalePrice)) + geom_col()

#SalePrice vs SF features
ggplot(feat[!is.na(feat$SalePrice),], aes(x=TotalBsmtSF, y=SalePrice)) + geom_point() + geom_smooth()
ggplot(feat[!is.na(feat$SalePrice),], aes(x=FirstFlrSF, y=SalePrice)) + geom_point() + geom_smooth()
ggplot(feat[!is.na(feat$SalePrice),], aes(x=GrLivArea, y=SalePrice)) + geom_point() + geom_smooth()

#SalePrice vs Room features
ggplot(feat[!is.na(feat$SalePrice),], aes(x=TotRmsAbvGrd, y=SalePrice)) + geom_col()
ggplot(feat[!is.na(feat$SalePrice),], aes(x=FullBath, y=SalePrice)) + geom_col()

#SalePrice vs Garage features
ggplot(feat[!is.na(feat$SalePrice),], aes(x=GarageYrBlt, y=SalePrice)) + geom_col()
ggplot(feat[!is.na(feat$SalePrice),], aes(x=GarageArea, y=SalePrice)) + geom_point() + geom_smooth()
ggplot(feat[!is.na(feat$SalePrice),], aes(x=GarageCars, y=SalePrice)) + geom_col()

#SalePrice vs Year features
ggplot(feat[!is.na(feat$SalePrice),], aes(x=YearBuilt, y=SalePrice)) + geom_col()
ggplot(feat[!is.na(feat$SalePrice),], aes(x=YearRemodAdd, y=SalePrice)) + geom_col()

#SalePrice vs Categorical features: mean/median of SalePrice for each factor in Neighborhood/MSSubClass,
#dashed lines at the median for median plots, and mean +/- one standard deviation for mean plots.
#Will use the mean plots to bin factors later on. 
ggplot(feat[!is.na(feat$SalePrice),], aes(x=reorder(Neighborhood, SalePrice, FUN=mean), y=SalePrice)) + 
  geom_bar(stat="summary", fun = "mean", fill="red") + xlab("Neighborhood") + ylab("Mean SalePrice") +
  scale_y_continuous(breaks = seq(0, 350000, by=50000)) + 
  geom_hline(yintercept = c(101478.7,180921.2,260363.7), linetype="dashed", color="blue") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(feat[!is.na(feat$SalePrice),], aes(x=reorder(Neighborhood, SalePrice, FUN=median), y=SalePrice)) + 
  geom_bar(stat="summary", fun = "median", fill="red") + xlab("Neighborhood") + ylab("Med SalePrice") +
  scale_y_continuous(breaks = seq(0, 350000, by=50000)) + 
  geom_hline(yintercept = 163000, linetype="dashed", color="blue") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(feat[!is.na(feat$SalePrice),], aes(x=reorder(MSSubClass, SalePrice, FUN=mean), y=SalePrice)) +
  geom_bar(stat="summary", fun = "mean", fill="red") + xlab("MSSubClass") + ylab("Mean SalePrice") +
  scale_y_continuous(breaks = seq(0, 300000, by=50000)) +
  geom_hline(yintercept = c(101478.7,180921.2,260363.7), linetype="dashed", color="blue") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(feat[!is.na(feat$SalePrice),], aes(x=reorder(MSSubClass, SalePrice, FUN=median), y=SalePrice)) +
  geom_bar(stat="summary", fun = "median", fill="red") + xlab("MSSubClass") + ylab("Med SalePrice") +
  scale_y_continuous(breaks = seq(0, 300000, by=50000)) +
  geom_hline(yintercept = 163000, linetype="dashed", color="blue") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


##################### FEATURE ENGINEERING ######################


#Creating new features from combining existing ones

feat = feat %>% mutate(OverallScore = OverallCond * OverallQual,
                       ExterScore = ExterQual * ExterCond,
                       BsmtScore = BsmtQual * BsmtCond,
                       TotalSF = TotalBsmtSF + FirstFlrSF + SecondFlrSF + LowQualFinSF,
                       TotalArea = GrLivArea + TotalBsmtSF,
                       TotalRooms = TotRmsAbvGrd + FullBath + +BsmtFullBath + HalfBath*0.5 + BsmtHalfBath*0.5, 
                       TotalPorchSF = WoodDeckSF + OpenPorchSF + EnclosedPorch + ThreeSsnPorch + ScreenPorch,
                       Age = YrSold - YearBuilt,
                       YrsSinceRemodel = YrSold - YearRemodAdd)

#Now we can factorize YrSold after using it
feat$YrSold = as.factor(feat$YrSold)

#Binning categorical features: Quick gradient boost model to check variable importance of categoricals
gboost = gbm(SalePrice ~. -Id, distribution = "gaussian", data=feat[!is.na(feat$SalePrice),], 
             n.trees = 1000, interaction.depth = 4, shrinkage = 0.01)
summary(gboost)

#Most relevant: Neighborhood, MSSubClass.

#Neighborhood looks like it can be binned using 4 cuts. There are 3 neighborhoods that are significantly
#"richer" than other neighborhoods according to both the median and mean plots: NridgHt, NoRidge, and StoneBr.
#Conversely, there are 3 neighborhoods significantly below the median/mean: MeadowV, IDOTRR, and BRDale. The
#former exceeding 1 standard deviation above the mean and the latter exceeding 1 standard deviation below
#the mean, or extremely close to exceeding it. The remaining neighborhoods below the mean, but not less than
#1 full standard deviation below can be binned together, and the same for those above the mean, but not more
#than a full standard deviation. 

#Store a copy of neighborhood into our new feature, then encode since it is ordinal. 
feat$NeighborhoodType = feat$Neighborhood
feat$NeighborhoodType = as.integer(revalue(feat$NeighborhoodType, c("MeadowV" = 0, "IDOTRR" = 0, "BrDale" = 0,
                                                         "BrkSide" = 1, "Edwards"= 1, "OldTown" = 1, "Sawyer" = 1,
                                                         "Blueste" = 1, "SWISU" = 1, "NPkVill" = 1, "NAmes" = 1,
                                                         "Mitchel" = 1,
                                                         "SawyerW" = 2, "NWAmes" = 2, "Gilbert" = 2, "Blmngtn" = 2,
                                                         "CollgCr" = 2, "Crawfor" = 2, "ClearCr" = 2,
                                                         "Somerst" = 2, "Veenker" = 2, "Timber" = 2,
                                                         "StoneBr"= 3, "NridgHt"= 3, "NoRidge"= 3)))

#While mean/median SalePrice does differ quite significantly between the different levels of MSSubclass,
#the differences aren"t as drastic as in Neighborhood. So, I have decided against binning it for now.


################### MODELING & PREDICTING #######################


#Predictors need to be preprocessed before most R algorithms can use them. 

#Dealing with multicollinearity: dropping highly correlated variables - will look at the correlation plot
#to determine variables with high correlation between themselves and SalePrice, will drop the one with a
#lower correlation to SalePrice.

varsToDrop = c('GarageArea','TotalBsmtSF','TotRmsAbvGrd','GarageYrBlt','KitchenQual','YearRemodAdd')
feat = feat[,!names(feat) %in% varsToDrop]

#Preprocessing numeric features using caret preProcess, removing encoded categorical features
num = select_if(feat, is.numeric)
num = within(num, rm("LotShape","LandContour","Utilities","LandSlope","ExterQual","ExterCond",
"BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","HeatingQC","CentralAir",
"Electrical","Functional","FireplaceQu","GarageFinish","GarageQual","GarageCond",
"PavedDrive","PoolQC","Fence","ExterScore","BsmtScore","NeighborhoodType","SalePrice"))

preproc = preProcess(num, method = c("center","scale"))
normalized = predict(preproc, num)

#Categorical predictors: one hot encode

#Select true categoricals from our data set and keep ordinals converted to integers in a separate df
categorical = select_if(feat, is.factor)
int = select_if(feat, is.integer)

#One hot encode using model.matrix: -1 to remove intercept column
dummify = as.data.frame(model.matrix(~.-1, categorical))

#Removing levels with sparse/no observations 
noneTrain = which(colSums(dummify[1:nrow(feat[!is.na(feat$SalePrice),]),])==0)
colnames(dummify[noneTrain])
dummify = dummify[,-noneTrain]

noneTest = which(colSums(dummify[(nrow(feat[!is.na(feat$SalePrice),])+1):nrow(feat),])==0)
colnames(dummify[noneTest])
dummify = dummify[,-noneTest]

fewOnes <- which(colSums(dummify[1:nrow(feat[!is.na(feat$SalePrice),]),])<10)
colnames(dummify[fewOnes])
dummify = dummify[,-fewOnes]

#Removing skewness of SalePrice
feat$SalePrice = log(feat$SalePrice)

#Combine dataframes for modelling, remove Id column
featclean = cbind(normalized, dummify, int)
featclean = featclean[,-1]

#Create training and testing sets
trainclean = featclean[!is.na(feat$SalePrice),]
testclean = featclean[is.na(feat$SalePrice),]

#Gradient boosting algorithm, requires target variable in same training set data frame as predictors
#Also ensuring variable names are legal, some models throw an error otherwise
set.seed(1106)
featclean1 = rbind(trainclean, testclean)
featclean1$SalePrice = feat$SalePrice
trainclean1 = featclean1[!is.na(featclean1$SalePrice),]
testclean1 = featclean1[is.na(featclean1$SalePrice),]
names(trainclean1) = make.names(names(trainclean1))
names(testclean1) = make.names(names(testclean1))

gboost = gbm(SalePrice~., distribution = "gaussian", data=trainclean1, 
             n.trees = 10000, interaction.depth = 4, shrinkage = 0.01)
gbpred = predict(gboost, newdata=testclean1, n.trees = 10000)
gboostPredPrices = exp(gbpred)
head(gboostPredPrices)

#Random Forest algorithm
rf = randomForest(SalePrice ~., ntree = 10000, importance = TRUE, data=trainclean1)
rfpred = predict(rf, newdata=testclean1)
rfImp = importance(rf)
rfPredPrices = exp(rfpred)
head(rfPredPrices)

#Lasso Regression
mc = trainControl(method="cv", number=5, summaryFunction = defaultSummary)
lassoGrid = expand.grid(alpha = 1, lambda = seq(0.001,0.1,by = 0.0005))
lasso_mod = train(x=trainclean, y=feat$SalePrice[!is.na(feat$SalePrice)], method='glmnet', 
                  trControl = mc, tuneGrid=lassoGrid)
lasso_mod
ggplot(lasso_mod)
lasso_mod$bestTune
lassoVI = varImp(lasso_mod,scale=F)
lassoImp = lassoVI$importance
varsSelected = length(which(lassoImp$Overall!=0))
varsNotSelected = length(which(lassoImp$Overall==0))
cat('Lasso uses', varsSelected, 'variables in its model, and did not use', varsNotSelected, 'variables.')

lassoPred = predict(lasso_mod, testclean)
lassoPredPrices = exp(lassoPred)
head(lassoPredPrices)

#Ridge Regression
ridgeGrid = expand.grid(alpha = 0, lambda = seq(0.001,0.1,by = 0.0005))

ridge_mod = train(x=trainclean, y=feat$SalePrice[!is.na(feat$SalePrice)], method='glmnet',
                  trControl = mc, tuneGrid = ridgeGrid)
ridge_mod$bestTune
ridgeVI = varImp(ridge_mod, scale=F)
ridgeImp = ridgeVI$importance
coef(ridge_mod$finalModel, ridge_mod$bestTune$lambda)

ridgePred = predict(ridge_mod, testclean)
ridgePredPrices = exp(ridgePred)
head(ridgePredPrices)

#Elastic net regression
elasticGrid = expand.grid(alpha = seq(0,1,by = 0.1), lambda = seq(0.001,0.1,by = 0.0005))

elastic_mod = train(x=trainclean, y=feat$SalePrice[!is.na(feat$SalePrice)], method='glmnet',
                  trControl = mc, tuneGrid = elasticGrid)
elastic_mod$bestTune
elasticVI = varImp(elastic_mod, scale=F)
elasticImp = elasticVI$importance
coef(elastic_mod$finalModel, elastic_mod$bestTune$lambda)

elasticPred = predict(elastic_mod, testclean)
elasticPredPrices = exp(elasticPred)
head(elasticPredPrices)

#XGBoost
labeltrain = feat$SalePrice[!is.na(feat$SalePrice)]
dtrain = xgb.DMatrix(data = as.matrix(trainclean), label = labeltrain)
dtest = xgb.DMatrix(data = as.matrix(testclean))

bestparams = list(objective = "reg:squarederror",
                  booster = "gbtree",
                  eta = 0.1,
                  gamma = 0,
                  max_depth = 5,
                  min_child_weight = 4,
                  subsample = 1,
                  colsample_bytree = 1)

xgbcv = xgb.cv(params = bestparams, data=dtrain, nrounds = 500, nfold = 5, showsd = T,
               stratified = T, print_every_n = 40, early_stopping_rounds = 10, maximize = F)

xgbmod = xgb.train(data = dtrain, params = bestparams, nrounds = 226)

xgbpred = predict(xgbmod, dtest)
xgbPredPrices = exp(xgbpred)
head(xgbPredPrices)

avgPredPrices = data_frame('Id' = testing$Id, 'SalePrice' = (lassoPredPrices+ridgePredPrices+gboostPredPrices+
                                                               elasticPredPrices+xgbPredPrices)/5)
write.csv(avgPredPrices, 'avgPredPrices1.csv', row.names = F)