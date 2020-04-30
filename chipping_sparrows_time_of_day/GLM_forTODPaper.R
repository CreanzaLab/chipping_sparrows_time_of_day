##Set directory here
setwd("C:/Users/abiga/Box Sync/Abigail_Nicole/ChippiesTimeOfDay/GLMM_TOD")

### Loads packages and my functions
source("MetaFunctionUpdate_reduced.R")

dir <- getwd()

FinalDfCombo <- read.csv('FinalChippiesDataReExportedAs44100Hz_LogTransformed_forTOD_wSunrise.csv', header = TRUE, na.strings=c("","NA"))

# calculate variance for each columns/syllable variable
p.varfull <- apply(FinalDfCombo[, 9:12], 2, var)

set.seed(49)

song_var <- colnames(FinalDfCombo)[9:12]
fixed_effects = list('Latitude', 'RecordingMonth', 'Sunrise')
glm <- list()

# loop through song variables, all continous fixed effects together, no random
# remove NA's from dataframe for recording time (will take care of any empties in sunrise)
df_rmTimeNA <- subset(FinalDfCombo, Sunrise != "--")
df_beforeSunriseMorning <- subset(df_rmTimeNA, Sunrise !="after noon")
df_beforeSunriseMorning$Sunrise <- droplevels(df_beforeSunriseMorning)$Sunrise
table(df_beforeSunriseMorning$Sunrise)

#glm
for (feature in song_var){
  print(feature)
  loop = paste(feature)
  glm[[loop]] <- glm(as.formula(paste0(feature, "~", fixed_effects[1], "+", fixed_effects[2], "+", fixed_effects[3], "+", fixed_effects[1], ":", fixed_effects[2])), data=df_beforeSunriseMorning)
}

for (i in seq_along(song_var)){
  print(glm[[i]]$formula)
  print(glm[[i]]$converged)
  print(song_var[i])
  print(summary(glm[[i]]))
}

