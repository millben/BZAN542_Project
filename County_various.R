data = read.csv("county_complete.csv")
data = subset(data, select=c("state", "name", "fips", "median_individual_income_2019",
                             "pop_2019", "white_2019", "black_2019", "median_age_2019",
                             "hs_grad_2019", "bachelors_2019", "veterans_2019",
                             "uninsured_2019", "household_has_computer_2019",
                             "household_has_smartphone_2019", "household_has_broadband_2019"))

unique(is.na(data)) # No NAs // all are in percentages of the county with said characteristic.
write.csv(data, "Various_County_Metrics,csv")
