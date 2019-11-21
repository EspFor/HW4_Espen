# Prepare data

library(tidyverse)
library(skimr)
library(readxl)
library(recipes)
library(rsample)
library(dplyr)
library(plyr)
library(httr)
library(rjstat)

options(encoding="UTF-8")

# Read all excel-files
ads_raw <- read_excel("input/ads.xlsx")
geo_raw <- read_excel("input/geo.xlsx")
zip_raw <- read_excel("input/zip.xlsx")
inc_raw <- read_excel("input/income.xlsx")
att_raw <- read_excel("input/attributes.xlsx")

# Select relevant columns
municipalities <- geo_raw %>% 
  select(ad_id, kommune_no, kommune_name, fylke_no, fylke_name)

income <- inc_raw %>% 
  select(zip_no = postnr, avg_income = gjsnitt_inntekt, avg_fortune = gjsnitt_formue)

# Join data
ads <- ads_raw %>% 
  select(-ad_title, -ad_address, -ad_url, -ad_img) %>%
  left_join(municipalities, by = "ad_id") %>% 
  left_join(zip_raw, by = "ad_id") %>% 
  left_join(income, by = "zip_no") %>% 
  left_join(att_raw, by = "ad_id")

# Remove unnecesarry objects
rm(ads_raw, geo_raw, zip_raw, inc_raw, att_raw, income, municipalities)

# Get a quick overview
skimr::skim(ads)
dplyr::count(ads, ad_owner_type, sort = TRUE)
dplyr::count(ads, ad_home_type, sort = TRUE)

# Replace NA and modify variables
ads <- ads %>%
  replace_na(list(ad_debt = 0,
                  ad_expense = 0)) %>%
  mutate(
    ad_tot_price         = ad_price + ad_debt,
    ad_tot_price_per_sqm = ad_tot_price / ad_sqm,
    ad_bedrooms          = parse_number(ad_bedrooms)
  )

#Det er flere rader med verdi "Fellesutg.:" som eierform og "kr" som boligform. Undersøker disse. Det er 78 slike rader.
fellesutg <- subset(ads, ad_owner_type == "Fellesutg.:")

sum(fellesutg$ad_debt) #Ingen i utvalget har gjeld
sum(fellesutg$ad_expense) #Ingen i utvalget har felleskostnader
mean(fellesutg$ad_sqm) #Gjennomsnittlig størrelse på bolig er 161 m2

#Basert på at det ikke er gjeld eller felleskostnader på noen av datapunktene er det rimelig å anta at eierformen er "Eier (Selveier)"
#Basert på at eierformen er selveier og at gjennomsnittlig m2 er 161 er det rimelig å anta at boligformen er enebolig eller tomannsbolig. Settes til enebolig for alle, da dette er den vanligste eierformen.
#Samtlige datapunkter mangler antall soverom, dette tar xgBoost seg av håper jeg

ads <- ads %>%
  mutate(
    ad_owner_type = revalue(ad_owner_type, c("Fellesutg.:" = "Eier (Selveier)")), 
    ad_home_type = revalue(ad_home_type, c("kr" = "Enebolig"))
  )

# Split in train/test
set.seed(42)

finn_split     <- initial_split(ads)
finn_train_raw <- training(finn_split)
finn_test_raw  <- testing(finn_split)

