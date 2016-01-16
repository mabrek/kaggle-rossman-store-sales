source("functions.R")

store <- read.csv("../input/store.csv", as.is = TRUE) %>%
  mutate(CompetitionOpenDate = as.POSIXct(
           ifelse(!is.na(CompetitionOpenSinceYear),
                  paste(CompetitionOpenSinceYear,
                        CompetitionOpenSinceMonth,
                        "01", sep='-'),
                  NA),
           format="%Y-%m-%d"))

train_csv <- read.csv("../input/train.csv", as.is = TRUE) %>%
  mutate(Date = as.POSIXct(Date),
         Open = ifelse(Sales == 0, 0, Open))

train_range <- seq(min(train_csv$Date), max(train_csv$Date), by = "DSTday")

train_cartesian <- merge(
  data_frame(Date = train_range, DayOfWeek = eu_wday(train_range),
             Sales = 0, Customers = 0, Open = 0, Promo = 0, 
             StateHoliday = "0", SchoolHoliday = 0),
  data_frame(Store = unique(train_csv$Store)))

train_full <- bind_rows(
  train_csv,
  anti_join(train_cartesian, train_csv, by = c("Date", "Store")) )

test_csv <- read.csv("../input/test.csv", as.is = TRUE) %>%
  mutate(Date = as.POSIXct(Date),
         Open = ifelse(is.na(Open), 1, Open))

test_range <- seq(min(test_csv$Date), max(test_csv$Date), by = "DSTday")

full_range <- seq(min(train_csv$Date), max(test_csv$Date), by = "DSTday")

fourier_terms <- 5
fourier_features <- as.data.frame(
  fourier(ts(seq_along(full_range), frequency = 365), fourier_terms))
fourier_names <- paste("Fourier", seq_len(fourier_terms * 2), sep = '')
colnames(fourier_features) <- fourier_names
fourier_features <- mutate(fourier_features, Date = full_range)

base_linear_features <-
  c("Promo", "Promo2Active", "SchoolHoliday",
    "DayOfWeek1", "DayOfWeek2", "DayOfWeek2", "DayOfWeek3", 
    "DayOfWeek4", "DayOfWeek5", "DayOfWeek6",
    "StateHolidayA", "StateHolidayB", "StateHolidayC",
    "CompetitionOpen", "Open")

trend_features <- c("DateTrend", "DateTrendLog")

decay_features <- c("PromoDecay", "Promo2Decay",
                   "StateHolidayCLastDecay", "StateHolidayCNextDecay",
                   "StateHolidayBNextDecay")

log_decay_features <- paste(decay_features, "Log", sep = "")

stairs_steps <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 28)

stairs_features <-
  c("Opened", "PromoStarted", "Promo2Started", "TomorrowClosed",
    "WasClosedOnSunday",
    cross_paste("PromoStartedLastDate", c(2, 3, 4), "after"),
    cross_paste(c("StateHolidayCNextDate", "StateHolidayBNextDate",
                  "StateHolidayANextDate" ,"LongClosedNextDate"),
                stairs_steps, "before"),
    cross_paste(c("StateHolidayCLastDate","StateHolidayBLastDate",
                  "Promo2StartedDate", "LongOpenLastDate"),
                stairs_steps, "after"))

month_day_features <- paste("MDay", 1:31, sep = "")

month_features <- paste("Month", 12, sep = "")

linear_features <- c(base_linear_features, trend_features, decay_features,
                     log_decay_features, stairs_features, fourier_names,
                     month_day_features, month_features)

glm_features <- c("Promo", "SchoolHoliday",
                  "DayOfWeek1", "DayOfWeek2", "DayOfWeek2", "DayOfWeek3", 
                  "DayOfWeek4", "DayOfWeek5", "DayOfWeek6",
                  "StateHolidayA", "StateHolidayB", "StateHolidayC",
                  "CompetitionOpen", "PromoDecay", "Promo2Decay",
                  "DateTrend", "DateTrendLog",
                  "StateHolidayCLastDecay", "StateHolidayCNextDecay",
                  "Fourier1", "Fourier2", "Fourier3", "Fourier4")

log_lm_features <- c("Promo", "Promo2Active", "SchoolHoliday",
                     "DayOfWeek1", "DayOfWeek2", "DayOfWeek2", "DayOfWeek3", 
                     "DayOfWeek4", "DayOfWeek5", "DayOfWeek6",
                     "StateHolidayA", "StateHolidayB", "StateHolidayC",
                     "CompetitionOpen", "PromoDecay", "Promo2Decay",
                     "DateTrend", "DateTrendLog",
                     "StateHolidayCLastDecay", "StateHolidayCNextDecay",
                     "Fourier1", "Fourier2", "Fourier3", "Fourier4")


categorical_numeric_features <-
  c("Store", "Promo", "Promo2Active", "SchoolHoliday", "DayOfWeek",
    "StateHolidayN", "CompetitionOpen", "StoreTypeN", "AssortmentN",
    "DateTrend", "MDay", "Month", "Year", decay_features, stairs_features,
    fourier_names)

past_date <- as.POSIXct("2000-01-01")
future_date <- as.POSIXct("2099-01-01")
## TODO additive or multiplicative decay?

all_data <- bind_rows(train_full, test_csv) %>%
  inner_join(store, by="Store") %>%
  mutate(DayOfWeek1 = ifelse(DayOfWeek == 1, 1, 0)) %>%
  mutate(DayOfWeek2 = ifelse(DayOfWeek == 2, 1, 0)) %>%
  mutate(DayOfWeek3 = ifelse(DayOfWeek == 3, 1, 0)) %>%
  mutate(DayOfWeek4 = ifelse(DayOfWeek == 4, 1, 0)) %>%
  mutate(DayOfWeek5 = ifelse(DayOfWeek == 5, 1, 0)) %>%
  mutate(DayOfWeek6 = ifelse(DayOfWeek == 6, 1, 0)) %>%
  mutate(StateHolidayA = ifelse(StateHoliday == "a", 1, 0)) %>%
  mutate(StateHolidayB = ifelse(StateHoliday == "b", 1, 0)) %>%
  mutate(StateHolidayC = ifelse(StateHoliday == "c", 1, 0)) %>%
  mutate(StateHolidayN = as.integer(as.factor(StateHoliday))) %>%
  mutate(CompetitionOpen =
           zero_na(ifelse(Date >= CompetitionOpenDate, 1, 0))) %>%
  mutate(StoreTypeN = as.integer(as.factor(StoreType))) %>%
  mutate(AssortmentN = as.integer(as.factor(Assortment))) %>%
  mutate(DateTrend =
           as.integer(difftime(Date, min(Date), unit = 'days')) + 1) %>%
  mutate(
    Promo2Active = Promo2 *
      as.integer(
        (((year(Date) == Promo2SinceYear) &
          (week(Date) >= Promo2SinceWeek)) |
         (year(Date) > Promo2SinceYear)
        ) &
        mapply(function(pat, x) { grepl(pat, x, fixed = TRUE) },
               pat = month(Date, label = TRUE, abbr = TRUE),
               x = PromoInterval))) %>%
  mutate(PromoIntervalN = as.integer(as.factor(PromoInterval))) %>%
  mutate(MDay = mday(Date)) %>%
  mutate(Month = month(Date)) %>%
  mutate(Year = year(Date)) %>%
  make_month_day() %>%
  make_month() %>%
  group_by(Store) %>%
  arrange(Date) %>%
  inner_join(fourier_features, by = "Date") %>%
  mutate(PromoStarted = zero_na((Promo - lag(Promo)) > 0)) %>%
  mutate(
    PromoStartedLastDate = as.POSIXct(
      cummax(ifelse(PromoStarted, Date, past_date)),
      origin = "1970-01-01 00:00:00")) %>%
  mutate(StateHolidayCLastDate =
           as.POSIXct(cummax(ifelse(StateHoliday == "c", Date, past_date)),
                      origin = "1970-01-01 00:00:00")) %>%
  mutate(StateHolidayBLastDate =
           as.POSIXct(cummax(ifelse(StateHoliday == "b", Date, past_date)),
                      origin = "1970-01-01 00:00:00")) %>%
  mutate(Promo2Started = zero_na((Promo2Active - lag(Promo2Active)) > 0)) %>%
  mutate(
    Promo2StartedDate = as.POSIXct(
      cummax(ifelse(Promo2Started, Date, past_date)),
      origin = "1970-01-01 00:00:00")) %>%
  mutate(Opened = zero_na((Open - lag(Open)) > 0)) %>%
  mutate(Closed = zero_na((Open - lag(Open)) < 0)) %>%
  mutate(TomorrowClosed = zero_na(lead(Closed))) %>%
  mutate(
    ClosedLastDate = as.POSIXct(
      cummax(ifelse(Closed, Date, past_date)),
      origin = "1970-01-01 00:00:00")) %>%
  mutate(IsClosedForDays =
           as.integer(difftime(Date, ClosedLastDate, unit = 'days'))) %>%
  mutate(
    OpenedLastDate = as.POSIXct(
      cummax(ifelse(Opened, Date, past_date)),
      origin = "1970-01-01 00:00:00")) %>%
  mutate(
    LastClosedSundayDate = as.POSIXct(
      cummax(ifelse(!Open & (DayOfWeek == 7), Date, past_date)),
      origin = "1970-01-01 00:00:00")) %>%
  mutate(
    WasClosedOnSunday = as.integer(
      as.integer(difftime(Date, LastClosedSundayDate, unit = 'days')) < 7
    )) %>%
  arrange(desc(Date)) %>%
  mutate(StateHolidayCNextDate =
           as.POSIXct(cummin(ifelse(StateHoliday == "c", Date, future_date)),
                      origin = "1970-01-01 00:00:00")) %>%
  mutate(OpenedNextDate =
           as.POSIXct(cummin(ifelse(Opened == 1, Date, future_date)),
                      origin = "1970-01-01 00:00:00")) %>%
  mutate(ClosedNextDate =
           as.POSIXct(cummin(ifelse(Closed, Date, future_date)),
                      origin = "1970-01-01 00:00:00")) %>%
  mutate(
    WillBeClosedForDays =
      as.integer(difftime(OpenedNextDate, Date, unit = 'days'))) %>%
  mutate(
    LongClosedNextDate =
      as.POSIXct(cummin(ifelse(
        Closed & (WillBeClosedForDays > 5) & (WillBeClosedForDays < 180),
        Date, future_date)),
                 origin = "1970-01-01 00:00:00")) %>%
  mutate(StateHolidayBNextDate =
           as.POSIXct(cummin(ifelse(StateHoliday == "b", Date, future_date)),
                      origin = "1970-01-01 00:00:00")) %>%
  mutate(StateHolidayANextDate =
           as.POSIXct(cummin(ifelse(StateHoliday == "a", Date, future_date)),
                      origin = "1970-01-01 00:00:00")) %>%
  arrange(Date) %>%
  mutate(
    LongOpenLastDate =
      as.POSIXct(cummax(ifelse(
        Opened & (IsClosedForDays > 5) & (IsClosedForDays < 180),
        Date, past_date)),
                origin = "1970-01-01 00:00:00")) %>%
  ungroup %>%
  make_decay_features(promo_after = 4, promo2_after = 3,
                      holiday_b_before = 3,
                      holiday_c_before = 15, holiday_c_after = 3) %>%
  scale_log_features(c(decay_features, "DateTrend")) %>%
  make_before_stairs(c("StateHolidayCNextDate", "StateHolidayBNextDate",
                       "StateHolidayANextDate", "LongClosedNextDate"),
                     stairs_steps) %>%
  make_after_stairs("PromoStartedLastDate", c(2, 3, 4)) %>%
  make_after_stairs(c("StateHolidayCLastDate", "StateHolidayBLastDate",
                      "Promo2StartedDate", "LongOpenLastDate"),
                    stairs_steps)

train <- all_data %>%
  filter(Date >= min(train_range), Date <= max(train_range)) %>%
  select(-Id)

test <- all_data %>%
  filter(Date >= min(test_range), Date <= max(test_range)) %>%
  select(-Sales, -Customers)

example_stores <- c(
  388, # most typical by svd of Sales time series
  562, # second most typical by svd
  851, # large gap in 2014
  357  # small gap
)

small_train <- filter(train, is.element(Store, example_stores))
small_fold <- make_fold(small_train)

one_train <- filter(train, Store == 388)
one_test <- filter(test, Store == 388)

seed <- 9623125
set.seed(seed)
