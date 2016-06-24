# Solution for [Kaggle Rossmann Store Sales Competition](https://www.kaggle.com/c/rossmann-store-sales)

Result: 66th from 3303

[Short strategy description](/strategy.md)

[Detailed description](http://mabrek.github.io/blog/kaggle-forecasting)

## Dependencies

- R
    - xts
    - parallel
    - forecast
    - shiny
    - dygraphs
    - ggplot2
    - lubridate
    - xgboost
    - glmnet
    - e1071
    - dplyr

## Code Layout

* `input/` unpack data files here (`store.csv`, `test.csv`, `train.csv`)
* `R/` scripts
    * [`functions.R`](R/functions.R) function definitions
    * [`data.R`](R/data.R) data loading, feature generation
    * [`playground.R`](R/playground.R) code snippets used for experiments and submissions


```
Copyright 2015 Anton Lebedevich

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
