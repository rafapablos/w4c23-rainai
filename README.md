![Title](/images/weather4cast_v1000-26.png?raw=true "Weather4cast competition")

# RainAI - Weather4Cast 2023 Results
This repository contains the code and trained models presented for the Weather4Cast 2023 NeurIPS competition. 
A detailed description of the approach is available as a short scientific paper at **ADD ARXIV**.

```
@ ADD CITATION
``````

# [Weather4cast](https://www.weather4cast.org) - Super-Resolution Rain Movie Prediction under Spatio-Temporal Shifts

-   Predict super-resolution rain movies in various regions of Europe
-   Transfer learning across space and time under strong shifts
-   Exploit data fusion to model ground-radar and multi-band satellite images

## Contents
-   [Introduction](#introduction)
-   [Prediction task](#prediction-task)
-   [Cite](#citation)
-   [Credits](#credits)

## Introduction

The aim of the 2023 edition of the Weather4cast competition is to predict **quantitatively** future high resolution rainfall events from lower resolution satellite radiances. Ground-radar reflectivity measurements are used to calculate pan-European composite rainfall rates by the [Operational Program for Exchange of Weather Radar Information (OPERA)](https://www.eumetnet.eu/activities/observations-programme/current-activities/opera/) radar network. While these are more precise, accurate, and of higher resolution than satellite data, they are expensive to obtain and not available in many parts of the world. We thus want to learn how to predict this high value rain rates from radiation measured by geostationary satellites operated by the [European Organisation for the Exploitation of Meteorological Satellites (EUMETSAT)](https://www.eumetsat.int/).

## Prediction task

Competition participants should predict the exact amount of rainfall for the next 8 hours in 32 time slots from an input sequence of 4 time slots of the preceeding hour. The input sequence consists of four 11-band spectral satellite images. These 11 channels show slightly noisy satellite radiances covering so-called visible (VIS), water vapor (WV), and infrared (IR) bands. Each satellite image covers a 15 minute period and its pixels correspond to a spatial area of about 12km x 12km. The prediction output is a sequence of 32 images representing rain rates from ground-radar reflectivities. Output images also have a temporal resolution of 15 minutes but have higher spatial resolution, with each pixel corresponding to a spatial area of about 2km x 2km. So in addition to predicting the weather in the future, converting satellite inputs to ground-radar outputs, this adds a super-resolution task due to the coarser spatial resolution of the satellite data

### Weather4cast 2023 dataset

We provide data from 10 Eureopean regions selected based on their preciptation characteristics for 2019, 2020 and 2021. In total there then are 7 regions with full training data in both 2019 and 2020. Those regions then be used for training, while three additional regions provide a spatial transfer learning challenge in years 2019 and 2020. For all ten regions, the year 2021 provides a temporal transfer learning challenge.

### Core Challege dataset

For the Core Challege we provide data from 7 Eureopean regions selected based on their preciptation characteristics for two years covering(`boxi_0015`, `boxi_0034`, `boxi_0076`, `roxi_0004`, `roxi_0005`, `roxi_0006`, and `roxi_0007`). This data covers the time February to December 2019 and January to December 2020.

The task is to predict exact amount of rain events 4 hours into the future from a 1 hour sequence of satellite images. Rain rates computed from OPERA ground-radar reflectivities provide a ground truth.

### Transfer Learning Challege dataset

<!---
During the Phase2 of the competition the metric(s) derived from the exploratory Phase1 will be applied to test models performance.


Phase 2 will consist of Core Challenge (as desribed above) and the Transfer Learning Challege.

-->

For the Transfer Learning Challege we provide satellite data for additional 3 regions (`roxi_0008`, `roxi_0009` and `roxi_0010`) for years 2019 and 2020 and 10 all regions in 2021. New regions provide a spatial transfer learning challenge in years 2019 and 2020 and a temporal transfer learning challenge in 2021. For the seven regions with extensive training data in 2019 and 2020 this constitutes a pure temporal transfer learning challenge.


## Citation

When using or referencing the Weather4cast Competition in general or the competition data please cite:

```
@InProceedings{pmlr-v220-gruca22a,
  title = 	 {Weather4cast at NeurIPS 2022: Super-Resolution Rain Movie Prediction under Spatio-temporal Shifts},
  author =       {Gruca, Aleksandra and Serva, Federico and Lliso, Lloren\c{c} and R\'ipodas, Pilar and Calbet, Xavier and Herruzo, Pedro and Pihrt, Ji\v{r}\'{\i} and Raevskyi, Rudolf and \v{S}im\'{a}nek, Petr and Choma, Matej and Li, Yang and Dong, Haiyu and Belousov, Yury and Polezhaev, Sergey and Pulfer, Brian and Seo, Minseok and Kim, Doyi and Shin, Seungheon and Kim, Eunbin and Ahn, Sewoong and Choi, Yeji and Park, Jinyoung and Son, Minseok and Cho, Seungju and Lee, Inyoung and Kim, Changick and Kim, Taehyeon and Kang, Shinhwan and Shin, Hyeonjeong and Yoon, Deukryeol and Eom, Seongha and Shin, Kijung and Yun, Se-Young and {Le Saux}, Bertrand and Kopp, Michael K and Hochreiter, Sepp and Kreil, David P},
  booktitle = 	 {Proceedings of the NeurIPS 2022 Competitions Track},
  pages = 	 {292--313},
  year = 	 {2022},
  editor = 	 {Ciccone, Marco and Stolovitzky, Gustavo and Albrecht, Jacob},
  volume = 	 {220},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {28 Nov--09 Dec},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/v220/gruca22a.html},
}


@INPROCEEDINGS{9672063,
author={Herruzo, Pedro and Gruca, Aleksandra and Lliso, Llorenç and Calbet, Xavier and Rípodas, Pilar and Hochreiter, Sepp and Kopp, Michael and Kreil, David P.},
booktitle={2021 IEEE International Conference on Big Data (Big Data)},
title={High-resolution multi-channel weather forecasting – First insights on transfer learning from the Weather4cast Competitions 2021},
year={2021},
volume={},
number={},
pages={5750-5757},
doi={10.1109/BigData52589.2021.9672063}
}

@inbook{10.1145/3459637.3482044,
author = {Gruca, Aleksandra and Herruzo, Pedro and R\'{\i}podas, Pilar and Kucik, Andrzej and Briese, Christian and Kopp, Michael K. and Hochreiter, Sepp and Ghamisi, Pedram and Kreil, David P.},
title = {CDCEO'21 - First Workshop on Complex Data Challenges in Earth Observation},
year = {2021},
isbn = {9781450384469},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3459637.3482044},
booktitle = {Proceedings of the 30th ACM International Conference on Information &amp; Knowledge Management},
pages = {4878–4879},
numpages = {2}
}
```

## Credits

The competition is organized / supported by:

-   [Silesian University of Technology, Poland](https://polsl.pl)
-   [Spanish State Meteorological Agency, AEMET, Spain](http://aemet.es/)
