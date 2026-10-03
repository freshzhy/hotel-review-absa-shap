# dataset/

This folder is intentionally empty in the repository. Download the public datasets
below and place them here so that the layout matches what `analysis_v3/common.py` and
`analysis_v3/v3_12_absa_validation.py` expect:

```
dataset/
├── 515K Hotel Reviews Data in Europe/
│   └── Hotel_Reviews.csv
├── Trip Advisor Hotel Reviews/
│   └── tripadvisor_hotel_reviews.csv
└── Hotel Reviews- Aspects, Sentiments and Topics .zip   (contains HRAST.csv; or HRAST.csv directly)
```

- Booking.com — 515K Hotel Reviews Data in Europe:
  https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe
  
- TripAdvisor Hotel Reviews:
  https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews
  
- HRAST — Hotel Reviews: Aspects, Sentiments and Topics (Andreou, Tsapatsoulis &
  Anastasopoulou, 2023; CC BY-NC 4.0), used only to validate aspect detection and VADER polarity.
  
  https://www.kaggle.com/datasets/costastziouvas/hotel-reviews-aspects-sentiments-and-topics

All datasets are third-party and are not redistributed here; see their pages for licensing terms.
