# !pip install -q pycaret[full]
from pycaret.classification import *
import pandas as pd

data = pd.read_csv("/content/sample_data/penguins_100k.csv")

data.head(10)

clf = setup(data, target = "species")

# compare models
clf.compare_models()

best = clf.compare_models()

# plot feature importance
plot_model(best, plot = "feature")

# check to see available plots
help(plot_model)

# evaluate best model
evaluate_model(best)

# prediction
prediction = predict_model(best)

prediction.head(20)

# predict on new data
new_data = pd.DataFrame({
    "island": "Torgersen",
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181.0,
    "body_mass_g": 3750.0,
    "sex": "Female",
    "year": 2009
}, index = [0])

prediction = predict_model(best, data = new_data)
prediction.head()

# save model
save_model(best, "penguins_best_model")