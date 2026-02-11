import pickle
import pandas as pd

# Load model and features
model = pickle.load(open('app/models/revenue_model.pkl', 'rb'))
features = pickle.load(open('app/models/feature_columns.pkl', 'rb'))

# Get feature importance
importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print('=' * 50)
print('TOP 15 FEATURE IMPORTANCE RANKING')
print('=' * 50)
for i, row in importance.head(15).iterrows():
    bar = '#' * int(row['importance'] * 40)
    yt = ' (YouTube)' if 'trailer' in row['feature'] else ''
    print(f"{row['feature'][:25]:25} {bar} {row['importance']:.4f}{yt}")

print()
print('YOUTUBE FEATURES RANKING:')
yt_features = importance[importance['feature'].str.contains('trailer')]
for i, row in yt_features.iterrows():
    rank = list(importance['feature']).index(row['feature']) + 1
    print(f"   #{rank}: {row['feature']} = {row['importance']:.4f}")
