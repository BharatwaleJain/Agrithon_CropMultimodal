import pandas as pd
import numpy as np
import random
from itertools import combinations
def generate_synthetic_crop_data(num_samples=8000):
    np.random.seed(42)
    random.seed(42)
    questions_df = pd.read_csv('characteristics.csv')
    diseases = [
        'Healthy',
        'Early_Blight', 
        'Late_Blight',
        'Bacterial_Spot',
        'Septoria_Leaf_Spot',
        'Target_Spot',
        'Leaf_Mold'
    ]
    disease_patterns = {
        'Healthy': {
            'yellow_halo': 0.05, 'circular_rings': 0.02, 'lower_leaves': 0.1, 
            'lesions_expanding': 0.02, 'dry_brown_center': 0.03, 'spots_merging': 0.02,
            'early_yellowing': 0.1, 'stems_fruits_affected': 0.02, 'wilting': 0.05,
            'spreading_upward': 0.02, 'concentric_rings': 0.02, 'fruit_rot': 0.02,
            'brown_margins': 0.05, 'moisture_stress': 0.2, 'active_rainy': 0.1,
            'nearby_plants': 0.1, 'black_mold': 0.02, 'whole_plant': 0.02,
            'large_spots': 0.02, 'both_sides': 0.1, 'mature_leaves': 0.1,
            'veins_visible': 0.05, 'uniform_damage': 0.1, 'previous_history': 0.15,
            'resistant_varieties': 0.6, 'fungicide_applied': 0.4, 'poor_circulation': 0.3,
            'overhead_irrigation': 0.3, 'sanitation_practices': 0.7, 'other_crops': 0.1
        },
        'Early_Blight': {
            'yellow_halo': 0.85, 'circular_rings': 0.9, 'lower_leaves': 0.95, 
            'lesions_expanding': 0.9, 'dry_brown_center': 0.85, 'spots_merging': 0.8,
            'early_yellowing': 0.8, 'stems_fruits_affected': 0.7, 'wilting': 0.6,
            'spreading_upward': 0.85, 'concentric_rings': 0.95, 'fruit_rot': 0.6,
            'brown_margins': 0.7, 'moisture_stress': 0.6, 'active_rainy': 0.7,
            'nearby_plants': 0.8, 'black_mold': 0.3, 'whole_plant': 0.7,
            'large_spots': 0.8, 'both_sides': 0.7, 'mature_leaves': 0.9,
            'veins_visible': 0.4, 'uniform_damage': 0.7, 'previous_history': 0.8,
            'resistant_varieties': 0.2, 'fungicide_applied': 0.3, 'poor_circulation': 0.8,
            'overhead_irrigation': 0.7, 'sanitation_practices': 0.3, 'other_crops': 0.6
        },
        'Late_Blight': {
            'yellow_halo': 0.7, 'circular_rings': 0.4, 'lower_leaves': 0.8, 
            'lesions_expanding': 0.95, 'dry_brown_center': 0.6, 'spots_merging': 0.9,
            'early_yellowing': 0.85, 'stems_fruits_affected': 0.9, 'wilting': 0.85,
            'spreading_upward': 0.9, 'concentric_rings': 0.3, 'fruit_rot': 0.8,
            'brown_margins': 0.8, 'moisture_stress': 0.8, 'active_rainy': 0.95,
            'nearby_plants': 0.9, 'black_mold': 0.7, 'whole_plant': 0.9,
            'large_spots': 0.9, 'both_sides': 0.8, 'mature_leaves': 0.7,
            'veins_visible': 0.3, 'uniform_damage': 0.8, 'previous_history': 0.7,
            'resistant_varieties': 0.3, 'fungicide_applied': 0.4, 'poor_circulation': 0.8,
            'overhead_irrigation': 0.9, 'sanitation_practices': 0.2, 'other_crops': 0.7
        },
        'Bacterial_Spot': {
            'yellow_halo': 0.9, 'circular_rings': 0.2, 'lower_leaves': 0.6, 
            'lesions_expanding': 0.7, 'dry_brown_center': 0.8, 'spots_merging': 0.7,
            'early_yellowing': 0.7, 'stems_fruits_affected': 0.8, 'wilting': 0.5,
            'spreading_upward': 0.6, 'concentric_rings': 0.1, 'fruit_rot': 0.7,
            'brown_margins': 0.8, 'moisture_stress': 0.4, 'active_rainy': 0.8,
            'nearby_plants': 0.7, 'black_mold': 0.2, 'whole_plant': 0.6,
            'large_spots': 0.4, 'both_sides': 0.9, 'mature_leaves': 0.5,
            'veins_visible': 0.6, 'uniform_damage': 0.6, 'previous_history': 0.6,
            'resistant_varieties': 0.3, 'fungicide_applied': 0.5, 'poor_circulation': 0.7,
            'overhead_irrigation': 0.8, 'sanitation_practices': 0.3, 'other_crops': 0.5
        },
        'Septoria_Leaf_Spot': {
            'yellow_halo': 0.8, 'circular_rings': 0.7, 'lower_leaves': 0.9, 
            'lesions_expanding': 0.8, 'dry_brown_center': 0.9, 'spots_merging': 0.6,
            'early_yellowing': 0.8, 'stems_fruits_affected': 0.3, 'wilting': 0.4,
            'spreading_upward': 0.8, 'concentric_rings': 0.6, 'fruit_rot': 0.2,
            'brown_margins': 0.6, 'moisture_stress': 0.5, 'active_rainy': 0.8,
            'nearby_plants': 0.7, 'black_mold': 0.4, 'whole_plant': 0.5,
            'large_spots': 0.3, 'both_sides': 0.8, 'mature_leaves': 0.9,
            'veins_visible': 0.7, 'uniform_damage': 0.6, 'previous_history': 0.7,
            'resistant_varieties': 0.2, 'fungicide_applied': 0.3, 'poor_circulation': 0.7,
            'overhead_irrigation': 0.6, 'sanitation_practices': 0.4, 'other_crops': 0.4
        },
        'Target_Spot': {
            'yellow_halo': 0.7, 'circular_rings': 0.9, 'lower_leaves': 0.8, 
            'lesions_expanding': 0.8, 'dry_brown_center': 0.7, 'spots_merging': 0.8,
            'early_yellowing': 0.7, 'stems_fruits_affected': 0.6, 'wilting': 0.5,
            'spreading_upward': 0.7, 'concentric_rings': 0.85, 'fruit_rot': 0.5,
            'brown_margins': 0.8, 'moisture_stress': 0.6, 'active_rainy': 0.8,
            'nearby_plants': 0.7, 'black_mold': 0.3, 'whole_plant': 0.6,
            'large_spots': 0.7, 'both_sides': 0.8, 'mature_leaves': 0.8,
            'veins_visible': 0.5, 'uniform_damage': 0.7, 'previous_history': 0.6,
            'resistant_varieties': 0.3, 'fungicide_applied': 0.4, 'poor_circulation': 0.7,
            'overhead_irrigation': 0.7, 'sanitation_practices': 0.3, 'other_crops': 0.5
        },
        'Leaf_Mold': {
            'yellow_halo': 0.6, 'circular_rings': 0.3, 'lower_leaves': 0.9, 
            'lesions_expanding': 0.7, 'dry_brown_center': 0.4, 'spots_merging': 0.8,
            'early_yellowing': 0.9, 'stems_fruits_affected': 0.2, 'wilting': 0.6,
            'spreading_upward': 0.8, 'concentric_rings': 0.2, 'fruit_rot': 0.1,
            'brown_margins': 0.5, 'moisture_stress': 0.7, 'active_rainy': 0.9,
            'nearby_plants': 0.8, 'black_mold': 0.9, 'whole_plant': 0.7,
            'large_spots': 0.6, 'both_sides': 0.9, 'mature_leaves': 0.9,
            'veins_visible': 0.8, 'uniform_damage': 0.8, 'previous_history': 0.6,
            'resistant_varieties': 0.2, 'fungicide_applied': 0.3, 'poor_circulation': 0.9,
            'overhead_irrigation': 0.4, 'sanitation_practices': 0.2, 'other_crops': 0.4
        }
    }
    feature_names = [
        'yellow_halo', 'circular_rings', 'lower_leaves', 'lesions_expanding', 
        'dry_brown_center', 'spots_merging', 'early_yellowing', 'stems_fruits_affected',
        'wilting', 'spreading_upward', 'concentric_rings', 'fruit_rot', 'brown_margins',
        'moisture_stress', 'active_rainy', 'nearby_plants', 'black_mold', 'whole_plant',
        'large_spots', 'both_sides', 'mature_leaves', 'veins_visible', 'uniform_damage',
        'previous_history', 'resistant_varieties', 'fungicide_applied', 'poor_circulation',
        'overhead_irrigation', 'sanitation_practices', 'other_crops'
    ]
    data = []
    for _ in range(num_samples):
        disease = random.choice(diseases)
        pattern = disease_patterns[disease]
        record = {}
        for feature in feature_names:
            probability = pattern.get(feature, 0.1)
            record[feature] = 1 if random.random() < probability else 0
        record['disease'] = disease
        data.append(record)
    df = pd.DataFrame(data)
    return df
print("Generating synthetic training data...")
synthetic_data = generate_synthetic_crop_data(8000)
synthetic_data.to_csv('data.csv', index=False)
print("Synthetic data generated successfully!")
print(f"Dataset shape: {synthetic_data.shape}")
print("\nDisease distribution:")
print(synthetic_data['disease'].value_counts())
print("\nFirst 5 rows:")
print(synthetic_data.head())