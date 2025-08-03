import pandas as pd
import numpy as np
import random
def generate_synthetic_insect_data(num_samples=8000):
    np.random.seed(42)
    random.seed(42)
    questions_df = pd.read_csv('characteristics.csv')
    insects = [
        'No_Insect',
        'Armyworm_Green',
        'Armyworm_Brown', 
        'Cutworm',
        'Bollworm',
        'Aphids',
        'Whitefly'
    ]
    insect_patterns = {
        'No_Insect': {
            'armyworm_present': 0.02, 'green_color': 0.05, 'brown_color': 0.05,
            'leaf_top': 0.1, 'leaf_underside': 0.1, 'stem_present': 0.05,
            'feeding_crop': 0.02, 'bite_marks': 0.05, 'multiple_worms': 0.02,
            'frass_visible': 0.02, 'eggs_visible': 0.02, 'larvae_visible': 0.02,
            'previous_attack': 0.15, 'pesticide_applied': 0.4, 'population_increase': 0.05,
            'daylight_active': 0.1, 'night_active': 0.1, 'leaf_affected': 0.1,
            'stem_affected': 0.05, 'damage_small_area': 0.3, 'nearby_plants': 0.1,
            'moving_actively': 0.05, 'curled_leaves': 0.05, 'multiple_sections': 0.05,
            'discoloration': 0.05, 'body_stripes': 0.05, 'length_20mm': 0.05,
            'dead_worms': 0.1, 'chewing_sound': 0.02, 'nearby_reports': 0.1
        },
        'Armyworm_Green': {
            'armyworm_present': 0.95, 'green_color': 0.9, 'brown_color': 0.1,
            'leaf_top': 0.8, 'leaf_underside': 0.6, 'stem_present': 0.4,
            'feeding_crop': 0.9, 'bite_marks': 0.85, 'multiple_worms': 0.7,
            'frass_visible': 0.8, 'eggs_visible': 0.5, 'larvae_visible': 0.9,
            'previous_attack': 0.6, 'pesticide_applied': 0.3, 'population_increase': 0.8,
            'daylight_active': 0.4, 'night_active': 0.9, 'leaf_affected': 0.9,
            'stem_affected': 0.3, 'damage_small_area': 0.4, 'nearby_plants': 0.8,
            'moving_actively': 0.8, 'curled_leaves': 0.7, 'multiple_sections': 0.7,
            'discoloration': 0.6, 'body_stripes': 0.8, 'length_20mm': 0.7,
            'dead_worms': 0.2, 'chewing_sound': 0.6, 'nearby_reports': 0.7
        },
        'Armyworm_Brown': {
            'armyworm_present': 0.95, 'green_color': 0.1, 'brown_color': 0.9,
            'leaf_top': 0.7, 'leaf_underside': 0.8, 'stem_present': 0.5,
            'feeding_crop': 0.9, 'bite_marks': 0.85, 'multiple_worms': 0.6,
            'frass_visible': 0.8, 'eggs_visible': 0.4, 'larvae_visible': 0.9,
            'previous_attack': 0.7, 'pesticide_applied': 0.3, 'population_increase': 0.8,
            'daylight_active': 0.3, 'night_active': 0.95, 'leaf_affected': 0.9,
            'stem_affected': 0.4, 'damage_small_area': 0.3, 'nearby_plants': 0.8,
            'moving_actively': 0.7, 'curled_leaves': 0.8, 'multiple_sections': 0.8,
            'discoloration': 0.7, 'body_stripes': 0.9, 'length_20mm': 0.8,
            'dead_worms': 0.2, 'chewing_sound': 0.7, 'nearby_reports': 0.8
        },
        'Cutworm': {
            'armyworm_present': 0.1, 'green_color': 0.3, 'brown_color': 0.7,
            'leaf_top': 0.3, 'leaf_underside': 0.2, 'stem_present': 0.9,
            'feeding_crop': 0.8, 'bite_marks': 0.6, 'multiple_worms': 0.4,
            'frass_visible': 0.6, 'eggs_visible': 0.3, 'larvae_visible': 0.8,
            'previous_attack': 0.5, 'pesticide_applied': 0.4, 'population_increase': 0.6,
            'daylight_active': 0.2, 'night_active': 0.9, 'leaf_affected': 0.4,
            'stem_affected': 0.9, 'damage_small_area': 0.7, 'nearby_plants': 0.6,
            'moving_actively': 0.5, 'curled_leaves': 0.3, 'multiple_sections': 0.5,
            'discoloration': 0.5, 'body_stripes': 0.6, 'length_20mm': 0.6,
            'dead_worms': 0.3, 'chewing_sound': 0.4, 'nearby_reports': 0.5
        },
        'Bollworm': {
            'armyworm_present': 0.2, 'green_color': 0.6, 'brown_color': 0.4,
            'leaf_top': 0.6, 'leaf_underside': 0.4, 'stem_present': 0.3,
            'feeding_crop': 0.8, 'bite_marks': 0.7, 'multiple_worms': 0.5,
            'frass_visible': 0.7, 'eggs_visible': 0.6, 'larvae_visible': 0.8,
            'previous_attack': 0.6, 'pesticide_applied': 0.4, 'population_increase': 0.7,
            'daylight_active': 0.6, 'night_active': 0.7, 'leaf_affected': 0.8,
            'stem_affected': 0.2, 'damage_small_area': 0.6, 'nearby_plants': 0.7,
            'moving_actively': 0.7, 'curled_leaves': 0.6, 'multiple_sections': 0.6,
            'discoloration': 0.6, 'body_stripes': 0.7, 'length_20mm': 0.5,
            'dead_worms': 0.3, 'chewing_sound': 0.5, 'nearby_reports': 0.6
        },
        'Aphids': {
            'armyworm_present': 0.05, 'green_color': 0.8, 'brown_color': 0.2,
            'leaf_top': 0.9, 'leaf_underside': 0.95, 'stem_present': 0.7,
            'feeding_crop': 0.9, 'bite_marks': 0.3, 'multiple_worms': 0.95,
            'frass_visible': 0.6, 'eggs_visible': 0.8, 'larvae_visible': 0.4,
            'previous_attack': 0.7, 'pesticide_applied': 0.5, 'population_increase': 0.9,
            'daylight_active': 0.8, 'night_active': 0.6, 'leaf_affected': 0.95,
            'stem_affected': 0.6, 'damage_small_area': 0.8, 'nearby_plants': 0.9,
            'moving_actively': 0.4, 'curled_leaves': 0.9, 'multiple_sections': 0.8,
            'discoloration': 0.8, 'body_stripes': 0.1, 'length_20mm': 0.1,
            'dead_worms': 0.4, 'chewing_sound': 0.1, 'nearby_reports': 0.8
        },
        'Whitefly': {
            'armyworm_present': 0.05, 'green_color': 0.1, 'brown_color': 0.1,
            'leaf_top': 0.7, 'leaf_underside': 0.95, 'stem_present': 0.3,
            'feeding_crop': 0.8, 'bite_marks': 0.2, 'multiple_worms': 0.9,
            'frass_visible': 0.4, 'eggs_visible': 0.7, 'larvae_visible': 0.3,
            'previous_attack': 0.6, 'pesticide_applied': 0.6, 'population_increase': 0.8,
            'daylight_active': 0.9, 'night_active': 0.4, 'leaf_affected': 0.9,
            'stem_affected': 0.2, 'damage_small_area': 0.7, 'nearby_plants': 0.9,
            'moving_actively': 0.8, 'curled_leaves': 0.7, 'multiple_sections': 0.6,
            'discoloration': 0.7, 'body_stripes': 0.1, 'length_20mm': 0.1,
            'dead_worms': 0.3, 'chewing_sound': 0.1, 'nearby_reports': 0.8
        }
    }
    feature_names = [
        'armyworm_present', 'green_color', 'brown_color', 'leaf_top', 'leaf_underside',
        'stem_present', 'feeding_crop', 'bite_marks', 'multiple_worms', 'frass_visible',
        'eggs_visible', 'larvae_visible', 'previous_attack', 'pesticide_applied',
        'population_increase', 'daylight_active', 'night_active', 'leaf_affected',
        'stem_affected', 'damage_small_area', 'nearby_plants', 'moving_actively',
        'curled_leaves', 'multiple_sections', 'discoloration', 'body_stripes',
        'length_20mm', 'dead_worms', 'chewing_sound', 'nearby_reports'
    ]
    data = []
    for _ in range(num_samples):
        insect = random.choice(insects)
        pattern = insect_patterns[insect]
        record = {}
        for feature in feature_names:
            probability = pattern.get(feature, 0.1)
            record[feature] = 1 if random.random() < probability else 0
        record['insect'] = insect
        data.append(record)
    df = pd.DataFrame(data)
    return df
print("Generating synthetic insect training data...")
synthetic_data = generate_synthetic_insect_data(8000)
synthetic_data.to_csv('insect_data.csv', index=False)
print("Synthetic insect data generated successfully!")
print(f"Dataset shape: {synthetic_data.shape}")
print("\nInsect distribution:")
print(synthetic_data['insect'].value_counts())
print("\nFirst 5 rows:")
print(synthetic_data.head())