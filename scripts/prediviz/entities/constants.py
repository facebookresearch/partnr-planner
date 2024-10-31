#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source

# Dictionary mapping category to normalized RGB tuple
category_color_map = {
    "sports_equipment": (1.0, 0.5, 0.0),  # Orange
    "stationery": (0.95, 0.6, 0.1),  # Amber
    "kitchenware": (1.0, 0.8, 0.2),  # Gold
    "toys": (0.9, 0.4, 0.1),  # Tangerine
    "electronics": (0.8, 0.2, 0.0),  # Vermilion
    "food": (0.9, 0.3, 0.2),  # Tomato
    "clothing": (1.0, 0.6, 0.4),  # Coral
    "household_items": (1.0, 0.4, 0.3),  # Salmon
    "miscellaneous": (0.95, 0.5, 0.3),  # Apricot
    "tools": (0.85, 0.4, 0.2),  # Burnt Sienna
    "accessories": (0.95, 0.6, 0.5),  # Peach
    "furniture": (0.95, 0.8, 0.7),  # Beige
}

# Dictionary mapping receptacle type to normalized RGB tuple
receptacle_color_map = {
    "table": (0.95, 0.8, 0.2),  # Gold
    "couch": (1.0, 0.6, 0.4),  # Coral
    "chair": (0.8, 0.5, 0.3),  # Copper
    "shelves": (1.0, 0.5, 0.2),  # Tangerine
    "stand": (0.9, 0.4, 0.1),  # Tawny
    "chest_of_drawers": (1.0, 0.6, 0.5),  # Peach
    "washer_dryer": (0.9, 0.3, 0.2),  # Tomato
    "bathtub": (0.8, 0.2, 0.0),  # Vermilion
    "stool": (1.0, 0.4, 0.3),  # Salmon
    "bench": (0.95, 0.5, 0.3),  # Apricot
    "cabinet": (0.85, 0.4, 0.2),  # Burnt Sienna
    "counter": (0.95, 0.6, 0.1),  # Amber
    "fridge": (0.3, 0.75, 0.93),  # Light Sky Blue
    "bed": (0.94, 0.9, 0.55),  # Pale Goldenrod
    "trashcan": (0.5, 0.5, 0.5),  # Gray
    "microwave": (0.4, 0.8, 0.8),  # Medium Turquoise
    "dishwasher": (0.6, 0.8, 1.0),  # Light Blue
    "oven": (0.9, 0.3, 0.3),  # Dark Red
    "sink": (0.7, 0.9, 1.0),  # Light Blue
    "toilet": (0.6, 0.9, 0.7),  # Light Green
    "wardrobe": (0.5, 0.3, 0.1),  # Saddle Brown
    "shower": (0.5, 0.5, 1.0),  # Cornflower Blue
    "filing_cabinet": (0.9, 0.6, 0.2),  # Dark Goldenrod
    "car": (0.0, 0.0, 0.8),  # Blue
    "unknown": (0.7, 0.7, 0.7),  # Silver
}


object_category_map = {
    "action_figure": "toys",
    "android_figure": "toys",
    "apple": "kitchenware",
    "backpack": "clothing",
    "baseballbat": "sports_equipment",
    "basket": "household_items",
    "basketball": "sports_equipment",
    "bath_towel": "household_items",
    "battery_charger": "electronics",
    "board_game": "toys",
    "book": "stationery",
    "bottle": "kitchenware",
    "bowl": "kitchenware",
    "box": "miscellaneous",
    "bread": "kitchenware",
    "bundt_pan": "kitchenware",
    "butter_dish": "kitchenware",
    "c-clamp": "tools",
    "cake_pan": "kitchenware",
    "can": "kitchenware",
    "can_opener": "kitchenware",
    "candle": "household_items",
    "candle_holder": "household_items",
    "candy_bar": "food",
    "canister": "kitchenware",
    "carrying_case": "miscellaneous",
    "casserole": "kitchenware",
    "cellphone": "electronics",
    "clock": "household_items",
    "credit_card": "stationery",
    "cup": "kitchenware",
    "cushion": "furniture",
    "doll": "toys",
    "dumbbell": "sports_equipment",
    "egg": "food",
    "electric_kettle": "kitchenware",
    "electronic_cable": "electronics",
    "file_sorter": "stationery",
    "folder": "stationery",
    "fork": "kitchenware",
    "gaming_console": "electronics",
    "glass": "kitchenware",
    "hammer": "tools",
    "hand_towel": "household_items",
    "handbag": "clothing",
    "hard_drive": "electronics",
    "hat": "clothing",
    "helmet": "sports_equipment",
    "jug": "kitchenware",
    "kettle": "kitchenware",
    "keychain": "miscellaneous",
    "knife": "kitchenware",
    "ladle": "kitchenware",
    "lamp": "household_items",
    "laptop": "electronics",
    "laptop_cover": "electronics",
    "laptop_stand": "electronics",
    "lunch_box": "kitchenware",
    "milk_frother_cup": "kitchenware",
    "monitor_stand": "electronics",
    "mouse_pad": "electronics",
    "multiport_hub": "electronics",
    "pan": "kitchenware",
    "pen": "stationery",
    "pencil_case": "stationery",
    "phone_stand": "electronics",
    "picture_frame": "household_items",
    "pitcher": "kitchenware",
    "plant_container": "household_items",
    "plant_saucer": "household_items",
    "plate": "kitchenware",
    "potato": "food",
    "ramekin": "kitchenware",
    "scissors": "stationery",
    "screwdriver": "tools",
    "shoe": "clothing",
    "soap_dish": "household_items",
    "soap_dispenser": "household_items",
    "spatula": "kitchenware",
    "spectacles": "stationery",
    "spicemill": "kitchenware",
    "sponge": "household_items",
    "spoon": "kitchenware",
    "spray_bottle": "household_items",
    "squeezer": "kitchenware",
    "statue": "household_items",
    "stuffed toy": "toys",
    "stuffed_toy": "toys",
    "sushi_mat": "kitchenware",
    "tape": "stationery",
    "teapot": "kitchenware",
    "tennis_racquet": "sports_equipment",
    "tomato": "food",
    "toy_airplane": "toys",
    "toy_animal": "toys",
    "toy_bee": "toys",
    "toy_cactus": "toys",
    "toy_construction_set": "toys",
    "toy_fire_truck": "toys",
    "toy_food": "toys",
    "toy_fruits": "toys",
    "toy_pineapple": "toys",
    "toy_swing": "toys",
    "toy_vehicle": "toys",
    "tray": "kitchenware",
    "vase": "household_items",
    "watch": "accessories",
}

receptacle_properties = {
    "table": {"is_on_top": True, "is_inside": False, "is_same": False},
    "couch": {"is_on_top": True, "is_inside": False, "is_same": False},
    "chair": {"is_on_top": True, "is_inside": False, "is_same": False},
    "shelves": {"is_on_top": True, "is_inside": True, "is_same": False},
    "stand": {"is_on_top": True, "is_inside": False, "is_same": False},
    "chest_of_drawers": {
        "is_on_top": True,
        "is_inside": True,
        "is_same": False,
    },
    "wardrobe": {"is_on_top": True, "is_inside": True, "is_same": False},
    "washer_dryer": {"is_on_top": True, "is_inside": True, "is_same": False},
    "bathtub": {"is_on_top": True, "is_inside": True, "is_same": True},
    "stool": {"is_on_top": True, "is_inside": False, "is_same": False},
    "bench": {"is_on_top": True, "is_inside": False, "is_same": False},
    "cabinet": {"is_on_top": True, "is_inside": True, "is_same": False},
    "counter": {"is_on_top": True, "is_inside": False, "is_same": False},
    "fridge": {"is_on_top": True, "is_inside": True, "is_same": False},
    "bed": {"is_on_top": True, "is_inside": False, "is_same": False},
    "trashcan": {"is_on_top": True, "is_inside": True, "is_same": True},
    "microwave": {"is_on_top": True, "is_inside": True, "is_same": False},
    "dishwasher": {"is_on_top": True, "is_inside": True, "is_same": False},
    "oven": {"is_on_top": True, "is_inside": False, "is_same": False},
    "sink": {"is_on_top": True, "is_inside": True, "is_same": True},
    "toilet": {"is_on_top": True, "is_inside": True, "is_same": False},
    "shower": {"is_on_top": True, "is_inside": False, "is_same": False},
    "car": {"is_on_top": True, "is_inside": True, "is_same": False},
    "unknown": {"is_on_top": True, "is_inside": True, "is_same": False},
    "filing_cabinet": {"is_on_top": True, "is_inside": True, "is_same": False},
}

# Define a color palette for the propositions
color_palette = {
    "Soft Coral": "#FF6F61",
    "Butter Yellow": "#FFDB58",
    "Mint": "#98FF98",
    "Sky Blue": "#87CEEB",
    "Peach": "#FFDAB9",
    "Blush Pink": "#FFB6C1",
    "Cool Cyan": "#00CED1",
    "Warm Orange": "#FFA07A",
    "Soft Violet": "#DDA0DD",
    "Lavender": "#E6E6FA",
    "Medium Purple": "#9370DB",
    "Canary Yellow": "#FFFF00",
    "Mauve": "#E0B0FF",
    "Pale Gold": "#EEE8AA",
    "Turquoise": "#40E0D0",
    "Pale Yellow": "#FFFACD",
    "Light Blue": "#ADD8E6",
    "Pale Pink": "#FADADD",
    "Periwinkle": "#CCCCFF",
    "Light Lime": "#BFFF00",
    "Soft Teal": "#66CDAA",
    "Light Rose": "#FFCCE5",
    "Amethyst": "#9966CC",
}
ROOM_COLOR = "#5A6F8E"
BACKGROUND_COLOR = "#3E4C60"
LEGEND_BACKGROUND_COLOR = "#2D3541"
RECEPTACLE_ICONS_PATH = "scripts/prediviz/assets/receptacles"
CROPPED_RECEPTACLE_ICONS_PATH = "scripts/prediviz/assets/cropped_receptacles"
OBJECT_STATE_ICONS_PATH = "scripts/prediviz/assets/object_states"
FONTS_DIR_PATH = "scripts/prediviz/assets/fonts"
