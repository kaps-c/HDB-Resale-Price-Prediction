import pandas as pd


def transaction_frame():
    """Synthetic transactions spanning each evaluation period."""
    rows = []
    for year in (2022, 2023, 2024, 2025):
        for i in range(24):
            area = 50 + i * 3
            rows.append(
                {
                    "month": f"{year}-{1 + i % 2:02d}",
                    "town": "GEYLANG" if i % 2 else "PUNGGOL",
                    "flat_type": "3 ROOM" if i % 2 else "4 ROOM",
                    "street_name": "CIRCUIT RD" if i % 2 else "EDGEFIELD PLAINS",
                    "storey_range": "01 TO 03",
                    "floor_area_sqm": area,
                    "flat_model": "Standard",
                    "lease_commence_date": 1980,
                    "remaining_lease": "54 years 02 months",
                    "resale_price": 150000 + area * 3000 + (year - 2022) * 10000,
                }
            )
    return pd.DataFrame(rows)
