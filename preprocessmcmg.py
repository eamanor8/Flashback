import json
from pathlib import Path

import pandas as pd

MIN_CHECKINS = 101  # from setting.py; reject users with fewer than this number of checkins
# With 101 checkins, the first 80 will go to train (which results in length-79 X-y pairs due to the off-by-one alignment)
#                    and the latter 21 will go to test (which results in length-20 X-y pairs, length-20 is the minimum
#                        for existence in the test set)

df = pd.read_csv("data/CAL_checkin.csv")
df = df.loc[:, ["UserId", "Local_Time", "Latitude", "Longitude", "VenueId"]]

df.loc[:, "Local_Time"] = pd.to_datetime(df.loc[:, "Local_Time"], dayfirst=True, utc=True)
df.loc[:, "Local_Time"] = df.loc[:, "Local_Time"].apply(lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ"))
df = df.sort_values(by=["UserId", "Local_Time"])

total_checkins_before_drop = len(df)
total_users_before_drop = len(df.loc[:, "UserId"].unique())

num_checkins_per_user = df.loc[:, "UserId"].value_counts()
df.loc[:, "num_checkins_from_this_user"] = df.loc[:, "UserId"].apply(lambda i: num_checkins_per_user[i])
df = df.loc[
    df.loc[:, "num_checkins_from_this_user"] >= MIN_CHECKINS,
    df.columns != "num_checkins_from_this_user"
]
venue_ids = list(df.loc[:, "VenueId"].unique())
df.loc[:, "VenueId"] = df.loc[:, "VenueId"].map(lambda s: venue_ids.index(s))

total_checkins_after_drop = len(df)
total_users_after_drop = len(df.loc[:, "UserId"].unique())

print(f"After dropping users with less than {MIN_CHECKINS} checkins:")
print(f"    {total_checkins_after_drop}/{total_checkins_before_drop} records remain ({total_checkins_after_drop * 100 / total_checkins_before_drop:.2f}%)")
print(f"    {total_users_after_drop}/{total_users_before_drop} users remain ({total_users_after_drop * 100 / total_users_before_drop:.2f}%)")

df.to_csv("data/CAL_checkin.txt", sep="\t", header=False, index=False)
with (Path("data") / "CAL_checkin_venue_ids_reverse_map.json").open("w") as f:
    f.write(json.dumps(
        {i: venue_id for i, venue_id in enumerate(venue_ids)},
        indent=4
    ))
