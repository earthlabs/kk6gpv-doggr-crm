import plotly.graph_objects as go
from pymongo import MongoClient
import pandas as pd
import numpy as np

client = MongoClient(
    "mongodb+srv://kk6gpv:kk6gpv@cluster0-kglzh.azure.mongodb.net/test?retryWrites=true&w=majority"
)
db = client.petroleum

header = get_graph_oilgas(str(api))

df, offsets = get_offsets_oilgas(header, 0.1, 35)

df_cons = pd.DataFrame(
    list(
        db.doggr.aggregate(
            [
                {"$unwind": "$crm.cons"},
            ]
        )
    )
)
cons = []
for row in df_cons["crm"]:
    cons.append(row["cons"])

fig_mapb = go.Figure()

for con in cons:
    if con["gain"] > 0.5:
        clr = "#f542d7"
    elif con["gain"] > 0.4:
        clr = "#f5428a"
    elif con["gain"] > 0.3:
        clr = "#f55d42"
    elif con["gain"] > 0.1:
        clr = "#f5c542"
    else:
        clr = "#aaf542"
    if con["gain"] > 0.5:
        fig_mapb.add_trace(
            go.Scattermapbox(
                lon=[con["x0"], con["x1"]],
                lat=[con["y0"], con["y1"]],
                mode="lines",
                line=dict(color=clr, width=1 + con["gain"] * 10),
            )
        )

fig_mapb.update_layout(
    margin={"l": 0, "t": 0, "b": 0, "r": 0},
    mapbox={
        "style": "white-bg",
        "layers": [
            {
                "below": "traces",
                "sourcetype": "raster",
                "source": [
                    "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
                ],
            }
        ],
        "center": {"lon": header["longitude"], "lat": header["latitude"]},
        "zoom": 17,
    },
)

fig_mapb.show()

# # serialize 2D array y
# record['feature2'] = pymongo.binary.Binary( pickle.dumps( y, protocol=2) ) )

# # deserialize 2D array y
# y = pickle.loads( record['feature2'] )
