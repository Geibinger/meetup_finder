import googlemaps
import folium
import numpy as np
import branca.colormap as cm
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import json
import os
import matplotlib.pyplot as plt
import geojsoncontour

# Initialize Google Maps client
gmaps = googlemaps.Client(key=os.environ.get("GOOGLE_MAPS_API_KEY"))

# Load cached travel time data
cache_file = "travel_times.cache.json"
if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
        travel_time_cache = json.load(f)
else:
    travel_time_cache = {}


# Function to save cached data
def save_cache():
    with open(cache_file, "w") as f:
        json.dump(travel_time_cache, f)


# Geocode an address using Google Maps
def geocode_address(address):
    try:
        geocode_result = gmaps.geocode(address)
        if geocode_result:
            location = geocode_result[0]["geometry"]["location"]
            return (location["lat"], location["lng"])
        else:
            print("Error: No result found for the address.")
            return None
    except Exception as e:
        print(f"Error during geocoding: {e}")
        return None


# Function to get travel time between two points using public transport
def get_travel_time(origin, destination):
    origin_key = f"{origin[0]},{origin[1]}_{destination[0]},{destination[1]}"
    if origin_key in travel_time_cache:
        return travel_time_cache[origin_key]
    try:
        result = gmaps.distance_matrix(origin, destination, mode="transit")
        if result["rows"][0]["elements"][0]["status"] == "OK":
            duration = result["rows"][0]["elements"][0]["duration"]["value"]
            travel_time_cache[origin_key] = duration / 60  # Convert to minutes
            save_cache()
            return travel_time_cache[origin_key]
        else:
            return None
    except Exception as e:
        return None


# Generate a grid of points covering the bounding box for all addresses
def generate_grid(lat_min, lat_max, lon_min, lon_max, step_size):
    lat_range = np.arange(lat_min, lat_max, step_size)
    lon_range = np.arange(lon_min, lon_max, step_size)
    return [(lat, lon) for lat in lat_range for lon in lon_range]


# Load input data (addresses and names) from JSON
input_file = "input_addresses.json"
with open(input_file, "r") as f:
    input_data = json.load(f)

addresses = [entry["address"] for entry in input_data]
names = [entry["name"] for entry in input_data]

# Geocode each friend's address
locations = [
    geocode_address(address)
    for address in addresses
    if geocode_address(address) is not None
]

# Calculate bounding box based on the addresses (min/max latitudes and longitudes)
latitudes = [loc[0] for loc in locations]
longitudes = [loc[1] for loc in locations]
buffer = 0.05  # Increase buffer around the min/max values
lat_min = min(latitudes) - buffer
lat_max = max(latitudes) + buffer
lon_min = min(longitudes) - buffer
lon_max = max(longitudes) + buffer

# Debugging: Print grid boundaries
print(
    f"Grid boundaries: lat_min={lat_min}, lat_max={lat_max}, lon_min={lon_min}, lon_max={lon_max}"
)

# Collect the average travel times for each grid point
grid_points = generate_grid(lat_min, lat_max, lon_min, lon_max, step_size=0.01)
avg_travel_times = []
point_travel_times = []

for point in grid_points:
    total_travel_time = 0
    valid_count = 0
    travel_times_for_point = []
    for location, name in zip(locations, names):
        travel_time = get_travel_time(point, location)
        if travel_time is not None:
            travel_times_for_point.append((name, travel_time))
            total_travel_time += travel_time
            valid_count += 1
    if valid_count > 0:
        avg_travel_times.append(total_travel_time / valid_count)
        point_travel_times.append(travel_times_for_point)
    else:
        avg_travel_times.append(None)
        point_travel_times.append(None)

# Prepare for interpolation
grid_points_array = np.array(grid_points)

# Convert None values in avg_travel_times to np.nan and enforce float dtype
avg_travel_times = np.array(
    [np.nan if x is None else x for x in avg_travel_times], dtype=float
)

# Filter out invalid data (NaNs or None values)
valid_idx = np.isfinite(avg_travel_times)
grid_points_array = grid_points_array[valid_idx]
avg_travel_times = avg_travel_times[valid_idx]
point_travel_times = [
    pt for pt, valid in zip(point_travel_times, valid_idx) if pt is not None
]

# Debugging: Check valid grid points
print(f"Valid grid points after filtering: {len(grid_points_array)}")

# Dynamically determine min and max average travel times for the colormap
min_travel_time = np.min(avg_travel_times)
max_travel_time = np.max(avg_travel_times)

# Debugging: Print min/max travel times
print(f"Min travel time: {min_travel_time}, Max travel time: {max_travel_time}")

# Find the location with the minimum average travel time
min_travel_time_idx = np.argmin(avg_travel_times)
min_travel_location = grid_points_array[min_travel_time_idx]

# Create a base map
center_lat = np.mean([loc[0] for loc in locations + [min_travel_location]])
center_lon = np.mean([loc[1] for loc in locations + [min_travel_location]])
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Create a grid for the bounding box, interpolate using nearest if linear fails
grid_lat, grid_lon = np.mgrid[lat_min:lat_max:200j, lon_min:lon_max:200j]
grid_z = griddata(
    grid_points_array, avg_travel_times, (grid_lat, grid_lon), method="linear"
)

if np.isnan(grid_z).all():
    print("Linear interpolation failed, using nearest neighbor interpolation.")
    grid_z = griddata(
        grid_points_array, avg_travel_times, (grid_lat, grid_lon), method="nearest"
    )

# Smooth the contour transitions
grid_z_smoothed = gaussian_filter(grid_z, sigma=3)

# Define a colormap
colormap = cm.LinearColormap(
    colors=["green", "yellow", "orange", "red"],
    vmin=min_travel_time,
    vmax=max_travel_time,
    caption="Average Travel Time (minutes)",
)

# Add color map legend to the folium map
colormap.add_to(m)

# Plot each cell in the grid on the folium map without grid lines
for i in range(grid_lat.shape[0] - 1):
    for j in range(grid_lon.shape[1] - 1):
        lat_points = [
            grid_lat[i, j],
            grid_lat[i + 1, j],
            grid_lat[i + 1, j + 1],
            grid_lat[i, j + 1],
        ]
        lon_points = [
            grid_lon[i, j],
            grid_lon[i + 1, j],
            grid_lon[i + 1, j + 1],
            grid_lon[i, j + 1],
        ]
        poly_coords = [(lat, lon) for lat, lon in zip(lat_points, lon_points)]
        avg_travel_time = np.mean(grid_z_smoothed[i : i + 2, j : j + 2])
        avg_travel_time = np.clip(
            avg_travel_time, min_travel_time, max_travel_time
        )  # Use actual min/max
        if not np.isnan(avg_travel_time):
            color = colormap(avg_travel_time)
            folium.Polygon(
                locations=poly_coords,
                color=color,  # Set color to match fill color
                weight=0,  # Remove grid lines
                fill=True,
                fill_opacity=0.4,  # Adjust transparency
                popup=f"Average Travel Time: {avg_travel_time:.2f} minutes",
            ).add_to(m)

# Add markers for input addresses
for name, location in zip(names, locations):
    folium.Marker(
        location=[location[0], location[1]],
        popup=f"Name: {name}<br>Address: {location}",
        icon=folium.Icon(color="blue", icon="home"),
    ).add_to(m)

# Add marker at the minimum travel time location
folium.Marker(
    location=[min_travel_location[0], min_travel_location[1]],
    popup=f"Best Meetup Spot! Average Travel Time: {min_travel_time:.2f} minutes",
    icon=folium.Icon(color="green", icon="star"),
).add_to(m)

# Add contour overlay
fig, ax = plt.subplots()
contour = ax.contour(grid_lon, grid_lat, grid_z_smoothed, levels=10, cmap="cool")

# Convert contour to GeoJSON format
geojson_contour = geojsoncontour.contour_to_geojson(
    contour=contour, min_angle_deg=3.0, ndigits=5
)

# Add contour overlay to the map
folium.GeoJson(geojson_contour, name="contour").add_to(m)

# Save the map
m.save("map.html")

print(
    "Average travel time heat map with individual times and address markers has been created and saved."
)
