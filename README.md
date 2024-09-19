# Meeting Location Finder

**DISCLAIMER**: This code has mostly been generated by AI, so please do not use it to feed AI again -> model collapse.

This Python script helps a group of friends find the optimal meeting location based on average public transit travel times using the Google Maps API. It generates a heat map showing the average travel times for different locations and marks the best spot where the travel time is minimized for all participants.

## Features

- **Geocoding**: Converts addresses into latitude and longitude.
- **Travel Time Calculation**: Fetches public transit travel times between grid points and participants' locations.
- **Heat Map Generation**: Creates a color-coded heat map showing travel times across an area.
- **Best Meeting Spot**: Highlights the location with the minimum average travel time.
- **Interactive Map**: Displays an interactive map with travel time information and markers for input addresses.

## Visualization

![Meeting Location Finder Map](sample.png)
(Random sample locations within Vienna)

- **Heatmap**: The heatmap shown on the map represents the average public transit travel times for various grid points in the city. The color ranges from **green** (shortest travel times) to **red** (longest travel times). In the screenshot, areas with green shading represent optimal meeting locations based on travel times, while yellow, orange, and red areas indicate increasing travel times.
  
- **Markers**:
  - **House icons** represent the addresses of participants. Clicking on one of these markers shows the participant's name and address.
  - The **star marker** identifies the best meeting location, where the average travel time for all participants is the lowest. Clicking on the star marker displays the exact average travel time for this optimal spot.
  
- **Map Interaction**: 
  - Clicking on different regions of the heatmap will reveal the average travel time for that grid point. 
  - Clicking on any marker provides additional details, such as the address or average travel time for that location.
  
## Requirements

- Python 3.x
- A valid [Google Maps API Key](https://developers.google.com/maps/documentation/javascript/get-api-key)
- Required Python libraries: `googlemaps`, `folium`, `numpy`, `branca`, `scipy`, `matplotlib`, `geojsoncontour`

## Installation

1. Clone or download the repository.
2. Install required Python packages:

   ```bash
   pip install googlemaps folium numpy branca scipy matplotlib geojsoncontour
   ```

3. Set up your Google Maps API key as an environment variable. Add this to your terminal configuration file (e.g., `.bashrc`, `.zshrc`, or directly in your terminal):

   ```bash
   export GOOGLE_MAPS_API_KEY='your_api_key_here'
   ```

## Input Addresses

The script reads a list of addresses and corresponding names from a JSON file named `input_addresses.json`. This file should contain the participants' addresses.

Example `input_addresses.json`:

```json
[
    {
        "name": "Alice",
        "address": "123 Main St, Anytown, USA"
    },
    {
        "name": "Bob",
        "address": "456 Oak Ave, Anytown, USA"
    }
]
```

Each entry requires a `"name"` field and an `"address"` field.

## How to Use

1. **Prepare the Input File**: Create or edit `input_addresses.json` with your friends' names and addresses.
2. **Run the Script**:

   In the project directory, run:

   ```bash
   python good_meeting_location_finder.py
   ```

   This will generate a heat map and save it as `map.html` in the same directory.

3. **View the Results**:

   Open `map.html` in any web browser to explore the heat map. The map will:
   - Show colored regions representing average travel times for each grid point.
   - Include markers for each friend's address.
   - Highlight the optimal meeting location (minimized average travel time) with a star marker.

## Example Use Case

This script can be used by a group of friends to determine the most convenient meeting location based on public transit travel times. Each participant provides their address, and the script generates an interactive map where you can visualize travel times and identify the best spot to meet.

## Notes

- Ensure you have enough requests available on your Google Maps API plan, as each address and travel time calculation consumes API requests.
- The result HTML file, `map.html`, is fully interactive. You can hover over regions to see the average travel time and click on the markers to see details about the addresses and the optimal location.

## Troubleshooting

- **API Quota Issues**: If you exceed the Google Maps API request quota, consider upgrading your API plan or caching results to avoid repeated requests.
- **Missing Data**: Make sure all addresses are valid and correctly geocoded. If geocoding or travel time fails, the script will ignore that location and proceed with available data.