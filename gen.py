import requests
import csv
from geopy.geocoders import Nominatim
from collections import defaultdict
from shapely.geometry import shape, Point

geojson = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "coordinates": [
                    [
                        [8.408112486684217, 49.33373147852433],
                        [8.408112486684217, 49.324152889738286],
                        [8.417882534413963, 49.324152889738286],
                        [8.417882534413963, 49.33373147852433],
                        [8.408112486684217, 49.33373147852433]
                    ]
                ],
                "type": "Polygon"
            }
        }
    ]
}

stadtteil_dict = defaultdict(list)


def is_point_in_area(lat, lon, geojson_obj):
    """Check if a (lat, lon) point is inside the GeoJSON polygon"""
    polygon = shape(geojson_obj['features'][0]['geometry'])
    point = Point(lon, lat)  # Note: Point(lon, lat), not (lat, lon)
    return polygon.contains(point)


def get_coordinates(data):
    nodes = dict((e['id'], e) for e in data['elements'] if e['type'] == 'node')
    ways = list(e for e in data['elements'] if e['type'] == 'way' and 'nodes' in e and e['tags'].get('name', 'Unbekannt') != 'Unbekannt')

    n = 0
    for way in ways:
        street_name = way['tags']['name']
        node_id = way['nodes'][0]
        node = nodes[node_id]
        if node:
            lat, lon = node['lat'], node['lon']
            if is_point_in_area(lat, lon, geojson):
                stadtteil_dict['test'].append((street_name, lat, lon))
                n += 1
                print(n)


def main():
    # Overpass API Endpoint und Abfrage für alle Straßen in Speyer (PLZ 67346)
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = '''
    [out:json];
    area["postal_code"="67346"]->.searchArea;
    way["highway"](area.searchArea);
    out body;
    >;
    out skel qt;
    '''

    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()

    get_coordinates(data)

    csv_file = "speyer_strassenverzeichnis.csv"
    header = ['Stadtteil', 'Straßenname', 'Latitude', 'Longitude']

    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for stadtteil, streets in stadtteil_dict.items():
            sorted_streets = sorted(streets, key=lambda x: (x[1], x[2]))
            for street_name, lat, lon in sorted_streets:
                writer.writerow([stadtteil, street_name, lat, lon])

    print(f"CSV-Datei '{csv_file}' wurde erfolgreich erstellt")


if __name__ == '__main__':
    main()
