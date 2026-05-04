from shapely import geometry
import geohash
#from https://blog.tafkas.net/2018/09/28/creating-a-grid-based-on-geohashes/

def build_geohash_box(current_geohash):
    """Returns a GeoJSON Polygon for a given geohash

    :param current_geohash: a geohash
    :return: a list representation of th polygon
    """

    b = geohash.bbox(current_geohash)
    polygon = [(b['w'], b['s']), (b['w'], b['n']), (b['e'], b['n']), (b['e'], b['s'],), (b['w'], b['s'])]
    return polygon

def compute_geohash_tiles(bbox_coordinates,GEOHASH_PRECISION=6):
    """Computes all geohash tile in the given bounding box

    :param bbox_coordinates: the bounding box coordinates of the geohashes
    :return: a list of geohashes
    """

    checked_geohashes = set()
    geohash_stack = set()
    geohashes = []
    # get center of bounding box, assuming the earth is flat ;)
    center_latitude = (bbox_coordinates[0] + bbox_coordinates[2]) / 2
    center_longitude = (bbox_coordinates[1] + bbox_coordinates[3]) / 2

    center_geohash = geohash.encode(center_latitude, center_longitude, precision=GEOHASH_PRECISION)
    geohashes.append(center_geohash)
    geohash_stack.add(center_geohash)
    checked_geohashes.add(center_geohash)
    while len(geohash_stack) > 0:
        current_geohash = geohash_stack.pop()
        neighbors = geohash.neighbors(current_geohash)
        for neighbor in neighbors:
            if neighbor not in checked_geohashes and is_geohash_in_bounding_box(neighbor, bbox_coordinates):
                geohashes.append(neighbor)
                geohash_stack.add(neighbor)
                checked_geohashes.add(neighbor)
    return geohashes

def compute_geohash_tiles_from_polygon(polygon,GEOHASH_PRECISION=6):
    """Computes all hex tile in the given polygon

    :param polygon: the polygon
    :return: a list of geohashes
    """

    checked_geohashes = set()
    geohash_stack = set()
    geohashes = []
    # get center of bounding, assuming the earth is flat ;)
    center_latitude = polygon.centroid.coords[0][1]
    center_longitude = polygon.centroid.coords[0][0]

    center_geohash = geohash.encode(center_latitude, center_longitude, precision=GEOHASH_PRECISION)
    geohashes.append(center_geohash)
    geohash_stack.add(center_geohash)
    checked_geohashes.add(center_geohash)
    while len(geohash_stack) > 0:
        current_geohash = geohash_stack.pop()
        neighbors = geohash.neighbors(current_geohash)
        for neighbor in neighbors:
            point = geometry.Point(geohash.decode(neighbor)[::-1])
            if neighbor not in checked_geohashes and polygon.contains(point):
                geohashes.append(neighbor)
                geohash_stack.add(neighbor)
                checked_geohashes.add(neighbor)
    return geohashes

