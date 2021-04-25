from exif import Image as ExifImage
from math import sin, cos, sqrt, atan2, radians

def decdeg2dms(dd):
    mnt,sec = divmod(dd*3600,60)
    deg,mnt = divmod(mnt,60)
    return deg,mnt,sec

def dms2decdeg(deg, mnt, sec):
    return deg + (mnt/60) + (sec/3600)

def dms2string(deg, mnt, sec, ref):
    return f'{deg}°{mnt}.{sec}\' {ref}'

def decdeg2string(dd):
    deg, mnt, sec = decdeg2dms(dd)
    return dms2string(deg, mnt, sec, '')

def gps_distance (lat1, lon1, lat2, lon2):
    R = 6373.0
    lat1 = radians(52.2296756)
    lon1 = radians(21.0122287)
    lat2 = radians(52.406374)
    lon2 = radians(16.9251681)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

def get_exif_tags(file_bytes):
    img = ExifImage(file_bytes)
    tags = {}
    if img.has_exif:
        tags['Width'] = img.get("pixel_x_dimension")
        tags['Height'] = img.get("pixel_y_dimension")
        lat = img.get("gps_latitude")
        lon = img.get("gps_longitude")
        if lat != None and lon != None:
            tags['GPS Latitude'] = dms2decdeg(lat[0], lat[1], lat[2])
            tags['GPS Longitude'] = dms2decdeg(lon[0], lat[1], lat[2])
        tags['Date'] = img.get("datetime")
    else:
        raise ValueError("No exif tags")
    return tags


def get_construction_site(con_sites, lat, lon):
    best_con_site = []
    best_distance = 10000000000
    for option, gps in con_sites.items():
        distance = gps_distance(lat, lon, gps[0], gps[1])
        if distance < best_distance:
            best_distance = distance
            best_con_site.insert(0, option)
        else:
            best_con_site.append(option)

    return best_con_site

