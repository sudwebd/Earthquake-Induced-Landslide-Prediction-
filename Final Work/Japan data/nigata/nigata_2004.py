#
# Apr 4, 2019
# read slope data along with litho , veg and earthquake data
# 
import ogr
import osgeo
import math
from osgeo import gdal,osr,ogr

# shapefile
drv = ogr.GetDriverByName('ESRI Shapefile')
ds_in = drv.Open("/home/btpbatch3/Downloads/shp_files/nigata_Sekiguchi_and_Sato_2006/Sekiguchi_and_Sato_2006.shp")    #Get the contents of the shape file
lyr_in = ds_in.GetLayer(0)    #Get the shape file's first layer

# soil type
geo = gdal.Open('/home/btpbatch3/Downloads/soiltype_hartmann-moosdorf_2012/glim_wgs84_0point5deg.txt.asc')
arr = geo.ReadAsArray()


#slope
slope_geo = gdal.Open('/home/btpbatch3/Downloads/final_slope_data/slope_deg/w001001.adf')
band = slope_geo.GetRasterBand(1)
slope_arr = band.ReadAsArray()
geotransform = slope_geo.GetGeoTransform() 
originX = geotransform[0]
originY = geotransform[3]
pixelWidth = geotransform[1]
pixelHeight = geotransform[5]

# vegetation
vegi_data=gdal.Open('HDF4_EOS:EOS_GRID:"/home/btpbatch3/Downloads/vegetation_MODIS_28th_March_data/MCD12C1.A2017001.006.2018257171411.hdf":MOD12C1:Majority_Land_Cover_Type_1')
vegi_band = vegi_data.GetRasterBand(1)
vegi_transform = vegi_data.GetGeoTransform() 
vegioriginX = vegi_transform[0]
vegioriginY = vegi_transform[3]
vegipixelWidth = vegi_transform[1]
vegipixelHeight = vegi_transform[5]
vegi_arr = vegi_band.ReadAsArray()

# get ref of layer
# geo_ref = lyr_in.GetSpatialRef()

# jgd2000
jgd_ref=ogr.osr.SpatialReference()
jgd_ref.ImportFromEPSG(2444)

#point to wgs84
geo_ref=ogr.osr.SpatialReference()
geo_ref.ImportFromEPSG(4326)
jgd_to_geo_ctran=ogr.osr.CoordinateTransformation(jgd_ref,geo_ref)

vegiref=ogr.osr.SpatialReference()
vegiref.ImportFromEPSG(4008)
vegictran=ogr.osr.CoordinateTransformation(jgd_ref, vegiref)

## get subdatasets in hdf file
# print(originX, originY, pixelWidth, pixelHeight)
# vegi_data = gdal.Open('/home/btpbatch3/Downloads/MODIS_28th_March_data/MCD12C1.A2017001.006.2018257171411.hdf')
# subdatasets = vegi_data.GetSubDatasets()

# for fname, name in subdatasets:
#     print (name, end=' ')
#     print ("\t", fname)

def choose_soil_type_from_surround(arr, p, q):
    if arr[p][q]!=-9999:
        return arr[p][q]
    elif arr[p+1][q]!=-9999:
        return arr[p+1][q]
    elif arr[p][q+1]!=-9999:
        return arr[p][q+1]
    elif arr[p+1][q+1]!=-9999:
        return arr[p+1][q+1]
    else:
        return -9999

def check(lon, lat):
    #Transform incoming longitude/latitude to the shapefile's projection
    # [lon,lat,z]=ctran.TransformPoint(lon,lat)
    [loni,lati,z]=jgd_to_geo_ctran.TransformPoint(lon,lat)
    [vegilon, vegilat, z] = vegictran.TransformPoint(lon,lat)
    #Create a point
    pt = ogr.Geometry(ogr.wkbPoint)
    pt.SetPoint_2D(0, loni, lati)

    #Set up a spatial filter such that the only features we see when we
    #loop through "lyr_in" are those which overlap the point defined above
    lyr_in.SetSpatialFilter(pt)

    q = math.floor(2*loni + 360) - 1
    p = math.floor(180 - 2*lati) - 1
	
    #slope
    # [lon_slope,lat_slope,z]=ctran_slope.TransformPoint(loni, lati)
    offx_slope = int( (loni-originX)/pixelWidth )
    offy_slope = int( (lati-originY)/pixelHeight )

    vegioffx_slope = int( (vegilon-vegioriginX)/vegipixelWidth )
    vegioffy_slope = int( (vegilat-vegioriginY)/vegipixelHeight )

    if len(lyr_in)>0:
        try:
            soilType=choose_soil_type_from_surround(arr, p, q)
            if soilType!=-9999:
                print(loni, lati, slope_arr[offy_slope][offx_slope], soilType, vegi_arr[vegioffy_slope][vegioffx_slope],  '1')
        except:
            print('NULL')
            # print(loni, lati, slope_arr[offy_slope][offx_slope], arr[p][q], vegi_arr[vegioffy_slope][vegioffx_slope], "NULL")
    else:
        try:
            soilType=choose_soil_type_from_surround(arr, p, q)
            if soilType!=-9999:
                print(loni, lati, slope_arr[offy_slope][offx_slope], soilType, vegi_arr[vegioffy_slope][vegioffx_slope],  '0')
        except:
            print('NULL')
            # print(loni, lati, slope_arr[offy_slope][offx_slope], arr[p][q], vegi_arr[vegioffy_slope][vegioffx_slope], "NULL")  


# xMin,yMin 138.789,37.2531 : xMax,yMax 138.994,37.4168
gap_distance=250
geo_to_jgd_ctran=ogr.osr.CoordinateTransformation(geo_ref, jgd_ref)

# print(lyr_in.GetFeatureCount())
for i in range(lyr_in.GetFeatureCount()):
    feature=lyr_in.GetFeature(i)
    x,y,z=feature.GetGeometryRef().Centroid().GetPoint()
    [x,y,z]=geo_to_jgd_ctran.TransformPoint(x,y)
    check(x,y)

[xMin,yMin,z]=geo_to_jgd_ctran.TransformPoint(138.789,37.2531)
[xMax,yMax,z]=geo_to_jgd_ctran.TransformPoint(138.994,37.4168)
xMin, yMin=int(xMin), int(yMin)
xMax, yMax=int(xMax), int(yMax)
for x in range(xMin, xMax, gap_distance):
    for y in range(yMin, yMax, gap_distance):
        check(x, y)

