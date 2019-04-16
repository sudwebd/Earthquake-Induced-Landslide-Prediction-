#
# Apr 4, 2019
# read slope data along with litho , veg and earthquake data
# 
import ogr
import osgeo
import math
from osgeo import gdal,osr,ogr
# from osgeo import ogr
drv = ogr.GetDriverByName('ESRI Shapefile') #We will load a shape file
ds_in = drv.Open("/home/btpbatch3/Downloads/shp_files/kumamoto_shp_files/s_aso_ls0418.shp")    #Get the contents of the shape file
lyr_in = ds_in.GetLayer(0)    #Get the shape file's first layer

geo = gdal.Open('/home/btpbatch3/Downloads/soiltype_hartmann-moosdorf_2012/glim_wgs84_0point5deg.txt.asc')
arr = geo.ReadAsArray()

geo_ref = lyr_in.GetSpatialRef()
# print(geo_ref)

point_ref=ogr.osr.SpatialReference()
point_ref.ImportFromEPSG(2444)
ctran=ogr.osr.CoordinateTransformation(point_ref,geo_ref)

#poin to wgs84
ref2=ogr.osr.SpatialReference()
ref2.ImportFromEPSG(4326)
ctran2=ogr.osr.CoordinateTransformation(point_ref,ref2)

vegiref=ogr.osr.SpatialReference()
vegiref.ImportFromEPSG(4008)
vegictran=ogr.osr.CoordinateTransformation(point_ref, vegiref)

#slope
# slope_geo = gdal.Open('/home/btpbatch3/Downloads/japan_dem_wgs84/japan_dem_wgs84.tif')
slope_geo = gdal.Open('/home/btpbatch3/Downloads/final_slope_data/slope_deg/w001001.adf')
# ds = gdal.Open('/home/btpbatch3/Downloads/gt30h1kas/as_dem.prj')
# prj=ds.GetProjection()
# geo_ref_slope=osr.SpatialReference(wkt=prj)
# #ref2 if 4326
# ctran_slope=ogr.osr.CoordinateTransformation(ref2, geo_ref_slope)
band = slope_geo.GetRasterBand(1)
slope_arr = band.ReadAsArray()

geotransform = slope_geo.GetGeoTransform() 
originX = geotransform[0]
originY = geotransform[3]
pixelWidth = geotransform[1]
pixelHeight = geotransform[5]

# print(originX, originY, pixelWidth, pixelHeight)
# vegi_data = gdal.Open('/home/btpbatch3/Downloads/MODIS_28th_March_data/MCD12C1.A2017001.006.2018257171411.hdf')
# subdatasets = vegi_data.GetSubDatasets()

# for fname, name in subdatasets:
#     print (name, end=' ')
#     print ("\t", fname)

# exit()
vegi_data=gdal.Open('HDF4_EOS:EOS_GRID:"/home/btpbatch3/Downloads/vegetation_MODIS_28th_March_data/MCD12C1.A2017001.006.2018257171411.hdf":MOD12C1:Majority_Land_Cover_Type_1')
vegi_band = vegi_data.GetRasterBand(1)

vegi_transform = vegi_data.GetGeoTransform() 
vegioriginX = vegi_transform[0]
vegioriginY = vegi_transform[3]
vegipixelWidth = vegi_transform[1]
vegipixelHeight = vegi_transform[5]

vegi_arr = vegi_band.ReadAsArray()


def check(lon, lat):
    #Transform incoming longitude/latitude to the shapefile's projection
    [lon,lat,z]=ctran.TransformPoint(lon,lat)
    [loni,lati,z]=ctran2.TransformPoint(lon,lat)
    [vegilon, vegilat, z] = vegictran.TransformPoint(lon,lat)
    #Create a point
    pt = ogr.Geometry(ogr.wkbPoint)
    pt.SetPoint_2D(0, lon, lat)

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
            if(arr[p][q]!=-9999):
                print(loni, lati, slope_arr[offy_slope][offx_slope], arr[p][q], vegi_arr[vegioffy_slope][vegioffx_slope],  '1')
        except:
            pass
            # print(loni, lati, slope_arr[offy_slope][offx_slope], arr[p][q], vegi_arr[vegioffy_slope][vegioffx_slope], "NULL")
    else:
        try:
            if(arr[p][q]!=-9999):
                print(loni, lati, slope_arr[offy_slope][offx_slope], arr[p][q], vegi_arr[vegioffy_slope][vegioffx_slope], '0')
        except:
            pass
            # print(loni, lati, slope_arr[offy_slope][offx_slope], arr[p][q], vegi_arr[vegioffy_slope][vegioffx_slope], "NULL")  


    # if len(lyr_in)>0:
    #     try:
    # 	    if(arr[p][q]!=-9999 and slope_arr[offx_slope][offy_slope] is not None):
    # 		    print(round(loni,4), round(lati,4), arr[p][q],  slope_arr[offx_slope][offy_slope], 1)
    # 	    else:
    # 		    print("NaN")
    #     except:
    #         pass
    # else:
    #     try:
    #         if(arr[p][q]!=-9999 and slope_arr[offx_slope][offy_slope] is not None):
    #             print(round(loni,4), round(lati,4), arr[p][q],  slope_arr[offx_slope][offy_slope], 0)
    #         else:		    
    #             print("NaN")
    #     except:
    #         pass

#xMin,yMin -9169.08,-22501.3 : xMax,yMax 19747.1,-105.061

# print(lyr_in.GetFeatureCount())
for i in range(lyr_in.GetFeatureCount()):
    feature=lyr_in.GetFeature(i)
    x,y,z=feature.GetGeometryRef().Centroid().GetPoint()
    # [x,y,z]=geo_to_jgd_ctran.TransformPoint(x,y)
    check(x,y)

for x in range(-9169, 19747, 250):
    for y in range(-22501, -105, 250):
        check(x, y)
