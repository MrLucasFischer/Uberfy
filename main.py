from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
import pyspark
from pyspark.sql import SparkSession
import traceback
import math

spark = SparkSession.builder.master('local[*]').appName('uberfy').getOrCreate()
sc = spark.sparkContext
filename = "./data/sorted_data.csv"

#For creating the cells https://www.emse.fr/~picard/publications/gillani15debs.pdf

##--------------------------------Data Attributes--------------------------------

# medallion	an md5sum of the identifier of the taxi – vehicle bound #Not to be used
# hack_license	an md5sum of the identifier for the taxi license    #Not to be used
# pickup_datetime	time when the passenger(s) were picked up
# dropoff_datetime	time when the passenger(s) were dropped off
# trip_time_in_secs	duration of the trip
# trip_distance	trip distance in miles
# pickup_longitude	longitude coordinate of the pickup location
# pickup_latitude	latitude coordinate of the pickup location
# dropoff_longitude	longitude coordinate of the drop-off location
# dropoff_latitude	latitude coordinate of the drop-off location
# payment_type	the payment method – credit card or cash    #Not to be used
# fare_amount	fare amount in dollars
# surcharge	surcharge in dollars    #Not to be used
# mta_tax	tax in dollars  #Not to be used
# tip_amount	tip in dollars
# tolls_amount	bridge and tunnel tolls in dollars  #Not to be used
# total_amount	total paid amount in dollars

##------------------------------------------------------------------------------

def create_row(line):
    """
        Add doc
    """
    splitted_line = line.split(',')
    return Row(pickup_dt = splitted_line[2], dropoff_dt = splitted_line[3], trip_time = int(splitted_line[4]), \
    trip_distance = float(splitted_line[5]), pickup_long = float(splitted_line[6]), pickup_lat = float(splitted_line[7]), \
    dropoff_long = float(splitted_line[8]), dropoff_lat = float(splitted_line[9]), fare_amount = float(splitted_line[11]), \
    tip_amount = float(splitted_line[14]), total_amount = float(splitted_line[16]))


def filter_lines(line):
    """
        Add doc
    """
    splitted_line = line.split(',')
    return (len(line) > 0) and (float(splitted_line[6]) != 0) and (float(splitted_line[8]) != 0)


def estimate_cellid(lat, lon):
    """
        Add doc
    """
    x0 = -74.913585 #longitude of cell 1.1
    y0 = 41.474937  #latitude of cell 1.1
    s = 500 #500 meters

    delta_x = 0.005986 / 500.0  #Change in longitude coordinates per meter
    delta_y = 0.004491556 /500.0    #Change in latitude coordinates per meter

    cell_x = 1 + math.floor((1/2) + (lon - x0)/(s * delta_x))
    cell_y = 1 + math.floor((1/2) + (y0 - lat)/(s * delta_y))
    
    return f"Cell {cell_x}.{cell_y}"

try:

    estimate_cellid_udf = udf(lambda lat, lon : estimate_cellid(float(lat), float(lon)), StringType())

    #read csv file (change this to the full dataset instead of just the sample)
    raw_data = sc.textFile(filename)

    #Filtering out non empty lines and lines that have a pick up or drop off coordinates as 0
    non_empty_lines = raw_data.filter(lambda line: filter_lines(line))

    #Creating Schema for dataframe
    fields = non_empty_lines.map(lambda line : create_row(line))

    #Creating DataFrame
    lines_df = spark.createDataFrame(fields)
    
    lines_df.select(
        estimate_cellid_udf("pickup_lat", "pickup_long")
    ).show(100)

    sc.stop()
except:
    traceback.print_exc()
    sc.stop()