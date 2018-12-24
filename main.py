from pyspark.sql import *
from pyspark.sql.types import *
import pyspark
from pyspark.sql import SparkSession
import traceback

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
    splitted_line = line.split(',')
    return Row(pickup_dt = splitted_line[2], dropoff_dt = splitted_line[3], trip_time = splitted_line[4], \
    trip_distance = splitted_line[5], pickup_long = splitted_line[6], pickup_lat = splitted_line[7], \
    dropoff_long = splitted_line[8], dropoff_lat = splitted_line[9], fare_amount = splitted_line[11], tip_amount = splitted_line[14], total_amount = splitted_line[16]) #TODO CONTINUAR

try:

    #read csv file (change this to the full dataset instead of just the sample)
    raw_data = sc.textFile(filename) 

    #Filtering out non empty lines
    non_empty_lines = raw_data.filter(lambda line : len(line) > 0)

    #Creating Schema for dataframe
    fields = non_empty_lines.map(lambda line : create_row(line))

    #Creating DataFrame
    lines_df = spark.createDataFrame(fields)
    lines_df.show(10)

    #TODO create a function to map each coordinate with a cell ID

    sc.stop()
except:
    traceback.print_exc()
    sc.stop()