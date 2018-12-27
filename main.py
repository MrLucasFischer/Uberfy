from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark import SparkContext
import traceback
import datetime
from datetime import datetime as dt
import calendar
import time
import math
import operator

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
# payment_type	the payment method – credit card or cash    #Not to be used #str
# fare_amount	fare amount in dollars #float
# surcharge	surcharge in dollars    #Not to be used #float
# mta_tax	tax in dollars  #Not to be used #float
# tip_amount	tip in dollars #float
# tolls_amount	bridge and tunnel tolls in dollars  #Not to be used #float
# total_amount	total paid amount in dollars #float

##------------------------------------------------------------------------------

def create_row(line):
    """
        Add doc
    """
    #Field - Array_position

    #pickup_dt - 0      fare_amount - 8
    #dropoff_dt - 1     tip_amount - 9
    #trip_time - 2      total_amount - 10
    #trip_distance - 3  pickup_cell - 11
    #pickup_long - 4    dropoff_cell - 12
    #pickup_lat - 5     taxi_id = 13
    #dropoff_long - 6
    #dropoff_lat - 7
    
    splitted_line = line.split(',')
    return (
        splitted_line[2], splitted_line[3], int(splitted_line[4]), float(splitted_line[5]), float(splitted_line[6]), \
        float(splitted_line[7]), float(splitted_line[8]), float(splitted_line[9]), float(splitted_line[11]), \
        float(splitted_line[14]), float(splitted_line[16]), estimate_cellid(float(splitted_line[7]), float(splitted_line[6])),\
        estimate_cellid(float(splitted_line[9]), float(splitted_line[8])), splitted_line[0]
    )


def create_row_df(line):
    """
        Add doc
    """
    #Field - Array_position

    #pickup_dt - 0      fare_amount - 8
    #dropoff_dt - 1     tip_amount - 9
    #trip_time - 2      total_amount - 10
    #trip_distance - 3  pickup_cell - 11
    #pickup_long - 4    dropoff_cell - 12
    #pickup_lat - 5     taxi_id = 13
    #dropoff_long - 6
    #dropoff_lat - 7
    
    splitted_line = line.split(',')
    return Row(
        pickup_dt = splitted_line[2], dropoff_dt = splitted_line[3], trip_time = int(splitted_line[4]), \
        trip_distance = float(splitted_line[5]), pickup_long = float(splitted_line[6]), pickup_lat = float(splitted_line[7]), \
        dropoff_long = float(splitted_line[8]), dropoff_lat = float(splitted_line[9]), fare_amount = float(splitted_line[11]), \
        tip_amount = float(splitted_line[14]), total_amount = float(splitted_line[16]), pickup_cell = estimate_cellid(float(splitted_line[7]), float(splitted_line[6])), \
        dropoff_cell = estimate_cellid(float(splitted_line[9]), float(splitted_line[8])), taxi_id = splitted_line[0]
        )   


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
    
    return f"{cell_x}.{cell_y}"



def create_key_value(line):
    """
        Add doc
    """

    weekday = convert_to_weekday(line[0])
    hour = convert_to_hour(line[0])
    route = f"{line[11]}-{line[12]}"

    return ((weekday, hour), {route: 1})


def create_key_value_query2(line):
    """
        Add doc
    """
    weekday = convert_to_weekday(line[0])
    hour = convert_to_hour(line[0])
    pass


def custom_reducer(accum, elem):
    key, value = elem.popitem()
    
    if(key in accum):
        accum[key] += value
    else:
        accum[key] = value

    return accum

def convert_to_weekday(date):
    """
        Function that converts a date to weekday
    """
    date_obj = dt.strptime(date, '%Y-%m-%d %H:%M:%S')
    return (calendar.day_name[date_obj.weekday()]).lower()



def convert_to_hour(date):
    """
        Function that gets the hour from a date
    """
    return date[11:13]



def filter_outliers(line):
    """
        Add doc
    """
    pickup_cell_x , pickup_cell_y = line[11].split(".")
    dropoff_cell_x , dropoff_cell_y = line[12].split(".")
    return (float(pickup_cell_x) <= 300) and (float(pickup_cell_y) <= 300) and (float(dropoff_cell_x) <= 300) and (float(dropoff_cell_y) <= 300)

def query1():
    try:

        #read csv file (change this to the full dataset instead of just the sample)
        raw_data = sc.textFile(filename)

        #Filtering out non empty lines and lines that have a pick up or drop off coordinates as 0
        non_empty_lines = raw_data.filter(lambda line: filter_lines(line))

        #Shapping the rdd rows
        fields = non_empty_lines.map(lambda line : create_row(line))

        # Filter out rows that have Cell ID's with 300 in them. They are considered as outliers (stated in http://debs.org/debs-2015-grand-challenge-taxi-trips/)
        filtered_rdd = fields.filter(lambda row: filter_outliers(row))

        # ((weekday, hour), {route})
        organized_lines = filtered_rdd.map(lambda line : create_key_value(line))

        grouped = organized_lines.reduceByKey(lambda accum, elem: custom_reducer(accum, elem))

        top_routes = grouped.mapValues(lambda route_dict: sorted(route_dict, key = route_dict.get, reverse = True)[:10])

        for a in top_routes.take(1000):
            print(a)

        sc.stop()
    except:
        traceback.print_exc()
        sc.stop()



def query2():
    try:

        #read csv file (change this to the full dataset instead of just the sample)
        raw_data = sc.textFile(filename)

        #Filtering out non empty lines and lines that have a pick up or drop off coordinates as 0
        non_empty_lines = raw_data.filter(lambda line: filter_lines(line))

        #Shapping the rdd rows
        fields = non_empty_lines.map(lambda line : create_row_df(line))

        #Creating DataFrame
        lines_df = spark.createDataFrame(fields)

        # Filter out rows that have Cell ID's with 300 in them. They are considered as outliers (stated in http://debs.org/debs-2015-grand-challenge-taxi-trips/)
        filtered_df = lines_df.filter(~((lines_df.pickup_cell.rlike("3\d\d")) | (lines_df.dropoff_cell.rlike("3\d\d"))))

        # Get the dropoffs of the last 15 minutes for each cell
        # get the average of the fare
        profit_by_area_15min = filtered_df \
            .groupBy(window("dropoff_dt", "900 seconds"), "pickup_cell") \
            .agg(avg(filtered_df.fare_amount + filtered_df.tip_amount).alias("median_fare")) \
            .orderBy("median_fare", ascending = False) \
            .select("pickup_cell" ,"median_fare")


        # empty_taxis = filtered_df \
        #     .groupBy(window("dropoff_dt", "900 seconds"), "dropoff_cell") \
        #     .agg(countDistinct("taxi_id").alias("empty_taxis")) \
        #     .select("dropoff_cell", "empty_taxis")

        profit_by_area_15min.show(10)

        sc.stop()
    except:
        traceback.print_exc()
        sc.stop()





def query3():
    """
        Add doc
    """
    schema = StructType([
        StructField("medallion", StringType()),
        StructField("hack_license", StringType()),
        StructField("pickup_datetime", StringType()),
        StructField("dropoff_datetime", StringType()),
        StructField("trip_time_in_secs", IntegerType()),
        StructField("trip_distance", FloatType()),
        StructField("pickup_longitude", FloatType()),
        StructField("pickup_latitude", FloatType()),
        StructField("dropoff_longitude", FloatType()),
        StructField("dropoff_latitude", FloatType()),
        StructField("payment_type", StringType()),
        StructField("fare_amount", FloatType()),
        StructField("surcharge", FloatType()),
        StructField("mta_tax", FloatType()),
        StructField("tip_amount", FloatType()),
        StructField("tolls_amount", FloatType()),
        StructField("total_amount", FloatType())
    ])
    
    # Load and parse the data
    data = spark.read.schema(schema).option("header", "false").csv("./data/sorted_data.csv")

    assembler = VectorAssembler(
        inputCols = ["pickup_latitude", "pickup_longitude"],
        outputCol = "features"
        )

    data_prepared = assembler.transform(data)

    evaluator = ClusteringEvaluator()

    for i in range(11, 52, 4): #find other k values
        kmeans = KMeans(k = i)

        #Fit the data
        model = kmeans.fit(data_prepared)

        #Make predictions on fitted data
        predictions = model.transform(data_prepared)

        #Evaluate clustering by computing Silhouettes score
        silhouette = evaluator.evaluate(predictions)

        #TODO get position of prototype with best silhouette score (probably going to be the last iteration)

        #The closer silhouette score is to 1 means the tighter the points of the same cluster are, and the farther they are from other clusters
        #This is optimal because it means that points will all be close to just one taxi stand (saving unecessary money to create another one)
        print(f"{i} -> {silhouette}")



query2()