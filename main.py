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
import numpy as np
import matplotlib.pyplot as plt

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
        Function that creates a structured tuple representing a row in a RDD

        Params:
            line - A line from the input file

        Rerturns:
            A Strcutured tuple with 14 positions
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
        Function that creates a Structured Row object representing a Row in a DataFrame

        Params:
            line - A line from the input file

        Returns:
            A Row object representing a row in a Dataframe
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
        Function that filters out empty lines as well as lines that have coordinates as 0.0000 (non relevant points)

        Params:
            line - A line from the input file

        Returns:
            True if the line passed this condition, False otherwise
    """
    splitted_line = line.split(',')

    lon_min = -74.916578
    lon_max = -73.120784
    lat_min = 40.129716
    lat_max = 41.477183

    return (
        len(line) > 0) and \
        (float(splitted_line[6]) != 0) and \
        (float(splitted_line[8]) != 0 and \
        (float(splitted_line[6]) >= lon_min) and \
        (float(splitted_line[6]) <= lon_max) and \
        (float(splitted_line[7]) >= lat_min) and \
        (float(splitted_line[7]) <= lat_max) and \
        (float(splitted_line[8]) >= lon_min) and \
        (float(splitted_line[8]) <= lon_max) and \
        (float(splitted_line[9]) >= lat_min) and \
        (float(splitted_line[9]) <= lat_max)
        )



def estimate_cellid(lat, lon):
    """
        Function that estimates a cell ID given a latitude and longitude based on the coordinates of cell 1.1

        Params:
            lat - Input latitude for which to find the cellID
            lon - Input longitude for which to fin the cellID

        Returns:
            A String such as 'xxx.xxx' representing the ID of the cell
    """
    x0 = -74.913585 #longitude of cell 1.1
    y0 = 41.474937  #latitude of cell 1.1
    s = 500 #500 meters

    delta_x = 0.005986 / 500.0  #Change in longitude coordinates per meter
    delta_y = 0.004491556 /500.0    #Change in latitude coordinates per meter

    cell_x = 1 + math.floor((1/2) + (lon - x0)/(s * delta_x))
    cell_y = 1 + math.floor((1/2) + (y0 - lat)/(s * delta_y))
    
    return f"{cell_x}.{cell_y}"



def create_key_value(structured_tuple):
    """
        Function that from a structured tuple organizes it into a Key-Value formation.
        The key is a tuple containing both the weekday and the hour.
        The value is a dictionary containing only one item, this dictionary is to be merged on the reducer.

        Params:
            structured_tuple - A tuple representing a line of the input file

        Returns:
            A tuple organized into a Key-Value formation
    """

    weekday = convert_to_weekday(structured_tuple[0])
    hour = convert_to_hour(structured_tuple[0])
    route = f"{structured_tuple[11]}-{structured_tuple[12]}"

    return ((weekday, hour), {route: 1})



def custom_reducer(accum, elem):
    """
        Custom function to be used in reduceByKey.
        This function well merge dictionaries counting the number of times each time appears

        Params:
            accum - An accumulator dictionary
            elem - The dictionary of the current iteration

        Returns:
            The accumulator dictionary updated with information obtained by elem
    """

    #store the only existing item inside elem
    key, value = elem.popitem()
    
    if(key in accum): #If accum already has this key, then update its value
        accum[key] += value
    else:   #If accum does not have this key, add it
        accum[key] = value

    return accum



def convert_to_weekday(date):
    """
        Function that converts a date to weekday

        Params:
            date - Unix timestamp formatted date in string form

        Returns:
            A string with the weekday of the input date
    """
    date_obj = dt.strptime(date, '%Y-%m-%d %H:%M:%S')
    return (calendar.day_name[date_obj.weekday()]).lower()



def convert_to_hour(date):
    """
        Function that gets the hour from a date

        Params:
            date - Unix timestamp formatted date in string form

        Returns:
            The hour portion of the input date
    """
    return date[11:13]


def query1():
    try:
        
        #timestamp to mesure the time taken
        time_before = dt.now()

        #read csv file (change this to the full dataset instead of just the sample)
        raw_data = sc.textFile(filename)

        #Filtering out non empty lines and lines that have a pick up or drop off coordinates as 0
        #Also filtering lines that have coordinates that would be mapped to cells with ID greater than 300 and lower de 1
        #These lines are considerer outliers (stated in http://debs.org/debs-2015-grand-challenge-taxi-trips/)
        non_empty_lines = raw_data.filter(lambda line: filter_lines(line))

        #Shaping the rdd rows
        fields = non_empty_lines.map(lambda line : create_row(line))

        # ((weekday, hour), {route})
        organized_lines = fields.map(lambda line : create_key_value(line))

        #Group all values by its key, reducing them acording to custom_reducer
        grouped = organized_lines.reduceByKey(lambda accum, elem: custom_reducer(accum, elem))

        #Sort descendingly the dictionaries present in the values and take only the first 10 elements
        top_routes = grouped.mapValues(lambda route_dict: sorted(route_dict, key = route_dict.get, reverse = True)[:10])

        #Store the retrieved results
        # top_routes.saveAsTextFile("spark_rdd_results/query1")

        for a in top_routes.take(2):
            print(a)

        time_after = dt.now()
        seconds = (time_after - time_before).total_seconds()
        print("Execution time {} seconds".format(seconds))
        sc.stop()
    except:
        traceback.print_exc()
        sc.stop()



def query2():
    try:

        #timestamp to measure the time taken
        time_before = dt.now()

        # convert_to_weekday_udf = udf(lambda pickup_date: convert_to_weekday(pickup_date), StringType())
        convert_to_weekday_udf = udf(lambda pickup_date: convert_to_weekday(pickup_date), StringType())
        convert_to_hour_udf = udf(lambda pickup_date: pickup_date[11:13], StringType())

        #read csv file (change this to the full dataset instead of just the sample)
        raw_data = sc.textFile(filename)

        #Filtering out non-empty lines and lines that have a pick up or drop off coordinates as 0
        non_empty_lines = raw_data.filter(lambda line: filter_lines(line))

        #Shaping the rdd rows
        fields = non_empty_lines.map(lambda line : create_row_df(line))

        #Creating DataFrame
        lines_df = spark.createDataFrame(fields)

        # Get the dropoffs of the last 15 minutes for each cell
        # get the average of the fare
        profit_by_area_15min = lines_df \
            .groupBy(window("dropoff_dt", "900 seconds"), convert_to_weekday_udf("pickup_dt").alias("weekday"), convert_to_hour_udf("pickup_dt").alias("hour"), "pickup_cell") \
            .agg(avg(lines_df.fare_amount + lines_df.tip_amount).alias("median_fare")) \
            .orderBy("median_fare", ascending = False) \
            .select("weekday", "hour", "pickup_cell")


        # empty_taxis = lines_df \
        #     .groupBy(window("dropoff_dt", "900 seconds"), "dropoff_cell") \
        #     .agg(countDistinct("taxi_id").alias("empty_taxis")) \
        #     .select("dropoff_cell", "empty_taxis")

        # TODO: Mudar de average para median no cálculo do lucro
        # TODO: Perceber o que falta nos empty_taxis e multiplicar este resultado pelo profit_by_area_15min (certo ?)

        profit_by_area_15min.show(2)

        profit_by_area_15min.rdd.map(lambda row: ((row.weekday, row.hour), row.pickup_cell)).saveAsTextFile("spark_rdd_results/query2")

        time_after = dt.now()
        seconds = (time_after - time_before).total_seconds()
        print("Execution time {} seconds".format(seconds))

        # sc.stop()
    except:
        traceback.print_exc()
        # sc.stop()



def query3():
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

    #Limits to the longitudes and latitudes, everypoint that isn't between these coordinates is considered an outlier
    lon_min = -74.916578
    lon_max = -73.120784
    lat_min = 40.129716
    lat_max = 41.477183

    filter_data = data.filter(
        (data.pickup_longitude != 0) & \
        (data.pickup_latitude != 0) & \
        (data.dropoff_longitude != 0) & \
        (data.dropoff_latitude != 0) & \
        (data.pickup_longitude <= lon_max) & \
        (data.pickup_longitude >= lon_min) & \
        (data.pickup_latitude >= lat_min) & \
        (data.pickup_latitude <= lat_max) & \
        (data.dropoff_longitude <= lon_max) & \
        (data.dropoff_longitude >= lon_min) & \
        (data.dropoff_latitude >= lat_min) & \
        (data.dropoff_latitude <= lat_max)
        )


    #Define the target columns and output column
    assembler = VectorAssembler(
        inputCols = ["pickup_latitude", "pickup_longitude"],
        outputCol = "features"
        )

    #Transform the data according to the Assembler created above
    data_prepared = assembler.transform(filter_data)

    for i in [5, 31, 75]: #find other k values

        centroids_file = "spark_rdd_results/query3/centroids_"+str(i)+".txt"
        # Write results to file
        f = open(centroids_file,"w+")

        #Instanciate Kmeans class with the given K value
        kmeans = KMeans(k = i)

        #Fit the data
        model = kmeans.fit(data_prepared)

        #Evalute clustering by computing Sum of Square Errors
        sum_square_error = model.computeCost(data_prepared)

        # To get the prototypes
        centers = model.clusterCenters()
        print("Cluster Centers: ")
        for center in centers:
            f.write('{:.8f}{}{:.8f}{}'.format(center[0],",",center[1],"\n"))

        # Close file where centroid positions are stored
        f.close()

        #The lower the Sum of Square errors is it means the closer the points are to the prototypes, this is equivalent to
        #the sum of the square that each person has to walk to nearest taxi stand, so minimizing it would be optimal
        print(f"{i} -> SSE: {sum_square_error}")